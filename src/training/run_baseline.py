"""CLI entry point: train one supervised baseline + optionally tune with Optuna.

Example
-------
    scripts\\run.bat -m src.training.run_baseline --model lr --task reg
    scripts\\run.bat -m src.training.run_baseline --model gbm --task cls --tune --n-trials 50
    scripts\\run.bat -m src.training.run_baseline --model mlp --task reg --tune --n-trials 25

Each invocation writes to ``artifacts/runs/{YYYY-MM-DD}_{model}_{task}_{seed}/``:
    config.json, metrics.json, predictions.parquet, model.{pkl|pt},
    optuna_history.csv (when --tune).
"""
from __future__ import annotations

# Windows DLL note: torch must load BEFORE pandas/numpy/optuna or its
# fbgemm.dll fails with WinError 127 (OpenMP / DLL search-path conflict).
import torch  # noqa: F401  (load order matters; do not remove)

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.supervised import bootstrap_registry, get_model_cls
from src.training import optuna_tuner
from src.training.data import FeatureMatrix, build_xy, load_features, load_splits
from src.training.loops import EvalResult, evaluate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


def _canonical_task(task: str) -> str:
    return {"reg": "regression", "cls": "classification"}.get(task, task)


def _model_registry_key(model: str, task_canon: str) -> str:
    suffix = "reg" if task_canon == "regression" else "cls"
    return f"{model}_{suffix}"


def _run_dir(model: str, task_canon: str, seed: int) -> Path:
    stamp = dt.date.today().isoformat()
    tag = "reg" if task_canon == "regression" else "cls"
    p = RUNS_DIR / f"{stamp}_{model}_{tag}_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _save_predictions(
    path: Path, task_canon: str, val_res: EvalResult, test_res: EvalResult
) -> None:
    frames: list[pd.DataFrame] = []
    for res in (val_res, test_res):
        df = pd.DataFrame(
            {
                "stay_id": res.stay_ids,
                "t_bin": res.t_bins,
                "fold": res.fold,
                "y_true": res.y_true,
                "y_pred": res.y_pred,
            }
        )
        if task_canon == "classification" and res.y_proba is not None:
            for c in range(res.y_proba.shape[1]):
                df[f"proba_{c}"] = res.y_proba[:, c]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)


def _build_final_model(
    model: str, task_canon: str, best_params: dict[str, Any] | None, seed: int
) -> Any:
    key = _model_registry_key(model, task_canon)
    cls = get_model_cls(key)
    best_params = best_params or {}

    if model == "gbm":
        params = dict(best_params)
        params.setdefault("n_estimators", 2000)
        params["random_state"] = seed
        return cls(params=params)
    if model == "mlp":
        from src.models._mlp import MLPConfig

        # Optuna params -> MLPConfig. Hidden sizes come in as separate keys.
        cfg_kwargs: dict[str, Any] = {"random_state": seed}
        hidden: list[int] = []
        for k, v in best_params.items():
            if k == "n_layers":
                continue
            if k.startswith("hidden_"):
                hidden.append(int(v))
                continue
            cfg_kwargs[k] = v
        if hidden:
            cfg_kwargs["hidden_sizes"] = tuple(hidden)
        return cls(config=MLPConfig(**cfg_kwargs))
    if model == "lr":
        if task_canon == "regression":
            return cls(alpha=float(best_params.get("alpha", 1.0)), random_state=seed)
        return cls(C=float(best_params.get("C", 1.0)), random_state=seed)
    if model == "tcn":
        from src.models._tcn import TCNConfig

        cfg_kwargs: dict[str, Any] = {"random_state": seed}
        # Pull TCN-specific channel list out of best_params if present.
        ch: list[int] = []
        for k, v in best_params.items():
            if k == "n_blocks":
                continue
            if k.startswith("ch_"):
                ch.append(int(v))
                continue
            cfg_kwargs[k] = v
        if ch:
            cfg_kwargs["channels"] = tuple(ch)
        return cls(config=TCNConfig(**cfg_kwargs))
    raise ValueError(f"Unknown model {model!r}")


def _summary_cls(metrics: dict[str, Any]) -> str:
    return (
        f"acc={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f} "
        f"top2={metrics['top2_accuracy']:.3f} ECE={metrics['ece']:.3f}"
    )


def _summary_reg(metrics: dict[str, Any]) -> str:
    return (
        f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.3f}"
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run(
    model: str,
    task: str,
    tune: bool = False,
    n_trials: int = 25,
    seed: int = 42,
    flatten: str = "last_bin",
    dataset_version: str = "v1",
) -> Path:
    bootstrap_registry()
    task_canon = _canonical_task(task)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading features + splits (version=%s)", dataset_version)
    features_df = load_features(version=dataset_version)
    splits = load_splits(version=dataset_version)

    folds, scaler = build_xy(
        features_df, splits, task=task_canon, flatten=flatten
    )
    train: FeatureMatrix = folds["train"]
    val: FeatureMatrix = folds["val"]
    test: FeatureMatrix = folds["test"]
    logger.info(
        "Shapes train=%s val=%s test=%s feats=%d",
        train.X.shape, val.X.shape, test.X.shape, train.X.shape[1],
    )

    best_params: dict[str, Any] = {}
    tuning_summary: dict[str, Any] | None = None
    if tune:
        logger.info("Tuning %s (%s) with Optuna, %d trials", model, task_canon, n_trials)
        tuning_summary = optuna_tuner.tune(
            model=model, task=task_canon, train=train, val=val,
            n_trials=n_trials, seed=seed,
        )
        best_params = tuning_summary["best_params"]
        logger.info("Best val objective=%.4f params=%s",
                    tuning_summary["best_value"], best_params)

    logger.info("Fitting final %s (%s)", model, task_canon)
    final = _build_final_model(model, task_canon, best_params, seed)
    val_metrics = final.fit(train.X, train.y, val.X, val.y)
    logger.info("Final val metrics: %s", val_metrics)

    val_res = evaluate(final, val)
    test_res = evaluate(final, test)
    if task_canon == "classification":
        logger.info("Val : %s", _summary_cls(val_res.metrics))
        logger.info("Test: %s", _summary_cls(test_res.metrics))
    else:
        logger.info("Val : %s", _summary_reg(val_res.metrics))
        logger.info("Test: %s", _summary_reg(test_res.metrics))

    out = _run_dir(model, task_canon, seed)
    final.save(out)

    config = {
        "model": model,
        "task": task_canon,
        "seed": seed,
        "tune": bool(tune),
        "n_trials": int(n_trials if tune else 0),
        "best_params": best_params,
        "flatten": flatten,
        "dataset_version": dataset_version,
        "git_sha": _git_sha(),
        "scaler": {
            "mean": scaler.mean.tolist(),
            "std": scaler.std.tolist(),
            "feature_names": scaler.feature_names,
            "flatten": scaler.flatten,
            "task": scaler.task,
            "class_counts": scaler.class_counts,
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=float))

    metrics_payload = {
        "train_final": val_metrics,  # from fit (last epoch / best iter)
        "val": val_res.metrics,
        "test": test_res.metrics,
        "tuning": tuning_summary,
    }
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=float))

    _save_predictions(out / "predictions.parquet", task_canon, val_res, test_res)

    if tune:
        history_path = out / "optuna_history.csv"
        try:
            optuna_tuner.export_history(model, task_canon, history_path)
        except (RuntimeError, ValueError) as exc:  # narrow; optuna raises these
            logger.warning("Could not export Optuna history: %s", exc)

    logger.info("Run artefacts written to %s", out)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a supervised baseline.")
    p.add_argument("--model", required=True, choices=["lr", "gbm", "mlp", "tcn"])
    p.add_argument("--task", required=True, choices=["reg", "cls", "regression", "classification"])
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flatten", choices=["last_bin", "all_bins"], default="last_bin")
    p.add_argument("--dataset-version", default="v1")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        model=args.model,
        task=args.task,
        tune=args.tune,
        n_trials=args.n_trials,
        seed=args.seed,
        flatten=args.flatten,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

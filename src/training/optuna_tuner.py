"""Optuna-based hyperparameter tuning for the supervised baselines.

Storage: sqlite at ``artifacts/optuna/{model}_{task}.db``. Runs are resumable
(`load_if_exists=True`).

Design:
    * Sampler = TPE (Optuna default) with explicit ``seed`` for reproducibility.
    * Pruner = MedianPruner (n_startup_trials=5, n_warmup_steps=10) so bad
      trials die early. LightGBM pruning is done via the sklearn callback.
    * ``single train/val split`` — CV is skipped on purpose (plan §3 Phase A
      "CV: single split" -> saves wall time for the 50-trial budgets).
    * Objective direction: ``maximize`` (we return ``-MAE`` for regression and
      macro-F1 for classification so the same code path works for both).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.data import FeatureMatrix

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTUNA_DIR = PROJECT_ROOT / "artifacts" / "optuna"


def _storage_url(model: str, task: str) -> str:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    db = OPTUNA_DIR / f"{model}_{task}.db"
    # Optuna requires forward slashes in the URL.
    return f"sqlite:///{db.as_posix()}"


def _make_study(
    model: str, task: str, seed: int
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    return optuna.create_study(
        study_name=f"{model}_{task}",
        storage=_storage_url(model, task),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


# -----------------------------------------------------------------------------
# Objective helpers
# -----------------------------------------------------------------------------


def _reg_objective_value(metrics: dict[str, float]) -> float:
    return -float(metrics["val_mae"])


def _cls_objective_value(metrics: dict[str, float]) -> float:
    return float(metrics["val_macro_f1"])


def _objective_value(task: str, metrics: dict[str, float]) -> float:
    return _reg_objective_value(metrics) if task == "regression" else _cls_objective_value(metrics)


# -----------------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------------


def _gbm_param_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 2000,
    }


def tune_lightgbm(
    task: str,
    train: FeatureMatrix,
    val: FeatureMatrix,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    from src.models._gbm import LightGBMClassifier, LightGBMRegressor

    def objective(trial: optuna.Trial) -> float:
        params = _gbm_param_space(trial)
        params["random_state"] = seed
        if task == "regression":
            m = LightGBMRegressor(params=params)
        else:
            m = LightGBMClassifier(params=params)
        metrics = m.fit(train.X, train.y, val.X, val.y, early_stopping_rounds=50)
        return _objective_value(task, metrics)

    study = _make_study("gbm", task, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "storage": _storage_url("gbm", task),
    }


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


def _mlp_param_space(trial: optuna.Trial, task: str) -> dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = tuple(
        trial.suggest_categorical(f"hidden_{i}", [64, 128, 256, 512])
        for i in range(n_layers)
    )
    params: dict[str, Any] = {
        "hidden_sizes": hidden_sizes,
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "max_epochs": 60,
        "patience": 8,
    }
    if task == "classification":
        params["use_focal"] = trial.suggest_categorical("use_focal", [False, True])
        if params["use_focal"]:
            params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    return params


def tune_mlp(
    task: str,
    train: FeatureMatrix,
    val: FeatureMatrix,
    n_trials: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    from src.models._mlp import MLPClassifier, MLPConfig, MLPRegressor

    def objective(trial: optuna.Trial) -> float:
        params = _mlp_param_space(trial, task)
        cfg = MLPConfig(random_state=seed, **params)
        if task == "regression":
            m = MLPRegressor(config=cfg)
        else:
            m = MLPClassifier(config=cfg)
        metrics = m.fit(
            train.X, train.y, val.X, val.y, optuna_trial=trial
        )
        return _objective_value(task, metrics)

    study = _make_study("mlp", task, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "storage": _storage_url("mlp", task),
    }


# -----------------------------------------------------------------------------
# Linear grid (Optuna is overkill; do a small explicit grid)
# -----------------------------------------------------------------------------


def tune_linear(
    task: str,
    train: FeatureMatrix,
    val: FeatureMatrix,
    n_trials: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    from src.models._linear import LogisticRegressionModel, RidgeRegressionModel

    if task == "regression":
        alphas = np.logspace(-3, 3, n_trials)
        best_val = -np.inf
        best_alpha = 1.0
        for a in alphas:
            m = RidgeRegressionModel(alpha=float(a), random_state=seed)
            metrics = m.fit(train.X, train.y, val.X, val.y)
            v = _reg_objective_value(metrics)
            if v > best_val:
                best_val = v
                best_alpha = float(a)
        return {
            "best_value": float(best_val),
            "best_params": {"alpha": best_alpha},
            "n_trials": len(alphas),
            "study_name": "linear_reg",
            "storage": None,
        }
    else:
        Cs = np.logspace(-3, 2, n_trials)
        best_val = -np.inf
        best_C = 1.0
        for c in Cs:
            m = LogisticRegressionModel(C=float(c), random_state=seed)
            metrics = m.fit(train.X, train.y, val.X, val.y)
            v = _cls_objective_value(metrics)
            if v > best_val:
                best_val = v
                best_C = float(c)
        return {
            "best_value": float(best_val),
            "best_params": {"C": best_C},
            "n_trials": len(Cs),
            "study_name": "linear_cls",
            "storage": None,
        }


# -----------------------------------------------------------------------------
# Dispatch + export
# -----------------------------------------------------------------------------


def _tcn_param_space(trial: optuna.Trial, task: str) -> dict[str, Any]:
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    channels = []
    for i in range(n_blocks):
        channels.append(trial.suggest_categorical(f"ch_{i}", [32, 64, 96, 128]))
    params: dict[str, Any] = {
        "n_blocks": n_blocks,
        **{f"ch_{i}": c for i, c in enumerate(channels)},
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "head_hidden": trial.suggest_categorical("head_hidden", [0, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }
    if task == "classification":
        params["use_focal"] = trial.suggest_categorical("use_focal", [False, True])
        if params["use_focal"]:
            params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    return params


def tune_tcn(
    task: str,
    train: FeatureMatrix,
    val: FeatureMatrix,
    n_trials: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    from src.models._tcn import TCNClassifier, TCNConfig, TCNRegressor

    # Infer T and F from the all_bins flat width (T*F columns).
    n_features = len(train.feature_names)  # post-flatten width
    # We know T = 18 from PROJECT_PLAN; original F = width / 18.
    n_timesteps = 18
    base_f = n_features // n_timesteps

    def objective(trial: optuna.Trial) -> float:
        params = _tcn_param_space(trial, task)
        # Pop helper keys not consumed by TCNConfig.
        n_blocks = params.pop("n_blocks")
        ch_keys = [k for k in list(params.keys()) if k.startswith("ch_")]
        channels = tuple(int(params.pop(k)) for k in sorted(ch_keys)[:n_blocks])
        cfg = TCNConfig(
            n_timesteps=n_timesteps,
            n_features=base_f,
            channels=channels,
            random_state=seed,
            **params,
        )
        m = TCNRegressor(config=cfg) if task == "regression" else TCNClassifier(config=cfg)
        metrics = m.fit(train.X, train.y, val.X, val.y, optuna_trial=trial)
        return _objective_value(task, metrics)

    study = _make_study("tcn", task, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "storage": _storage_url("tcn", task),
    }


def tune(
    model: str,
    task: str,
    train: FeatureMatrix,
    val: FeatureMatrix,
    n_trials: int,
    seed: int = 42,
) -> dict[str, Any]:
    if model == "gbm":
        return tune_lightgbm(task, train, val, n_trials=n_trials, seed=seed)
    if model == "mlp":
        return tune_mlp(task, train, val, n_trials=n_trials, seed=seed)
    if model == "tcn":
        return tune_tcn(task, train, val, n_trials=n_trials, seed=seed)
    if model == "lr":
        return tune_linear(task, train, val, n_trials=n_trials, seed=seed)
    raise ValueError(f"Unknown model {model!r}")


def export_history(model: str, task: str, out_csv: Path) -> Path:
    """Dump the Optuna trial history to CSV (for notebooks / reports)."""
    import pandas as pd

    storage = _storage_url(model, task)
    study = optuna.load_study(study_name=f"{model}_{task}", storage=storage)
    df = study.trials_dataframe()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

"""CLI entry point: train + OPE-evaluate one offline RL agent.

Example
-------
    scripts\\run.bat -m src.rl.run_rl --algo bc   --encoder mlp --seed 42
    scripts\\run.bat -m src.rl.run_rl --algo ddqn --encoder mlp --seed 42
    scripts\\run.bat -m src.rl.run_rl --algo dbcq --encoder mlp --seed 42
    scripts\\run.bat -m src.rl.run_rl --algo cql  --encoder mlp --seed 42
    scripts\\run.bat -m src.rl.run_rl --algo cql  --encoder tcn --seed 42

Writes to ``artifacts/runs/{YYYY-MM-DD}_rl_{algo}_{encoder}_{seed}/``:
    config.json, metrics.json, predictions.parquet, model.pt,
    ope_report.md, action_distribution.png, q_value_heatmap.png
"""
from __future__ import annotations

# Windows DLL note: torch must load BEFORE pandas/numpy/optuna or its
# fbgemm.dll fails with WinError 127. Keep this first.
import torch  # noqa: F401

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rl import ope as rl_ope
from src.rl.algorithms import build_agent
from src.rl.algorithms.bc import BehaviorCloning
from src.rl.dataset import (
    N_ACTIONS,
    RewardConfig,
    RLData,
    TrajectoryDataset,
    load_trajectories,
)
from src.rl.training import TrainConfig, resolve_device, seed_everything, train_agent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


# -----------------------------------------------------------------------------
# Git SHA (best effort)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Monitor functions for early stopping
# -----------------------------------------------------------------------------


def _bc_monitor(agent: Any, ds: TrajectoryDataset) -> float:
    """Validation macro accuracy — higher is better."""
    proba = agent.policy_proba(ds.states)
    pred = np.argmax(proba, axis=1)
    return float((pred == ds.actions).mean())


def _q_monitor(agent: Any, ds: TrajectoryDataset, gamma: float) -> float:
    """Val match rate: fraction of timesteps where argmax(policy) == clinician.

    Q values grow monotonically during training and make a useless early-stop
    signal. Match rate is bounded in [0, 1] and actually measures policy quality.
    """
    proba = agent.policy_proba(ds.states)
    pred = np.argmax(proba, axis=1)
    return float((pred == ds.actions).mean())


# -----------------------------------------------------------------------------
# Behaviour policy estimator (shared, used by WIS)
# -----------------------------------------------------------------------------


def fit_behaviour_policy(
    rl_data: RLData,
    encoder: str,
    device: torch.device,
    seed: int,
    max_epochs: int = 30,
    batch_size: int = 512,
) -> BehaviorCloning:
    """Supervised classifier for behaviour pi_b. Used for WIS + dBCQ baseline."""
    bc = BehaviorCloning.from_train(
        in_features=rl_data.train.n_features,
        n_actions=N_ACTIONS,
        y_train=rl_data.train.actions,
        encoder=encoder,
        device=device,
    )
    cfg = TrainConfig(batch_size=batch_size, max_epochs=max_epochs, patience=5, seed=seed)
    train_agent(
        bc,
        rl_data.train,
        rl_data.val,
        cfg,
        monitor_fn=_bc_monitor,
    )
    return bc


# -----------------------------------------------------------------------------
# Plotting (saves PNGs without opening a GUI)
# -----------------------------------------------------------------------------


def _plot_action_distribution(
    clinician: np.ndarray, policy: np.ndarray, out_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping action dist plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(N_ACTIONS)
    c = np.bincount(clinician, minlength=N_ACTIONS) / max(len(clinician), 1)
    p = np.bincount(policy, minlength=N_ACTIONS) / max(len(policy), 1)
    width = 0.4
    ax.bar(x - width / 2, c, width=width, label="Clinician", color="#555555")
    ax.bar(x + width / 2, p, width=width, label="Policy", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i}" for i in range(N_ACTIONS)])
    ax.set_ylabel("Share")
    ax.set_title("Action distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_q_heatmap(q_values: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Aggregate Q by (sampled stay, action).
    sample = q_values[: min(300, len(q_values))]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sample, aspect="auto", cmap="viridis")
    ax.set_xlabel("Action bin")
    ax.set_ylabel("Sample index (first 300)")
    ax.set_title("Q-value heatmap (val fold)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Result writing
# -----------------------------------------------------------------------------


def _run_dir(algo: str, encoder: str, seed: int) -> Path:
    stamp = dt.date.today().isoformat()
    p = RUNS_DIR / f"{stamp}_rl_{algo}_{encoder}_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_predictions(
    path: Path,
    val_ds: TrajectoryDataset,
    test_ds: TrajectoryDataset,
    val_pol: np.ndarray,
    test_pol: np.ndarray,
    val_q: np.ndarray,
    test_q: np.ndarray,
) -> None:
    frames: list[pd.DataFrame] = []
    for fold, ds, pol, q in (
        ("val", val_ds, val_pol, val_q),
        ("test", test_ds, test_pol, test_q),
    ):
        df = pd.DataFrame(
            {
                "stay_id": ds.stay_ids,
                "t_bin": ds.t_bins,
                "fold": fold,
                "behavior_action": ds.actions,
                "policy_action": np.argmax(pol, axis=1),
                "reward": ds.rewards,
                "done": ds.dones,
            }
        )
        for a in range(N_ACTIONS):
            df[f"policy_proba_{a}"] = pol[:, a]
            df[f"q_{a}"] = q[:, a]
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def _ope_report_md(
    algo: str,
    encoder: str,
    seed: int,
    learned: rl_ope.OPEReport,
    behaviour_ref: rl_ope.OPEReport,
    nulls: list[rl_ope.OPEReport],
) -> str:
    lines = [
        f"# OPE Report — {algo.upper()} ({encoder}, seed={seed})",
        "",
        "> **Honesty guard**: All values below are off-policy evaluation (OPE) "
        "estimates. They are NOT evidence of reduced mortality or clinical "
        "benefit. Prospective validation is required before any real-world use.",
        "",
        "## Summary",
        "",
        rl_ope.format_report_table([learned, behaviour_ref, *nulls]),
        "",
        "## Warnings",
        "",
    ]
    for r in [learned, behaviour_ref, *nulls]:
        if r.warnings:
            lines.append(f"- **{r.policy_name}**")
            for w in r.warnings:
                lines.append(f"  - {w}")
    if not any(r.warnings for r in [learned, behaviour_ref, *nulls]):
        lines.append("- none raised.")
    lines.extend(
        [
            "",
            "## Methodology",
            "- Discount gamma = 0.99",
            "- WIS: per-trajectory, IS ratio clip at 1e3, "
            "policy floor 1e-3, behaviour floor 1e-3.",
            "- FQE: 2-layer MLP critic, AdamW, Huber loss, soft target updates.",
            "- Bootstrap CI: 1000 resamples for WIS, 100 for FQE (CI width ~2x wider).",
            "- ESS < 5% -> estimate flagged unreliable.",
            "",
            "## References",
            "- Komorowski et al. 2018, Nat Med — AI Clinician",
            "- Raghu et al. 2017, MLHC — Dueling DDQN for sepsis",
            "- Fujimoto et al. 2019, NeurIPS — (Batch-Constrained) BCQ",
            "- Kumar et al. 2020, NeurIPS — CQL",
            "- Gottesman et al. 2018/19 — OPE pitfalls in healthcare",
            "- Tang & Wiens 2021, CHIL — dual OPE recommendation",
            "",
            "*Prospective RCT evidence for any of these policies does not exist.*",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------


def _build_agent_for_algo(
    algo: str,
    encoder: str,
    rl_data: RLData,
    device: torch.device,
    hparams: dict[str, Any],
) -> Any:
    in_features = rl_data.train.n_features
    common = dict(
        in_features=in_features,
        n_actions=N_ACTIONS,
        encoder=encoder,
        hidden_sizes=tuple(hparams.get("hidden_sizes", (256, 128))),
        dropout=float(hparams.get("dropout", 0.2)),
        device=device,
    )
    if algo == "bc":
        return BehaviorCloning.from_train(
            y_train=rl_data.train.actions,
            learning_rate=float(hparams.get("learning_rate", 1e-3)),
            weight_decay=float(hparams.get("weight_decay", 1e-4)),
            **common,
        )
    if algo == "ddqn":
        return build_agent(
            "ddqn",
            learning_rate=float(hparams.get("learning_rate", 5e-4)),
            weight_decay=float(hparams.get("weight_decay", 1e-4)),
            gamma=float(hparams.get("gamma", 0.99)),
            **common,
        )
    if algo == "dbcq":
        return build_agent(
            "dbcq",
            learning_rate=float(hparams.get("learning_rate", 5e-4)),
            weight_decay=float(hparams.get("weight_decay", 1e-4)),
            gamma=float(hparams.get("gamma", 0.99)),
            bcq_threshold=float(hparams.get("bcq_threshold", 0.3)),
            **common,
        )
    if algo == "cql":
        return build_agent(
            "cql",
            learning_rate=float(hparams.get("learning_rate", 5e-4)),
            weight_decay=float(hparams.get("weight_decay", 1e-4)),
            gamma=float(hparams.get("gamma", 0.99)),
            cql_alpha=float(hparams.get("cql_alpha", 1.0)),
            **common,
        )
    raise ValueError(f"Unknown algo={algo!r}")


def run(
    algo: str,
    encoder: str,
    seed: int = 42,
    dataset_version: str = "v1",
    max_epochs: int = 60,
    batch_size: int = 512,
    patience: int = 8,
    terminal_reward: float = 15.0,
    c_sofa: float = 0.025,
    c_lactate: float = 0.05,
    gamma: float = 0.99,
    cql_alpha: float = 1.0,
    bcq_threshold: float = 0.3,
    n_boot_wis: int = 1000,
    n_boot_fqe: int = 100,
    fqe_iterations: int = 40,
    terminal_target: str = "hospital",
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(seed)
    device = resolve_device(None)
    logger.info("Device: %s", device)

    reward_cfg = RewardConfig(
        c_sofa=c_sofa,
        c_lactate=c_lactate,
        terminal_reward=terminal_reward,
        gamma=gamma,
        terminal_target=terminal_target,
    )
    rl_data = load_trajectories(dataset_version=dataset_version, reward_config=reward_cfg)

    logger.info(
        "train %d stays / %d transitions | val %d / %d | test %d / %d | F=%d",
        rl_data.train.n_stays, rl_data.train.n_transitions,
        rl_data.val.n_stays, rl_data.val.n_transitions,
        rl_data.test.n_stays, rl_data.test.n_transitions,
        rl_data.train.n_features,
    )

    # 1. Behaviour policy estimator (shared).
    logger.info("Fitting behaviour policy estimator (BC classifier)...")
    behav = fit_behaviour_policy(rl_data, encoder=encoder, device=device, seed=seed)

    # Cache behaviour probabilities for val/test.
    behav_val = behav.policy_proba(rl_data.val.states)
    behav_test = behav.policy_proba(rl_data.test.states)

    # 2. Target agent.
    hparams = {
        "hidden_sizes": (256, 128),
        "dropout": 0.2,
        "learning_rate": 1e-3 if algo == "bc" else 5e-4,
        "weight_decay": 1e-4,
        "gamma": gamma,
        "cql_alpha": cql_alpha,
        "bcq_threshold": bcq_threshold,
    }
    agent = _build_agent_for_algo(algo, encoder, rl_data, device, hparams)

    def _monitor(a: Any, ds: TrajectoryDataset) -> float:
        if algo == "bc":
            return _bc_monitor(a, ds)
        return _q_monitor(a, ds, gamma=gamma)

    cfg = TrainConfig(
        batch_size=batch_size, max_epochs=max_epochs, patience=patience, seed=seed
    )
    logger.info("Training %s ...", algo)
    train_summary = train_agent(agent, rl_data.train, rl_data.val, cfg, _monitor)

    # 3. Inference on val/test.
    val_pol = agent.policy_proba(rl_data.val.states)
    test_pol = agent.policy_proba(rl_data.test.states)
    val_q = agent.q_values(rl_data.val.states)
    test_q = agent.q_values(rl_data.test.states)

    # 4. OPE.
    logger.info("Running OPE on val fold ...")
    ope_val_learned = rl_ope.evaluate_policy(
        rl_data.val, val_pol, behav_val, rl_data.val.actions,
        gamma=gamma, policy_name=f"{algo}_learned",
        n_boot_wis=n_boot_wis, n_boot_fqe=n_boot_fqe, fqe_iterations=fqe_iterations,
        seed=seed,
    )
    logger.info("Running OPE on test fold ...")
    ope_test_learned = rl_ope.evaluate_policy(
        rl_data.test, test_pol, behav_test, rl_data.test.actions,
        gamma=gamma, policy_name=f"{algo}_learned",
        n_boot_wis=n_boot_wis, n_boot_fqe=n_boot_fqe, fqe_iterations=fqe_iterations,
        seed=seed,
    )

    # Behaviour policy reference (pi_b = clinician observed actions one-hot).
    behav_ref_val = rl_ope.evaluate_policy(
        rl_data.val,
        policy_proba=np.eye(N_ACTIONS, dtype=np.float32)[rl_data.val.actions],
        behaviour_proba=behav_val,
        clinician_actions=rl_data.val.actions,
        gamma=gamma, policy_name="clinician",
        n_boot_wis=n_boot_wis, n_boot_fqe=n_boot_fqe, fqe_iterations=fqe_iterations,
        seed=seed,
    )

    # Null baselines on val.
    null_reports_val: list[rl_ope.OPEReport] = []
    for kind in ("zero", "uniform", "const_low", "const_mid"):
        pol = rl_ope.null_policy_proba(rl_data.val, kind)
        rep = rl_ope.evaluate_policy(
            rl_data.val, pol, behav_val, rl_data.val.actions,
            gamma=gamma, policy_name=f"null_{kind}",
            n_boot_wis=n_boot_wis, n_boot_fqe=max(50, n_boot_fqe // 2),
            fqe_iterations=max(20, fqe_iterations // 2), seed=seed,
        )
        null_reports_val.append(rep)

    # 5. Persist artefacts.
    out = _run_dir(algo, encoder, seed)
    torch.save(
        {
            "algo": algo,
            "encoder": encoder,
            "seed": seed,
            "hparams": hparams,
            "state_dict": agent.state_snapshot(),
            "in_features": rl_data.train.n_features,
            "n_actions": N_ACTIONS,
        },
        out / "model.pt",
    )

    _save_predictions(
        out / "predictions.parquet",
        rl_data.val, rl_data.test,
        val_pol, test_pol,
        val_q, test_q,
    )

    config = {
        "algo": algo,
        "encoder": encoder,
        "seed": seed,
        "dataset_version": dataset_version,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "reward": reward_cfg.__dict__,
        "hparams": hparams,
        "git_sha": _git_sha(),
        "device": str(device),
        "n_train_stays": rl_data.train.n_stays,
        "n_train_transitions": rl_data.train.n_transitions,
        "feature_names": rl_data.feature_names,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=float))

    def _report_to_dict(r: rl_ope.OPEReport) -> dict[str, Any]:
        return {
            "policy_name": r.policy_name,
            "wis": r.wis,
            "wis_ci_lo": r.wis_ci_lo,
            "wis_ci_hi": r.wis_ci_hi,
            "ess": r.ess,
            "ess_frac": r.ess_frac,
            "fqe": r.fqe,
            "fqe_ci_lo": r.fqe_ci_lo,
            "fqe_ci_hi": r.fqe_ci_hi,
            "match_rate": r.match_rate,
            "ood_rate": r.ood_rate,
            "n_trajectories": r.n_trajectories,
            "n_transitions": r.n_transitions,
            "warnings": list(r.warnings),
        }

    metrics = {
        "train_summary": train_summary,
        "val": {
            "learned": _report_to_dict(ope_val_learned),
            "clinician": _report_to_dict(behav_ref_val),
            "nulls": {r.policy_name: _report_to_dict(r) for r in null_reports_val},
        },
        "test": {
            "learned": _report_to_dict(ope_test_learned),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))

    # Markdown report.
    md = _ope_report_md(
        algo=algo, encoder=encoder, seed=seed,
        learned=ope_val_learned,
        behaviour_ref=behav_ref_val,
        nulls=null_reports_val,
    )
    (out / "ope_report.md").write_text(md, encoding="utf-8")

    # Plots.
    _plot_action_distribution(
        rl_data.val.actions, np.argmax(val_pol, axis=1), out / "action_distribution.png"
    )
    _plot_q_heatmap(val_q, out / "q_value_heatmap.png")

    logger.info("Wrote artefacts to %s", out)
    # Console summary (for easy copy-paste reporting).
    print("\n=== OPE summary (val fold) ===")
    print(rl_ope.format_report_table([ope_val_learned, behav_ref_val, *null_reports_val]))
    print("\n=== OPE summary (test fold) ===")
    print(rl_ope.format_report_table([ope_test_learned]))
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + OPE an offline RL agent.")
    p.add_argument("--algo", required=True, choices=["bc", "ddqn", "dbcq", "cql"])
    p.add_argument("--encoder", default="mlp", choices=["mlp", "tcn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-version", default="v1")
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--terminal-reward", type=float, default=15.0)
    p.add_argument("--c-sofa", type=float, default=0.025)
    p.add_argument("--c-lactate", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--cql-alpha", type=float, default=1.0)
    p.add_argument("--bcq-threshold", type=float, default=0.3)
    p.add_argument("--n-boot-wis", type=int, default=1000)
    p.add_argument("--n-boot-fqe", type=int, default=100)
    p.add_argument("--fqe-iterations", type=int, default=40)
    p.add_argument("--terminal-target", default="hospital", choices=["hospital", "90day"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        algo=args.algo,
        encoder=args.encoder,
        seed=args.seed,
        dataset_version=args.dataset_version,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        terminal_reward=args.terminal_reward,
        c_sofa=args.c_sofa,
        c_lactate=args.c_lactate,
        gamma=args.gamma,
        cql_alpha=args.cql_alpha,
        bcq_threshold=args.bcq_threshold,
        n_boot_wis=args.n_boot_wis,
        n_boot_fqe=args.n_boot_fqe,
        fqe_iterations=args.fqe_iterations,
        terminal_target=args.terminal_target,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

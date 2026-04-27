"""Shared training-loop helpers for offline RL agents.

All agents call :func:`train_agent` with:

    * an ``agent`` object with ``.update(batch)`` and ``.policy_action(states)``.
    * the train / val trajectory datasets.

Training yields a monitor metric per epoch (FQE-est V^pi on val by default for
Q-based agents; BC uses accuracy). Early stopping + LR plateau happen on the
monitor.

Notes:
    * We treat 1 epoch = 1 full pass over train transitions (not stays).
    * Determinism: seed everything; use ``torch.use_deterministic_algorithms``
      best-effort (full determinism only if cuBLAS workspace var is set).
"""
from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from src.rl.dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    lr_patience: int = 4
    lr_factor: float = 0.5
    seed: int = 42
    device: str | None = None  # None -> auto


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism; full determinism requires CUBLAS env var.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def resolve_device(pref: str | None) -> torch.device:
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_minibatches(
    ds: TrajectoryDataset,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Any:
    n = ds.n_transitions
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        chunk = idx[start : start + batch_size]
        yield {
            "state": ds.states[chunk],
            "action": ds.actions[chunk],
            "reward": ds.rewards[chunk],
            "next_state": ds.next_states[chunk],
            "next_action": ds.next_actions[chunk],
            "done": ds.dones[chunk],
        }


def to_tensor_batch(
    batch: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["state"] = torch.from_numpy(batch["state"]).to(device)
    out["action"] = torch.from_numpy(batch["action"].astype(np.int64)).to(device)
    out["reward"] = torch.from_numpy(batch["reward"].astype(np.float32)).to(device)
    out["next_state"] = torch.from_numpy(batch["next_state"]).to(device)
    out["next_action"] = torch.from_numpy(batch["next_action"].astype(np.int64)).to(
        device
    )
    out["done"] = torch.from_numpy(batch["done"].astype(np.float32)).to(device)
    return out


def train_agent(
    agent: Any,
    train_ds: TrajectoryDataset,
    val_ds: TrajectoryDataset,
    config: TrainConfig,
    monitor_fn: Callable[[Any, TrajectoryDataset], float],
) -> dict[str, Any]:
    """Generic training loop with early stopping on ``monitor_fn``.

    Returns dict with history + best monitor + best epoch.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)
    agent.to(device)

    rng = np.random.default_rng(config.seed)

    history: dict[str, list[float]] = {"train_loss": [], "val_monitor": []}
    best_monitor = -math.inf
    best_epoch = -1
    bad_epochs = 0
    best_state = None

    for epoch in range(config.max_epochs):
        agent.train()
        running = 0.0
        n_batches = 0
        for batch in iter_minibatches(train_ds, config.batch_size, rng):
            tb = to_tensor_batch(batch, device)
            loss_info = agent.update(tb)
            running += float(loss_info.get("loss", 0.0))
            n_batches += 1
        epoch_loss = running / max(n_batches, 1)

        agent.eval()
        monitor = float(monitor_fn(agent, val_ds))
        history["train_loss"].append(float(epoch_loss))
        history["val_monitor"].append(float(monitor))

        # LR schedule (optional): agents may own their own scheduler.
        if hasattr(agent, "lr_step"):
            agent.lr_step(monitor)

        if monitor > best_monitor + 1e-6:
            best_monitor = monitor
            best_epoch = epoch
            bad_epochs = 0
            best_state = agent.state_snapshot()
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                logger.info(
                    "Early stop at epoch %d (best=%.4f @ %d)",
                    epoch, best_monitor, best_epoch,
                )
                break

        logger.info(
            "epoch=%d loss=%.4f monitor=%.4f", epoch, epoch_loss, monitor
        )

    if best_state is not None:
        agent.load_snapshot(best_state)
    return {
        "history": history,
        "best_monitor": float(best_monitor),
        "best_epoch": int(best_epoch),
    }

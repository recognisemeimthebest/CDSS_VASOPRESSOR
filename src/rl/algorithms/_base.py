"""Base agent class: handles device management, optimiser, snapshotting.

Each concrete agent subclasses :class:`BaseAgent` and implements ``update``.
Agents return logits for Q (Q-based agents return Q-values directly; BC returns
policy logits so its argmax == greedy policy).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class BaseAgent:
    """Minimal common API for offline-RL agents.

    Subclasses MUST set ``self.net`` (an nn.Module). Q-based agents MUST also
    expose ``self.target_net`` so ``state_snapshot`` captures both.
    """

    net: nn.Module
    target_net: nn.Module | None = None

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    # ---------- module pass-through -------------------------------------------

    def to(self, device: torch.device) -> "BaseAgent":
        self.device = device
        self.net.to(device)
        if self.target_net is not None:
            self.target_net.to(device)
        return self

    def train(self) -> None:
        self.net.train()
        if self.target_net is not None:
            self.target_net.train()

    def eval(self) -> None:
        self.net.eval()
        if self.target_net is not None:
            self.target_net.eval()

    # ---------- snapshotting ---------------------------------------------------

    def state_snapshot(self) -> dict[str, Any]:
        blob: dict[str, Any] = {"net": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}
        if self.target_net is not None:
            blob["target_net"] = {k: v.detach().cpu().clone() for k, v in self.target_net.state_dict().items()}
        return blob

    def load_snapshot(self, blob: dict[str, Any]) -> None:
        self.net.load_state_dict(blob["net"])
        if self.target_net is not None and "target_net" in blob:
            self.target_net.load_state_dict(blob["target_net"])

    # ---------- default lr step -----------------------------------------------

    def lr_step(self, monitor: float) -> None:
        if self.scheduler is not None:
            self.scheduler.step(monitor)

    # ---------- inference helpers ---------------------------------------------

    def _to_tensor(self, states: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(states.astype(np.float32)).to(self.device)

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def policy_proba(self, states: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def policy_action(self, states: np.ndarray) -> np.ndarray:
        proba = self.policy_proba(states)
        return np.argmax(proba, axis=1).astype(np.int64)

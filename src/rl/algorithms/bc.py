"""Behaviour Cloning (BC) — supervised imitation of the clinician.

BC is a baseline *and* the behaviour policy estimator used by dBCQ / WIS.
Train: minimise cross-entropy between logits and observed actions. With the
observed 5-bin class imbalance we use inverse-frequency weighting + an
optional focal-loss knob (gamma defaults to 0, i.e. plain CE + weights).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from src.rl.algorithms._base import BaseAgent
from src.rl.networks import PolicyNetwork


class BehaviorCloning(BaseAgent):
    def __init__(
        self,
        in_features: int,
        n_actions: int,
        encoder: str = "mlp",
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: np.ndarray | None = None,
        focal_gamma: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.n_actions = int(n_actions)
        self.net = PolicyNetwork(
            in_features=in_features,
            n_actions=n_actions,
            encoder=encoder,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)
        self.focal_gamma = float(focal_gamma)

        if class_weights is None:
            w = torch.ones(n_actions, dtype=torch.float32)
        else:
            w = torch.from_numpy(class_weights.astype(np.float32))
        self.class_weights = w.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4
        )

    @classmethod
    def from_train(
        cls,
        in_features: int,
        n_actions: int,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> "BehaviorCloning":
        """Construct BC with class weights computed from y_train."""
        present = np.unique(y_train)
        raw = compute_class_weight("balanced", classes=present, y=y_train)
        w = np.ones(n_actions, dtype=np.float32)
        for c, val in zip(present, raw):
            w[int(c)] = float(val)
        return cls(
            in_features=in_features,
            n_actions=n_actions,
            class_weights=w,
            **kwargs,
        )

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        weight = self.class_weights[target]
        if self.focal_gamma > 0:
            p = gathered.exp()
            focal = (1 - p).pow(self.focal_gamma) * (-gathered)
            return (focal * weight).mean()
        return -(gathered * weight).mean()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        logits = self.net(batch["state"])
        loss = self._loss(logits, batch["action"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        logits = self.net(x)
        return logits.detach().cpu().numpy()

    @torch.no_grad()
    def policy_proba(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

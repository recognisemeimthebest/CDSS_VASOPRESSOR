"""Dueling Double DQN — Raghu et al. 2017 baseline.

This is the offline-unsafe DQN reference; included as an ablation baseline.
OOD actions can get over-optimistic Q; rely on dBCQ/CQL for anything that
feeds downstream decisions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.rl.algorithms._base import BaseAgent
from src.rl.networks import DuelingQNetwork


class DuelingDDQN(BaseAgent):
    def __init__(
        self,
        in_features: int,
        n_actions: int,
        encoder: str = "mlp",
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        gamma: float = 0.99,
        target_tau: float = 0.005,
        target_sync_every: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.target_tau = float(target_tau)
        self.target_sync_every = int(target_sync_every)
        self._step = 0

        self.net = DuelingQNetwork(
            in_features=in_features,
            n_actions=n_actions,
            encoder=encoder,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            in_features=in_features,
            n_actions=n_actions,
            encoder=encoder,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4
        )

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p, p_t in zip(self.net.parameters(), self.target_net.parameters()):
                p_t.data.mul_(1 - self.target_tau).add_(p.data, alpha=self.target_tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        assert self.optimizer is not None
        self.optimizer.zero_grad()

        q_all = self.net(batch["state"])
        q_taken = q_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.net(batch["next_state"])
            next_a = next_q_online.argmax(dim=-1)
            next_q_target = self.target_net(batch["next_state"])
            next_q = next_q_target.gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = batch["reward"] + (1.0 - batch["done"]) * self.gamma * next_q

        loss = torch.nn.functional.smooth_l1_loss(q_taken, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_sync_every == 0:
            self._soft_update()

        return {"loss": float(loss.item()), "q_mean": float(q_taken.mean().item())}

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        return self.net(x).detach().cpu().numpy()

    @torch.no_grad()
    def policy_proba(self, states: np.ndarray) -> np.ndarray:
        q = self.q_values(states).astype(np.float64)
        q -= q.max(axis=1, keepdims=True)
        e = np.exp(q)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

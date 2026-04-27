"""Discrete Batch-Constrained Q-learning (Fujimoto et al. 2019).

BCQ-discrete restricts Q-learning's max to actions whose behaviour-policy
probability exceeds a threshold tau * max-action-probability. This eliminates
OOD-action overestimation in offline batches.

Implementation notes:
    * We train a behaviour classifier G jointly with Q (alternating), which
      matches the original paper (Algorithm 1).
    * tau=0.3 by default (paper suggests 0.3 for medical tasks; Killian 2023
      table).
    * We use Dueling DQN as the Q-network (same as DDQN agent).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.rl.algorithms._base import BaseAgent
from src.rl.networks import DuelingQNetwork, PolicyNetwork


class DiscreteBCQ(BaseAgent):
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
        bcq_threshold: float = 0.3,
        policy_temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.target_tau = float(target_tau)
        self.bcq_threshold = float(bcq_threshold)
        self.policy_temperature = float(policy_temperature)
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

        # Behaviour classifier G(s, a).
        self.behaviour = PolicyNetwork(
            in_features=in_features,
            n_actions=n_actions,
            encoder=encoder,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.net.parameters()) + list(self.behaviour.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4
        )

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p, p_t in zip(self.net.parameters(), self.target_net.parameters()):
                p_t.data.mul_(1 - self.target_tau).add_(p.data, alpha=self.target_tau)

    def _allowed_actions_mask(self, states: torch.Tensor) -> torch.Tensor:
        """1 where G(s, a) / max_a' G(s, a') >= tau, else 0."""
        logits = self.behaviour(states)
        probs = torch.softmax(logits, dim=-1)
        max_p = probs.max(dim=-1, keepdim=True).values
        return (probs / (max_p + 1e-8) >= self.bcq_threshold).float()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        assert self.optimizer is not None
        self.optimizer.zero_grad()

        # Q(s,a) for taken action.
        q_all = self.net(batch["state"])
        q_taken = q_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

        # Behaviour classifier CE loss.
        g_logits = self.behaviour(batch["state"])
        bc_loss = torch.nn.functional.cross_entropy(g_logits, batch["action"])

        with torch.no_grad():
            # Online Q over next state, but masked by allowed actions.
            next_q_online = self.net(batch["next_state"])
            allowed = self._allowed_actions_mask(batch["next_state"])
            masked_q = next_q_online.masked_fill(allowed < 0.5, -1e9)
            next_a = masked_q.argmax(dim=-1)
            next_q_target = self.target_net(batch["next_state"])
            next_q = next_q_target.gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = batch["reward"] + (1.0 - batch["done"]) * self.gamma * next_q

        q_loss = torch.nn.functional.smooth_l1_loss(q_taken, target)
        loss = q_loss + bc_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.net.parameters()) + list(self.behaviour.parameters()), max_norm=5.0
        )
        self.optimizer.step()

        self._step += 1
        self._soft_update()
        return {
            "loss": float(loss.item()),
            "q_loss": float(q_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "q_mean": float(q_taken.mean().item()),
        }

    def train(self) -> None:
        super().train()
        self.behaviour.train()

    def eval(self) -> None:
        super().eval()
        self.behaviour.eval()

    def to(self, device: torch.device) -> "DiscreteBCQ":
        super().to(device)
        self.behaviour.to(device)
        return self

    def state_snapshot(self) -> dict[str, Any]:
        blob = super().state_snapshot()
        blob["behaviour"] = {
            k: v.detach().cpu().clone() for k, v in self.behaviour.state_dict().items()
        }
        return blob

    def load_snapshot(self, blob: dict[str, Any]) -> None:
        super().load_snapshot(blob)
        if "behaviour" in blob:
            self.behaviour.load_state_dict(blob["behaviour"])

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        return self.net(x).detach().cpu().numpy()

    @torch.no_grad()
    def behaviour_proba(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        return torch.softmax(self.behaviour(x), dim=-1).detach().cpu().numpy()

    @torch.no_grad()
    def policy_proba(self, states: np.ndarray) -> np.ndarray:
        x = self._to_tensor(states)
        q = self.net(x)
        allowed = self._allowed_actions_mask(x)
        # Mask disallowed actions with large negative before softmax.
        masked_q = q.masked_fill(allowed < 0.5, -1e9)
        q_np = masked_q.detach().cpu().numpy().astype(np.float64)
        # Temperature softmax — avoids one-hot IS ratio explosion.
        q_np = q_np / self.policy_temperature
        q_np -= q_np.max(axis=1, keepdims=True)
        e = np.exp(q_np)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

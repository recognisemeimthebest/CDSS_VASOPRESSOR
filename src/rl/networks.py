"""Neural network backbones for offline RL.

Reuses the supervised encoders so Phase 4 can stay drop-in compatible with
Phase 3 infrastructure. Two encoders are exposed:

    * ``MLPBackbone``    — per-timestep state -> hidden (like ``MLP_last``).
    * ``TCNBackbone``    — per-timestep output from a dilated conv stack on the
                           whole 18-bin stay sequence. We wrap the sequence
                           logic in :mod:`src.rl.tcn_runtime` (not here) because
                           at RL time each (state, action) pair is typed to a
                           single bin; we therefore implement a per-row TCN
                           equivalent by reshaping history into a 1D window.

The ``MLPBackbone`` path is the default everywhere (matches supervised
benchmark); ``TCNBackbone`` is the ablation encoder. Both expose ``.out_dim``.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.encoders.mlp import MLPEncoder
from src.models.encoders.tcn import TCNEncoder


# -----------------------------------------------------------------------------
# Encoders
# -----------------------------------------------------------------------------


class MLPBackbone(nn.Module):
    """Per-state MLP encoder (no sequence context)."""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(
            in_features=in_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.out_dim = self.encoder.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        return self.encoder(x)


class TCNBackbone(nn.Module):
    """Minimal per-state TCN backbone.

    Because offline-RL transitions are typed per (state, action) pair, we
    replicate the state F features across T steps of a short window and feed
    them through a dilated TCN. This is a lightweight ablation: strictly a
    different parameterisation of the state encoder rather than a full-sequence
    model. For a true sequence encoder, Phase 4.5 would sample *stays* instead
    of transitions — deferred.
    """

    def __init__(
        self,
        in_features: int,
        n_timesteps: int = 4,
        channels: Sequence[int] = (64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_timesteps = int(n_timesteps)
        self.n_features = int(in_features)
        self.encoder = TCNEncoder(
            n_features=self.n_features,
            n_timesteps=self.n_timesteps,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.out_dim = self.encoder.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F). Broadcast across T then flatten to (B, T*F) as the TCN
        # encoder expects.
        b = x.shape[0]
        tiled = x.unsqueeze(1).expand(b, self.n_timesteps, self.n_features)
        flat = tiled.reshape(b, self.n_timesteps * self.n_features)
        return self.encoder(flat)


def build_backbone(
    encoder: str,
    in_features: int,
    dropout: float = 0.2,
    hidden_sizes: Sequence[int] = (256, 128),
) -> nn.Module:
    if encoder == "mlp":
        return MLPBackbone(
            in_features=in_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
    if encoder == "tcn":
        return TCNBackbone(
            in_features=in_features,
            n_timesteps=4,
            channels=(64, 64),
            kernel_size=3,
            dropout=dropout,
        )
    raise ValueError(f"Unknown encoder={encoder!r}")


# -----------------------------------------------------------------------------
# Heads
# -----------------------------------------------------------------------------


class DuelingQHead(nn.Module):
    """Dueling DQN head: V(s) + A(s,a) - mean_a A(s,a).

    Follows Raghu et al. 2017 (MLHC) — separates state value and advantage.
    """

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class SoftmaxPolicyHead(nn.Module):
    """Linear head producing action logits (used by BC and behaviour policy)."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------------------------------------------------------
# Composite networks
# -----------------------------------------------------------------------------


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_actions: int,
        encoder: str = "mlp",
        hidden_sizes: Sequence[int] = (256, 128),
        head_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            encoder, in_features, dropout=dropout, hidden_sizes=hidden_sizes
        )
        self.head = DuelingQHead(self.backbone.out_dim, n_actions, hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_actions: int,
        encoder: str = "mlp",
        hidden_sizes: Sequence[int] = (256, 128),
        head_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            encoder, in_features, dropout=dropout, hidden_sizes=hidden_sizes
        )
        self.head = SoftmaxPolicyHead(self.backbone.out_dim, n_actions, hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

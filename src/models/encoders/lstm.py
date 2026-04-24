"""LSTM encoder stub (Phase 3.5+).

Expected I/O (documented for downstream code):

    forward(x: torch.Tensor) -> torch.Tensor
        x: (batch, T, F)  -- per-timestep feature sequence, already z-scored
        return: (batch, hidden) -- last-hidden-state or attention-pooled summary

Killian et al. (2020, ML4H) show encoder choice drives > 5% performance delta
in clinical RL benchmarks; swap this in for ablation in the RL phase.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_features = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, (h_n, _) = self.lstm(x)
        # Take last timestep's output: (B, out_features)
        return out[:, -1, :]

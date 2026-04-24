"""Simple feed-forward encoder.

Input: a *flattened* feature vector per (stay, t_bin).  Downstream callers are
responsible for choosing between ``last_bin`` (F dims) and ``all_bins`` (T*F
dims) — see :mod:`src.training.data`.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """BatchNorm -> (Linear -> ReLU -> Dropout) x n_layers."""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(in_features))
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_features) -> (B, out_features)
        return self.net(x)

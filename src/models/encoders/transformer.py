"""Transformer encoder stub (Phase 3.5+).

Expected I/O:

    forward(x: torch.Tensor) -> torch.Tensor
        x: (batch, T, F)  -- per-timestep feature sequence
        return: (batch, d_model) -- [CLS]-pooled or mean-pooled summary

Positional encoding omitted from this stub on purpose; the RL phase will decide
between sinusoidal vs learned embeddings based on encoder ablation (ch03).
"""
from __future__ import annotations

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_features = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x)
        h = self.encoder(h)
        # Mean-pool across timesteps.
        return h.mean(dim=1)

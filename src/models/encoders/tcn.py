"""Temporal Convolutional Network (TCN) encoder.

Smaller and faster than LSTM for our setting (T=18 bins, F~80 features).
Causal dilated 1D convolutions + residual connections. Input is the flat
``all_bins`` vector (B, T*F) produced by ``src.training.data._build_one_fold``;
the encoder reshapes it back to (B, T, F) internally so the rest of the
training loop in ``_MLPBase`` does not need to know about sequences.

Reference:
    Bai, Kolter, Koltun (2018) "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling", arXiv:1803.01271
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    """1D conv with left padding so output[t] depends only on input[<=t]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        # Left-pad so we never see future timesteps.
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    """Two causal-conv layers + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        residual = self.proj(x)
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.drop(self.act(self.bn2(self.conv2(h))))
        return self.act(h + residual)


class TCNEncoder(nn.Module):
    """Reshape (B, T*F) -> (B, F, T) -> dilated conv stack -> mean-pool over T."""

    def __init__(
        self,
        n_features: int,           # F
        n_timesteps: int,          # T
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_timesteps = int(n_timesteps)

        blocks: list[nn.Module] = []
        prev = self.n_features
        for i, ch in enumerate(channels):
            blocks.append(
                _TCNBlock(
                    in_channels=prev,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.out_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T*F) flat. Reshape to (B, T, F) then (B, F, T) for Conv1d.
        b = x.shape[0]
        x = x.view(b, self.n_timesteps, self.n_features)
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.blocks(x)     # (B, C_out, T)
        # Global average pool over the time axis -> (B, C_out)
        return h.mean(dim=-1)

"""TCN-based supervised baselines (regression + classification).

Reuses ``_MLPBase`` from :mod:`src.models._mlp` for training-loop boilerplate
(AdamW + ReduceLROnPlateau + early stopping + Optuna pruning hooks).
Only ``_build_net`` differs — we wrap the TCN encoder with the same heads.

The TCN encoder reshapes the flat ``all_bins`` input back to (B, T, F)
internally, so callers must use ``--flatten all_bins`` to feed sequence data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models._mlp import (
    MLPConfig,
    MLPClassifier,
    MLPRegressor,
    _MLPBase,
    _NonNegLinear,
)
from src.models.encoders.tcn import TCNEncoder
from src.models.supervised import register_model


@dataclass
class TCNConfig(MLPConfig):
    """Inherits MLPConfig; replaces hidden_sizes with TCN-specific channels."""

    n_timesteps: int = 18
    n_features: int = 80
    channels: tuple[int, ...] = (64, 64, 64)
    kernel_size: int = 3
    # Extra MLP head on top of TCN pooled output (optional).
    head_hidden: int = 0  # 0 means no extra hidden — direct head off pooled


def _build_tcn(cfg: TCNConfig) -> TCNEncoder:
    return TCNEncoder(
        n_features=cfg.n_features,
        n_timesteps=cfg.n_timesteps,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    )


@register_model("tcn_reg")
class TCNRegressor(MLPRegressor):
    task = "regression"

    def __init__(self, config: TCNConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = TCNConfig(**kwargs)
        # Make sure parent init runs with the right config type.
        self.config = config  # type: ignore[assignment]
        self._net: nn.Module | None = None
        self._device: torch.device | None = None
        self._in_features: int | None = None
        self._n_classes: int | None = None
        self._best_state: dict[str, torch.Tensor] | None = None
        self._history: dict[str, list[float]] = {}

    def _build_net(self) -> nn.Module:
        cfg: TCNConfig = self.config  # type: ignore[assignment]
        enc = _build_tcn(cfg)
        head_in = enc.out_features
        layers: list[nn.Module] = [enc]
        if cfg.head_hidden > 0:
            layers += [
                nn.Linear(head_in, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            head_in = cfg.head_hidden
        layers.append(_NonNegLinear(head_in))
        return nn.Sequential(*layers)


@register_model("tcn_cls")
class TCNClassifier(MLPClassifier):
    task = "classification"

    def __init__(self, config: TCNConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = TCNConfig(**kwargs)
        self.config = config  # type: ignore[assignment]
        self._net: nn.Module | None = None
        self._device: torch.device | None = None
        self._in_features: int | None = None
        self._n_classes: int | None = None
        self._best_state: dict[str, torch.Tensor] | None = None
        self._history: dict[str, list[float]] = {}

    def _build_net(self) -> nn.Module:
        cfg: TCNConfig = self.config  # type: ignore[assignment]
        assert self._n_classes is not None
        enc = _build_tcn(cfg)
        head_in = enc.out_features
        layers: list[nn.Module] = [enc]
        if cfg.head_hidden > 0:
            layers += [
                nn.Linear(head_in, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            head_in = cfg.head_hidden
        layers.append(nn.Linear(head_in, self._n_classes))
        return nn.Sequential(*layers)

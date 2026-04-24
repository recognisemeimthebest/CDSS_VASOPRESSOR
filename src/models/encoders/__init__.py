"""Time-series encoders (MLP / LSTM / Transformer) used by supervised + RL.

Phase 3 ships the MLP encoder used by :mod:`src.models._mlp`. The LSTM /
Transformer variants are stubs that document the intended I/O shape so the RL
phase (or an ablation run) can drop in a concrete implementation without
changing downstream code.
"""
from __future__ import annotations

from src.models.encoders.mlp import MLPEncoder

__all__ = ["MLPEncoder"]

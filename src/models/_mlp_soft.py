"""MLP with Gaussian soft labels for ordinal classification.

Motivation (Diaz et al. CVPR 2019 — ordinal soft labels)
---------------------------------------------------------
Standard CE treats each bin as independent (one-hot).  Ordinal structure is
ignored: predicting bin 1 instead of bin 2 costs the same as predicting
bin 0.  Soft labels replace the one-hot target with a Gaussian centred on the
true bin:

    p_k  =  exp(-|k - y|^2 / (2 σ^2))  (then normalised)

Loss = KL-divergence(soft_target || softmax(logit))
     = -sum_k  p_k * log_softmax_k(logit)

This is equivalent to cross-entropy against the soft distribution — the
gradient naturally penalises predictions far from the true bin more than
adjacent-bin mistakes.

σ (sigma) is the only new hyperparameter.  σ≈0.8 ≈ "adjacent bin tolerated",
σ≈1.5 ≈ "two bins tolerated".  σ→0 recovers hard CE.

Implementation
--------------
Inherits all training machinery from MLPClassifier in _mlp.py.
Only two hooks change:
    _y_to_tensor  → float (N, K) soft distribution
    _loss_fn      → KL-divergence (= soft CE, no reduction needed)
Prediction / argmax / val_metrics remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models._mlp import MLPClassifier, MLPConfig, _resolve_device, _seed_everything
from src.models.supervised import register_model


# ---------------------------------------------------------------------------
# Soft label generation
# ---------------------------------------------------------------------------

def make_soft_labels(
    y: np.ndarray,
    n_classes: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """Convert integer class labels to Gaussian soft distributions.

    y      : (N,) integer class indices
    returns: (N, n_classes) float32 distributions (rows sum to 1)
    """
    bins = np.arange(n_classes, dtype=np.float32)  # (K,)
    y_f = y.astype(np.float32)[:, None]            # (N, 1)
    unnorm = np.exp(-0.5 * ((bins[None, :] - y_f) / sigma) ** 2)  # (N, K)
    return (unnorm / unnorm.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# KL-divergence loss (soft CE)
# ---------------------------------------------------------------------------

class SoftLabelKLLoss(nn.Module):
    """KL( soft_target || softmax(logit) )  =  soft cross-entropy.

    Accepts float targets of shape (N, K) rather than integer indices.
    Optional per-class weight applied to the soft target (same effect as
    weighted CE for hard labels).
    """

    def __init__(self, class_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None else torch.empty(0),
        )

    def forward(self, logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        # log_softmax stable
        log_probs = torch.log_softmax(logits, dim=-1)           # (N, K)
        if self.class_weight.numel() > 0:
            w = self.class_weight.to(logits.device)             # (K,)
            # Weight the contribution of each class.
            loss = -(soft_target * log_probs * w).sum(dim=-1)
        else:
            loss = -(soft_target * log_probs).sum(dim=-1)       # (N,)
        return loss.mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MLPSoftConfig(MLPConfig):
    sigma: float = 1.0      # Gaussian width in bins (1.0 = adjacent bins get ~0.6 share)
    use_focal: bool = False  # disable focal by default for soft labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("mlp_soft")
class MLPSoftClassifier(MLPClassifier):
    """MLPClassifier with Gaussian soft labels."""

    def __init__(self, config: MLPSoftConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = MLPSoftConfig(**kwargs)
        # Call grandparent __init__ (bypassing MLPClassifier) to set self.config.
        super(MLPClassifier, self).__init__(config=config, **kwargs)  # type: ignore[arg-type]
        self.config = config  # type: ignore[assignment]

    # Override: return float (N, K) soft distribution tensor.
    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        assert self._n_classes is not None
        soft = make_soft_labels(y, self._n_classes, sigma=self.config.sigma)  # type: ignore[attr-defined]
        return torch.from_numpy(soft)

    # Override: soft KL loss instead of CE / focal.
    def _loss_fn(self, y_train: np.ndarray) -> nn.Module:
        from sklearn.utils.class_weight import compute_class_weight
        cfg = self.config
        alpha: torch.Tensor | None = None
        if cfg.use_class_weight:
            assert self._n_classes is not None
            classes = np.arange(self._n_classes)
            present = np.unique(y_train)
            raw = compute_class_weight("balanced", classes=present, y=y_train)
            w = np.ones(self._n_classes, dtype=np.float32)
            for c, val in zip(present, raw):
                w[int(c)] = float(val)
            alpha = torch.from_numpy(w)
        return SoftLabelKLLoss(class_weight=alpha)

    # Override: loss receives float soft target — no integer cast needed.
    def _compute_loss(
        self,
        criterion: nn.Module,
        out: torch.Tensor,
        yb: torch.Tensor,  # (N, K) float soft labels
    ) -> torch.Tensor:
        return criterion(out, yb.to(out.device))

    # Prediction uses argmax of logits — unchanged from MLPClassifier.
    # _val_metrics / _logits_to_pred / predict_proba all inherited.

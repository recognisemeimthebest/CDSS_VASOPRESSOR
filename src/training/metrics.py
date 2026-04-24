"""Evaluation metrics used by :mod:`src.training.run_baseline`.

Kept in a separate module so :mod:`src.eval` (or notebooks) can reuse them.
Scope: overall metrics only. Subgroup / fairness analysis lives with the
eval-agent (ch07 §Fairness).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int = 5,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    acc = float((y_pred == y_true).mean())
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    top2 = float("nan")
    ece = float("nan")
    if y_proba is not None and y_proba.ndim == 2:
        top2_idx = np.argsort(-y_proba, axis=1)[:, :2]
        top2 = float(np.mean([yt in row for yt, row in zip(y_true, top2_idx)]))
        ece = expected_calibration_error(y_true, y_proba)
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "top2_accuracy": top2,
        "ece": ece,
        "confusion_matrix": cm,
        "per_class": {
            f"class_{c}": {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in range(n_classes)
        },
    }


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15
) -> float:
    """Simple max-probability ECE (Guo et al. 2017)."""
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)

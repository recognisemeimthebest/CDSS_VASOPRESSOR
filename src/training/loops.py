"""Training-loop helpers used by the CLI in :mod:`src.training.run_baseline`.

Most work actually happens inside each model's ``fit``. These helpers wire up
prediction + metric collection on val and test folds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.supervised import SupervisedModel
from src.training.data import FeatureMatrix
from src.training.metrics import classification_metrics, regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    stay_ids: np.ndarray
    t_bins: np.ndarray
    fold: str


def evaluate(model: SupervisedModel, fm: FeatureMatrix) -> EvalResult:
    y_pred = model.predict(fm.X)
    y_proba: np.ndarray | None = None
    if model.task == "classification":
        try:
            y_proba = model.predict_proba(fm.X)
        except NotImplementedError:
            y_proba = None

    if model.task == "regression":
        metrics = regression_metrics(fm.y, y_pred)
    else:
        metrics = classification_metrics(fm.y, y_pred, y_proba, n_classes=5)

    return EvalResult(
        metrics=metrics,
        y_true=fm.y,
        y_pred=y_pred,
        y_proba=y_proba,
        stay_ids=fm.stay_ids,
        t_bins=fm.t_bins,
        fold=fm.fold,
    )

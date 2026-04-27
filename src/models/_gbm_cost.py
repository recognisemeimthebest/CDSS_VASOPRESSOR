"""LightGBM with cost-sensitive multiclass objective.

Motivation
----------
The standard GBM (macro-F1 0.529) is weakest at class 2 (8.4-20 mcg/min,
F1=0.356) — the boundary zone where dosing decisions are most ambiguous.

Standard cross-entropy treats all misclassifications equally:
    bin2 -> bin1  costs 1  (one step off)
    bin2 -> bin0  costs 1  (two steps off)  ← unfair

Cost-Sensitive objective uses a distance-proportional penalty:
    C[i, j] = (i - j)^2   (quadratic ordinal cost)

so predicting bin0 when truth is bin2 is 4× more penalized than predicting
bin1.  This pushes the model to stay close to the true bin, directly
improving F1 in the boundary region.

Ref: Cost-sensitive learning for imbalanced medical data (AIR 2023)

Implementation note (LightGBM >= 4.0)
--------------------------------------
Custom objective is passed via params["objective"] = callable.
y_pred is 2-D (N, K) — no reshape needed (changed from <4.0 column-major).
Gradient / hessian are also returned as 2-D (N, K).
feval still accepted as argument to lgb.train() but preds are 2-D.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

N_CLASSES = 5


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------

def make_cost_matrix(n_classes: int = N_CLASSES, power: float = 2.0) -> np.ndarray:
    """C[i, j] = |i - j|^power.  Quadratic by default."""
    idx = np.arange(n_classes)
    return np.abs(idx[:, None] - idx[None, :]) ** power


# ---------------------------------------------------------------------------
# Custom objective
# ---------------------------------------------------------------------------

def cost_sensitive_obj(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset,
    cost_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM custom multiclass objective with ordinal cost penalty.

    LightGBM >= 4.0: y_pred is 2-D (N, K) raw scores.
    Returns (gradient, hessian) both as 2-D (N, K).
    """
    if cost_matrix is None:
        cost_matrix = make_cost_matrix()

    y_true = dtrain.get_label().astype(np.int64)

    # y_pred is already (N, K) in LightGBM >= 4.0.
    y_pred_2d = np.asarray(y_pred, dtype=np.float64)
    if y_pred_2d.ndim == 1:
        # Fallback for older LightGBM: column-major flat → (N, K)
        K = cost_matrix.shape[0]
        N = len(y_true)
        y_pred_2d = y_pred_2d.reshape(K, N).T.copy()

    # Numerically stable softmax.
    y_pred_2d = y_pred_2d - y_pred_2d.max(axis=1, keepdims=True)
    e = np.exp(y_pred_2d)
    probs = e / e.sum(axis=1, keepdims=True)  # (N, K)

    # Cost row for each sample's true class: C[y_i, :]
    c_true = cost_matrix[y_true]  # (N, K)

    # Expected cost under policy: E_pi[C] = sum_j p_j * C[y, j]
    expected = (probs * c_true).sum(axis=1, keepdims=True)  # (N, 1)

    # Gradient: dL/df_k = p_k * (C[y, k] - E_pi[C])
    grad = probs * (c_true - expected)  # (N, K)

    # Diagonal hessian approximation: 2 * p_k * (1 - p_k)
    hess = np.maximum(2.0 * probs * (1.0 - probs), 1e-6)  # (N, K)

    return grad.astype(np.float64), hess.astype(np.float64)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_cost_sensitive_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, Any],
    n_estimators: int = 2000,
    early_stopping_rounds: int = 50,
    cost_power: float = 2.0,
    use_class_weight: bool = True,
    seed: int = 42,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train LightGBM with cost-sensitive objective.

    Uses the same Optuna-tuned best_params as the baseline GBM for a fair
    comparison — only the objective changes.
    """
    cost_matrix = make_cost_matrix(N_CLASSES, power=cost_power)

    params: dict[str, Any] = {
        "num_class": N_CLASSES,
        "num_leaves": int(best_params.get("num_leaves", 63)),
        "min_child_samples": int(best_params.get("min_child_samples", 50)),
        "subsample": float(best_params.get("subsample", 0.9)),
        "colsample_bytree": float(best_params.get("colsample_bytree", 0.9)),
        "reg_alpha": float(best_params.get("reg_alpha", 0.0)),
        "reg_lambda": float(best_params.get("reg_lambda", 0.0)),
        "learning_rate": float(best_params.get("learning_rate", 0.05)),
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": -1,
    }

    sample_weight = (
        compute_sample_weight("balanced", y_train) if use_class_weight else None
    )

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # In LightGBM >= 4.0, custom objective is passed via params["objective"].
    def _obj(y_pred: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        return cost_sensitive_obj(y_pred, data, cost_matrix=cost_matrix)

    params["objective"] = _obj

    # Custom eval: multiclass logloss. preds are 2-D (N, K) in LightGBM >= 4.0.
    def _eval_logloss(y_pred: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = data.get_label().astype(np.int64)
        N = len(y_true)
        probs = np.asarray(y_pred, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(N_CLASSES, N).T.copy()
        probs = probs - probs.max(axis=1, keepdims=True)
        e = np.exp(probs)
        probs = e / e.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-9, 1.0)
        logloss = float(-np.log(probs[np.arange(N), y_true]).mean())
        return "custom_logloss", logloss, False  # False = lower is better

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(50),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        feval=_eval_logloss,
        callbacks=callbacks,
    )

    # Predict: raw scores -> softmax -> argmax.
    def _predict(X: np.ndarray) -> np.ndarray:
        raw = booster.predict(X)  # (N, K) — LightGBM returns 2D for custom obj
        raw -= raw.max(axis=1, keepdims=True)
        e = np.exp(raw)
        probs = e / e.sum(axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        raw = booster.predict(X)
        raw -= raw.max(axis=1, keepdims=True)
        e = np.exp(raw)
        return e / e.sum(axis=1, keepdims=True)

    val_preds = _predict(X_val)
    summary: dict[str, Any] = {
        "val_macro_f1": float(f1_score(y_val, val_preds, average="macro")),
        "val_weighted_f1": float(f1_score(y_val, val_preds, average="weighted")),
        "val_accuracy": float((val_preds == y_val).mean()),
        "best_iteration": int(booster.best_iteration),
        "cost_power": cost_power,
    }
    return booster, summary, _predict, _predict_proba

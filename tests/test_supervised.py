"""Tests for Phase 3 supervised baselines.

Pure-logic only — does not touch the database. We construct a small synthetic
feature matrix so the trainers can be exercised end-to-end in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import NE_BIN_EDGES_MCG_MIN
from src.models.supervised import bootstrap_registry, get_model_cls, list_models
from src.training.data import build_xy, feature_columns
from src.training.metrics import (
    classification_metrics,
    expected_calibration_error,
    regression_metrics,
)

bootstrap_registry()


# -----------------------------------------------------------------------------
# Synthetic fixtures
# -----------------------------------------------------------------------------


def _synth_features(n_stays: int = 60, n_bins: int = 18, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stays = np.repeat(np.arange(1000, 1000 + n_stays), n_bins)
    subjects = stays  # 1:1
    t_bins = np.tile(np.arange(n_bins), n_stays)
    n = n_stays * n_bins
    hr = rng.normal(80, 15, n).astype(np.float32)
    sofa = rng.integers(0, 20, n).astype(np.float32)
    ne_now = np.clip(rng.normal(0.05, 0.1, n), 0, None).astype(np.float32)
    noise = rng.normal(0, 0.02, n).astype(np.float32)
    next_dose = np.clip(ne_now + 0.001 * (sofa - 5) + noise, 0, None).astype(np.float32)
    # Action bins via mcg/min (assume weight 80 kg).
    mcg_min = next_dose * 80.0
    bins = np.zeros(n, dtype=np.int8)
    bins[mcg_min > 0] = 1
    bins[mcg_min > NE_BIN_EDGES_MCG_MIN[1]] = 2
    bins[mcg_min > NE_BIN_EDGES_MCG_MIN[2]] = 3
    bins[mcg_min > NE_BIN_EDGES_MCG_MIN[3]] = 4
    return pd.DataFrame(
        {
            "stay_id": stays,
            "subject_id": subjects,
            "t_bin": t_bins,
            "next_ne_dose": next_dose,
            "next_ne_action_bin": bins,
            "ne_dose_now": ne_now,
            "hr_mean": hr,
            "sofa": sofa,
            "vent_flag": rng.integers(0, 2, n).astype(np.int8),
        }
    )


def _synth_splits(df: pd.DataFrame) -> dict[str, np.ndarray]:
    stays = np.sort(df["stay_id"].unique())
    n = len(stays)
    return {
        "train": stays[: int(0.6 * n)].astype(np.int64),
        "val": stays[int(0.6 * n) : int(0.8 * n)].astype(np.int64),
        "test": stays[int(0.8 * n) :].astype(np.int64),
    }


# -----------------------------------------------------------------------------
# Feature building
# -----------------------------------------------------------------------------


def test_feature_columns_excludes_ids_and_actions() -> None:
    df = _synth_features(30)
    cols = feature_columns(df)
    for banned in ("stay_id", "subject_id", "t_bin", "next_ne_dose", "next_ne_action_bin", "ne_dose_now"):
        assert banned not in cols
    assert "hr_mean" in cols and "sofa" in cols


def test_build_xy_last_bin_shape_and_scaling() -> None:
    df = _synth_features(40)
    splits = _synth_splits(df)
    folds, scaler = build_xy(df, splits, task="reg", flatten="last_bin")
    n_bins = 18
    assert folds["train"].X.shape[0] == len(splits["train"]) * n_bins
    # Train fold z-score: mean ~0, std ~1 per column.
    assert np.allclose(folds["train"].X.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(folds["train"].X.std(axis=0), 1, atol=1e-4)
    # Feature names match scaler.
    assert folds["train"].feature_names == scaler.feature_names
    # No NaNs in y.
    assert np.isfinite(folds["train"].y).all()


def test_build_xy_all_bins_one_sample_per_stay() -> None:
    df = _synth_features(20)
    splits = _synth_splits(df)
    folds, scaler = build_xy(df, splits, task="cls", flatten="all_bins")
    assert folds["train"].X.shape[0] == len(splits["train"])
    # 4 features x 18 bins per stay.
    expected_f = len(feature_columns(df)) * 18
    assert folds["train"].X.shape[1] == expected_f
    assert folds["train"].y.dtype == np.int64


# -----------------------------------------------------------------------------
# Model registry + linear sanity check
# -----------------------------------------------------------------------------


def test_registry_contains_expected_models() -> None:
    models = set(list_models())
    assert {"lr_reg", "lr_cls", "gbm_reg", "gbm_cls", "mlp_reg", "mlp_cls"} <= models


def test_ridge_fit_predict_roundtrip(tmp_path) -> None:
    df = _synth_features(40)
    splits = _synth_splits(df)
    folds, _ = build_xy(df, splits, task="reg", flatten="last_bin")
    cls = get_model_cls("lr_reg")
    m = cls(alpha=1.0, random_state=0)
    metrics = m.fit(folds["train"].X, folds["train"].y, folds["val"].X, folds["val"].y)
    assert "val_mae" in metrics and metrics["val_mae"] >= 0
    preds = m.predict(folds["val"].X)
    assert preds.shape == folds["val"].y.shape
    # Persistence
    m.save(tmp_path)
    reloaded = cls.load(tmp_path)
    np.testing.assert_allclose(reloaded.predict(folds["val"].X), preds, atol=1e-6)


def test_logistic_fit_predict(tmp_path) -> None:
    df = _synth_features(40, seed=1)
    splits = _synth_splits(df)
    folds, _ = build_xy(df, splits, task="cls", flatten="last_bin")
    cls = get_model_cls("lr_cls")
    m = cls(C=1.0, random_state=0, max_iter=500)
    metrics = m.fit(folds["train"].X, folds["train"].y, folds["val"].X, folds["val"].y)
    assert 0 <= metrics["val_macro_f1"] <= 1
    proba = m.predict_proba(folds["val"].X)
    assert proba.shape == (folds["val"].X.shape[0], len(np.unique(folds["train"].y)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def test_regression_metrics_finite() -> None:
    y = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    p = y + 0.01
    m = regression_metrics(y, p)
    assert {"mae", "rmse", "r2", "residual_mean", "residual_std"} <= set(m)
    assert m["mae"] > 0


def test_classification_metrics_confusion_matrix_dim() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    p = rng.integers(0, 5, 200)
    proba = rng.dirichlet(np.ones(5), size=200).astype(np.float32)
    m = classification_metrics(y, p, proba, n_classes=5)
    assert len(m["confusion_matrix"]) == 5
    assert len(m["confusion_matrix"][0]) == 5
    assert 0 <= m["top2_accuracy"] <= 1
    assert 0 <= m["ece"] <= 1


def test_ece_zero_when_perfectly_calibrated() -> None:
    # Dirac probability on the correct class => ECE == 0.
    y = np.array([0, 1, 2, 3, 4] * 20)
    proba = np.zeros((len(y), 5), dtype=np.float32)
    proba[np.arange(len(y)), y] = 1.0
    ece = expected_calibration_error(y, proba)
    assert ece == pytest.approx(0.0, abs=1e-6)


# -----------------------------------------------------------------------------
# GPU-marked MLP test
# -----------------------------------------------------------------------------


@pytest.mark.gpu
def test_mlp_cuda_smoke() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from src.models._mlp import MLPConfig, MLPRegressor

    df = _synth_features(30)
    splits = _synth_splits(df)
    folds, _ = build_xy(df, splits, task="reg", flatten="last_bin")
    cfg = MLPConfig(
        hidden_sizes=(32,), dropout=0.1, max_epochs=3, patience=5, batch_size=128,
        device="cuda", random_state=0,
    )
    m = MLPRegressor(config=cfg)
    metrics = m.fit(folds["train"].X, folds["train"].y, folds["val"].X, folds["val"].y)
    assert np.isfinite(metrics["val_mae"])

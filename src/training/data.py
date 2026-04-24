"""Load features + splits and materialise (X, y, stay_id, t_bin) arrays.

The key decisions baked in here (ch04 guards):
    * Train-fold statistics only for z-score; val/test are ``transform`` only.
    * No random row shuffling -- the ``stay_ids`` from ``splits_v1.json`` are the
      source of truth. One subject = one stay in this cohort.
    * ``flatten`` strategy is ``last_bin`` by default: feed a single-timestep
      feature vector to the model. ``all_bins`` concatenates T*F columns for an
      MLP that treats the 72h window as a flat vector. ``aggregate`` reserved
      for future summary-statistic encoders; raises NotImplementedError now.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

FlattenStrategy = Literal["last_bin", "all_bins"]
TaskKey = Literal["reg", "cls", "regression", "classification"]

# Columns that must never enter the feature matrix X.
_ID_COLS = ("stay_id", "subject_id", "t_bin")
_ACTION_COLS = ("next_ne_dose", "next_ne_action_bin", "ne_dose_now")


def _canon_task(task: TaskKey) -> str:
    if task in ("reg", "regression"):
        return "regression"
    if task in ("cls", "classification"):
        return "classification"
    raise ValueError(f"Unknown task {task!r}")


def _target_col(task: str) -> str:
    return "next_ne_dose" if task == "regression" else "next_ne_action_bin"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    X: np.ndarray               # (n_samples, n_features)
    y: np.ndarray               # (n_samples,)
    stay_ids: np.ndarray        # (n_samples,)
    t_bins: np.ndarray          # (n_samples,)
    feature_names: list[str]
    fold: str
    task: str
    flatten: str


@dataclass
class ScalerState:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    flatten: str
    task: str
    # For classification only: observed class frequencies on the train fold.
    class_counts: dict[int, int] | None = None


def load_features(version: str = "v1") -> pd.DataFrame:
    path = DATA_DIR / f"features_{version}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing features parquet at {path}. Run `src.features` first."
        )
    return pd.read_parquet(path)


def load_splits(version: str = "v1") -> dict[str, np.ndarray]:
    path = DATA_DIR / f"splits_{version}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing splits json at {path}. Run `src.splits` first."
        )
    payload = json.loads(path.read_text())
    return {
        "train": np.asarray(payload["train"], dtype=np.int64),
        "val": np.asarray(payload["val"], dtype=np.int64),
        "test": np.asarray(payload["test"], dtype=np.int64),
    }


def feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Numeric, non-leaky feature columns."""
    drop = set(_ID_COLS) | set(_ACTION_COLS)
    out: list[str] = []
    for c in features_df.columns:
        if c in drop:
            continue
        if not pd.api.types.is_numeric_dtype(features_df[c]):
            continue
        out.append(c)
    return out


# -----------------------------------------------------------------------------
# Materialisation
# -----------------------------------------------------------------------------


def _build_one_fold(
    features_df: pd.DataFrame,
    stay_ids: np.ndarray,
    feature_cols: list[str],
    task: str,
    flatten: str,
) -> FeatureMatrix:
    sub = features_df[features_df["stay_id"].isin(stay_ids)].sort_values(
        ["stay_id", "t_bin"]
    ).reset_index(drop=True)

    target = _target_col(task)
    y_series = sub[target].to_numpy()
    if task == "classification":
        y_series = y_series.astype(np.int64)
    else:
        y_series = y_series.astype(np.float32)

    if flatten == "last_bin":
        # Every (stay, t_bin) is a sample.
        X = sub[feature_cols].to_numpy(dtype=np.float32, copy=True)
        stays = sub["stay_id"].to_numpy(dtype=np.int64)
        tbins = sub["t_bin"].to_numpy(dtype=np.int64)
        flat_names = list(feature_cols)
    elif flatten == "all_bins":
        # One sample per stay: concatenate features across all 18 bins.
        grouped = sub.groupby("stay_id", sort=True)
        stays_list: list[int] = []
        rows: list[np.ndarray] = []
        y_rows: list[float | int] = []
        for sid, g in grouped:
            if len(g) == 0:
                continue
            g = g.sort_values("t_bin")
            arr = g[feature_cols].to_numpy(dtype=np.float32)
            rows.append(arr.flatten())
            stays_list.append(int(sid))
            # Use the last bin's next_target as the stay-level label.
            y_rows.append(g[target].iloc[-1])
        X = np.stack(rows, axis=0)
        stays = np.asarray(stays_list, dtype=np.int64)
        tbins = np.full_like(stays, fill_value=-1, dtype=np.int64)
        flat_names = [
            f"t{t}__{name}" for t in range(18) for name in feature_cols
        ][: X.shape[1]]
        y_series = (
            np.asarray(y_rows, dtype=np.int64)
            if task == "classification"
            else np.asarray(y_rows, dtype=np.float32)
        )
    else:
        raise NotImplementedError(f"flatten={flatten!r} not supported")

    return FeatureMatrix(
        X=X, y=y_series, stay_ids=stays, t_bins=tbins,
        feature_names=flat_names, fold="<pending>", task=task, flatten=flatten,
    )


def build_xy(
    features_df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    task: TaskKey,
    flatten: FlattenStrategy = "last_bin",
    feature_cols: list[str] | None = None,
    fit_scaler: bool = True,
    scaler: ScalerState | None = None,
) -> tuple[dict[str, FeatureMatrix], ScalerState]:
    """Build {fold -> FeatureMatrix} from features_df + splits.

    Returns also the ``ScalerState`` fit on the train fold; val/test are
    ``transform``-only (ch04 guard #3: "test fold 정규화 통계 사용" 금지).
    """
    task_c = _canon_task(task)
    feature_cols = feature_cols or feature_columns(features_df)

    folds: dict[str, FeatureMatrix] = {}
    for fold_name, ids in splits.items():
        fm = _build_one_fold(features_df, ids, feature_cols, task_c, flatten)
        fm.fold = fold_name
        folds[fold_name] = fm

    # Fit scaler on train fold, apply to all.
    if scaler is None and fit_scaler:
        train = folds["train"]
        mean = train.X.mean(axis=0)
        std = train.X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        class_counts = None
        if task_c == "classification":
            uniq, cnt = np.unique(train.y, return_counts=True)
            class_counts = {int(k): int(v) for k, v in zip(uniq, cnt)}
        scaler = ScalerState(
            feature_names=train.feature_names,
            mean=mean.astype(np.float32),
            std=std,
            flatten=flatten,
            task=task_c,
            class_counts=class_counts,
        )
    if scaler is None:
        raise ValueError("Provide scaler=... or set fit_scaler=True")

    for fm in folds.values():
        fm.X = (fm.X - scaler.mean) / scaler.std
    return folds, scaler

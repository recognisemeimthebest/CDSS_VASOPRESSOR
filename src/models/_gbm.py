"""LightGBM baselines (regression + multiclass).

Uses sklearn API so that hyperparameter search via Optuna is uniform. Class
imbalance in the 5-bin task is handled with ``class_weight='balanced'`` (sample
weights computed from the train fold; no SMOTE).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.models.supervised import SupervisedModel, register_model


def _default_reg_params() -> dict[str, Any]:
    return {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
    }


def _default_cls_params() -> dict[str, Any]:
    p = _default_reg_params()
    p.update({"objective": "multiclass", "num_class": 5})
    return p


@register_model("gbm_reg")
class LightGBMRegressor(SupervisedModel):
    task = "regression"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = _default_reg_params()
        if params:
            merged.update(params)
        self.params = merged
        self._model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> dict[str, float]:
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = self._model.predict(X_val)
        return {
            "val_mae": float(mean_absolute_error(y_val, preds)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "val_r2": float(r2_score(y_val, preds)),
            "best_iteration": int(self._model.best_iteration_ or self.params["n_estimators"]),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fit() first"
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as fh:
            pickle.dump({"params": self.params, "model": self._model}, fh)

    @classmethod
    def load(cls, path: Path) -> "LightGBMRegressor":
        path = Path(path)
        with (path / "model.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        inst = cls(params=blob["params"])
        inst._model = blob["model"]
        return inst


@register_model("gbm_cls")
class LightGBMClassifier(SupervisedModel):
    task = "classification"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = _default_cls_params()
        if params:
            merged.update(params)
        # sklearn API derives num_class automatically; drop the explicit key.
        merged.pop("num_class", None)
        merged.pop("objective", None)
        self.params = merged
        self._model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 50,
        use_class_weight: bool = True,
        **kwargs: Any,
    ) -> dict[str, float]:
        self._model = lgb.LGBMClassifier(**self.params)
        sample_weight = (
            compute_sample_weight("balanced", y_train) if use_class_weight else None
        )
        self._model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = self._model.predict(X_val)
        return {
            "val_macro_f1": float(f1_score(y_val, preds, average="macro")),
            "val_weighted_f1": float(f1_score(y_val, preds, average="weighted")),
            "val_accuracy": float((preds == y_val).mean()),
            "best_iteration": int(self._model.best_iteration_ or self.params["n_estimators"]),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fit() first"
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fit() first"
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as fh:
            pickle.dump({"params": self.params, "model": self._model}, fh)

    @classmethod
    def load(cls, path: Path) -> "LightGBMClassifier":
        path = Path(path)
        with (path / "model.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        inst = cls(params=blob["params"])
        inst._model = blob["model"]
        return inst

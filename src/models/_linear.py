"""Linear / Logistic baselines (sklearn)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.supervised import SupervisedModel, register_model


@register_model("lr_reg")
class RidgeRegressionModel(SupervisedModel):
    """Ridge regression — sanity-check baseline for continuous NE-equiv dose."""

    task = "regression"

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self._model: Ridge | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self._model.fit(X_train, y_train)
        preds = self._model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        return {
            "val_mae": float(mean_absolute_error(y_val, preds)),
            "val_rmse": rmse,
            "val_r2": float(r2_score(y_val, preds)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fit() first"
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.pkl").open("wb") as fh:
            pickle.dump(
                {"alpha": self.alpha, "random_state": self.random_state, "model": self._model},
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "RidgeRegressionModel":
        path = Path(path)
        with (path / "model.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        inst = cls(alpha=blob["alpha"], random_state=blob["random_state"])
        inst._model = blob["model"]
        return inst


@register_model("lr_cls")
class LogisticRegressionModel(SupervisedModel):
    """LogisticRegression with class_weight='balanced' for the 5-bin task."""

    task = "classification"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self._model: LogisticRegression | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        # multi_class is inferred automatically in sklearn >=1.5 (multinomial for
        # lbfgs with multi-class targets); avoid the deprecated kw to keep
        # warnings clean.
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
            n_jobs=1,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)
        preds = self._model.predict(X_val)
        return {
            "val_macro_f1": float(f1_score(y_val, preds, average="macro")),
            "val_weighted_f1": float(f1_score(y_val, preds, average="weighted")),
            "val_accuracy": float((preds == y_val).mean()),
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
            pickle.dump(
                {
                    "C": self.C,
                    "max_iter": self.max_iter,
                    "class_weight": self.class_weight,
                    "random_state": self.random_state,
                    "model": self._model,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        path = Path(path)
        with (path / "model.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        inst = cls(
            C=blob["C"],
            max_iter=blob["max_iter"],
            class_weight=blob["class_weight"],
            random_state=blob["random_state"],
        )
        inst._model = blob["model"]
        return inst

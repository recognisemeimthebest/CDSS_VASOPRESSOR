"""Common supervised-model API + registry.

Every baseline model (LR/Ridge, LightGBM, MLP) implements :class:`SupervisedModel`
so the training/eval pipeline in :mod:`src.training` only needs one code path.

Contract:
    * ``fit(X_train, y_train, X_val, y_val, **kwargs) -> dict`` returns validation
      metrics. Each model implementation is free to do early stopping internally.
    * ``predict(X) -> np.ndarray`` — continuous for regression, class index for
      classification.
    * ``predict_proba(X) -> np.ndarray`` — soft-max / sigmoid probabilities for
      classification; raises NotImplementedError for regressors.
    * ``save(path)`` / ``load(path)`` — persist to *directory* (one model per
      run folder). Contents are implementation-specific.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np  # imported for type-only hints on subclasses' arrays

TaskType = Literal["regression", "classification"]


class SupervisedModel(ABC):
    """Common API for all supervised baselines."""

    task: TaskType

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Train the model. Returns a dict of validation metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Deterministic predictions (dose for regression, label for classification)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities (classifier only)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement predict_proba")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts under `path` (directory)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "SupervisedModel":
        """Restore a model previously written by :meth:`save`."""


_MODEL_REGISTRY: dict[str, type[SupervisedModel]] = {}


def register_model(
    name: str,
) -> Callable[[type[SupervisedModel]], type[SupervisedModel]]:
    """Decorator: register a model class under a CLI-friendly name."""

    def _wrap(cls: type[SupervisedModel]) -> type[SupervisedModel]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return _wrap


def get_model_cls(name: str) -> type[SupervisedModel]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


# Trigger model registration by importing the implementation modules. The
# imports happen inside a function so that a failed optional dependency
# (e.g. lightgbm) does not break `from src.models import SupervisedModel`.


def bootstrap_registry() -> None:
    # Import concrete models here — each uses @register_model on fit.
    # Import order matters on Windows: torch must load BEFORE lightgbm so its
    # bundled OpenMP runtime claims the slot first; otherwise lightgbm's MKL
    # OMP claims it and torch fails with WinError 127 on fbgemm.dll.
    from src.models import _mlp  # noqa: F401  (torch first!)
    from src.models import _tcn  # noqa: F401  (also torch — keep before lightgbm)
    from src.models import _linear  # noqa: F401
    from src.models import _gbm  # noqa: F401

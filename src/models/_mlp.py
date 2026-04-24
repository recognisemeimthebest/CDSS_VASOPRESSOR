"""PyTorch MLP baseline (regression + classification).

Design:
    * Uses :class:`src.models.encoders.mlp.MLPEncoder` as the backbone.
    * Regression head = single linear with non-negative output (softplus) so
      doses stay in [0, inf).
    * Classification head = 5-class logits + CrossEntropy (optional focal loss).
    * Class imbalance: sample weights (``class_weight='balanced'``) OR focal
      loss (gamma=2 default). Chosen via ``use_focal`` flag.
    * Training: AdamW + ReduceLROnPlateau. Early stopping on val metric.
    * Runs on CUDA when available; falls back to CPU otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.encoders.mlp import MLPEncoder
from src.models.supervised import SupervisedModel, register_model


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------


@dataclass
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10  # early stopping
    lr_patience: int = 5
    lr_factor: float = 0.5
    use_focal: bool = False
    focal_gamma: float = 2.0
    use_class_weight: bool = True
    random_state: int = 42
    device: str | None = None  # "cuda" | "cpu" | None -> auto


# -----------------------------------------------------------------------------
# Focal loss for multi-class
# -----------------------------------------------------------------------------


class MulticlassFocalLoss(nn.Module):
    """Focal loss with optional per-class alpha.

    L = -alpha_c * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        gathered_log = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        gathered_p = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1.0 - gathered_p).pow(self.gamma) * (-gathered_log)
        if self.alpha.numel() > 0:
            a = self.alpha.to(logits.device)[target]
            focal = focal * a
        return focal.mean()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_device(pref: str | None) -> torch.device:
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Base wrapper
# -----------------------------------------------------------------------------


class _MLPBase(SupervisedModel):
    task: str  # set by subclasses

    def __init__(self, config: MLPConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = MLPConfig(**kwargs)
        self.config = config
        self._net: nn.Module | None = None
        self._device: torch.device | None = None
        self._in_features: int | None = None
        self._n_classes: int | None = None
        self._best_state: dict[str, torch.Tensor] | None = None
        self._history: dict[str, list[float]] = {}

    # ---------- architecture ----------

    def _build_head(self) -> nn.Module:  # pragma: no cover - abstract-like
        raise NotImplementedError

    def _build_net(self) -> nn.Module:
        enc = MLPEncoder(
            in_features=self._in_features or 0,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        )
        return nn.Sequential(enc, self._build_head())

    # ---------- training loop ----------

    def _loss_fn(self, y_train: np.ndarray) -> nn.Module:  # pragma: no cover - abstract-like
        raise NotImplementedError

    def _val_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_logits: np.ndarray | None
    ) -> tuple[dict[str, float], float]:
        """Return (metrics_dict, monitor_value). Higher monitor = better."""
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optuna_trial: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        cfg = self.config
        _seed_everything(cfg.random_state)
        device = _resolve_device(cfg.device)
        self._device = device
        self._in_features = X_train.shape[1]

        # Classification subclass sets _n_classes before building net.
        self._maybe_set_n_classes(y_train)

        self._net = self._build_net().to(device)
        criterion = self._loss_fn(y_train).to(device)
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
        )

        train_ds = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            self._y_to_tensor(y_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val.astype(np.float32)),
            self._y_to_tensor(y_val),
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        best_monitor = -math.inf
        best_epoch = -1
        bad_epochs = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_monitor": []}

        for epoch in range(cfg.max_epochs):
            self._net.train()
            running = 0.0
            n = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = self._net(xb)
                loss = self._compute_loss(criterion, out, yb)
                loss.backward()
                optimizer.step()
                bs = xb.size(0)
                running += loss.item() * bs
                n += bs
            epoch_loss = running / max(n, 1)

            metrics, monitor = self._evaluate(val_loader, y_val)
            history["train_loss"].append(float(epoch_loss))
            history["val_monitor"].append(float(monitor))
            scheduler.step(monitor)

            if optuna_trial is not None:
                optuna_trial.report(monitor, epoch)
                # Optuna pruners raise TrialPruned; let it propagate.
                if optuna_trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

            if monitor > best_monitor:
                best_monitor = monitor
                best_epoch = epoch
                bad_epochs = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

        # Restore best weights.
        if self._best_state is not None:
            self._net.load_state_dict(self._best_state)

        self._history = history
        final_metrics, _ = self._evaluate(val_loader, y_val)
        final_metrics["best_epoch"] = int(best_epoch)
        final_metrics["best_monitor"] = float(best_monitor)
        return final_metrics

    # ---------- subclass hooks ----------

    def _maybe_set_n_classes(self, y_train: np.ndarray) -> None:
        pass

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:  # pragma: no cover - abstract-like
        raise NotImplementedError

    def _compute_loss(
        self,
        criterion: nn.Module,
        out: torch.Tensor,
        yb: torch.Tensor,
    ) -> torch.Tensor:
        return criterion(out, yb)

    def _evaluate(
        self, loader: DataLoader, y_true: np.ndarray
    ) -> tuple[dict[str, float], float]:
        assert self._net is not None
        self._net.eval()
        preds: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self._device)  # type: ignore[arg-type]
                out = self._net(xb)
                logits.append(out.detach().cpu().numpy())
                preds.append(self._logits_to_pred(out).detach().cpu().numpy())
        y_pred = np.concatenate(preds, axis=0)
        y_logits = np.concatenate(logits, axis=0)
        return self._val_metrics(y_true, y_pred, y_logits)

    def _logits_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---------- prediction ----------

    def _predict_tensor(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "fit() first"
        assert self._device is not None
        self._net.eval()
        xs = torch.from_numpy(X.astype(np.float32)).to(self._device)
        with torch.no_grad():
            out = self._net(xs)
        return out.detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_tensor(X)
        return self._post_predict(raw)

    def _post_predict(self, raw: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract-like
        raise NotImplementedError

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._net is not None, "fit() first"
        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self._net.state_dict(),
                "in_features": self._in_features,
                "n_classes": self._n_classes,
                "history": self._history,
                "task": self.task,
            },
            path / "model.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "_MLPBase":
        path = Path(path)
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        cfg = MLPConfig(**blob["config"])
        inst = cls(config=cfg)
        inst._in_features = blob["in_features"]
        inst._n_classes = blob["n_classes"]
        inst._device = _resolve_device(cfg.device)
        inst._net = inst._build_net().to(inst._device)
        inst._net.load_state_dict(blob["state_dict"])
        inst._history = blob.get("history", {})
        return inst


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------


class _NonNegLinear(nn.Module):
    """Linear -> softplus so the predicted dose >= 0."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.linear(x)).squeeze(-1)


@register_model("mlp_reg")
class MLPRegressor(_MLPBase):
    task = "regression"

    def _build_head(self) -> nn.Module:
        return _NonNegLinear(self.config.hidden_sizes[-1])

    def _loss_fn(self, y_train: np.ndarray) -> nn.Module:
        return nn.SmoothL1Loss(beta=0.1)

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y.astype(np.float32))

    def _logits_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        return out  # already the dose prediction

    def _val_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_logits: np.ndarray | None
    ) -> tuple[dict[str, float], float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        metrics = {"val_mae": mae, "val_rmse": rmse, "val_r2": r2}
        # Monitor = negative MAE so "higher = better" matches classification.
        return metrics, -mae

    def _post_predict(self, raw: np.ndarray) -> np.ndarray:
        return raw


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------


@register_model("mlp_cls")
class MLPClassifier(_MLPBase):
    task = "classification"

    def _maybe_set_n_classes(self, y_train: np.ndarray) -> None:
        self._n_classes = int(np.max(y_train)) + 1

    def _build_head(self) -> nn.Module:
        assert self._n_classes is not None
        return nn.Linear(self.config.hidden_sizes[-1], self._n_classes)

    def _loss_fn(self, y_train: np.ndarray) -> nn.Module:
        cfg = self.config
        alpha: torch.Tensor | None = None
        if cfg.use_class_weight:
            assert self._n_classes is not None
            classes = np.arange(self._n_classes)
            present = np.unique(y_train)
            # sklearn refuses unseen classes; fill missing with 1.0
            raw = compute_class_weight("balanced", classes=present, y=y_train)
            w = np.ones(self._n_classes, dtype=np.float32)
            for c, val in zip(present, raw):
                w[int(c)] = float(val)
            alpha = torch.from_numpy(w)
        if cfg.use_focal:
            return MulticlassFocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        if alpha is not None:
            return nn.CrossEntropyLoss(weight=alpha)
        return nn.CrossEntropyLoss()

    def _y_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y.astype(np.int64))

    def _logits_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        return out.argmax(dim=-1)

    def _val_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_logits: np.ndarray | None
    ) -> tuple[dict[str, float], float]:
        macro = float(f1_score(y_true, y_pred, average="macro"))
        weighted = float(f1_score(y_true, y_pred, average="weighted"))
        acc = float((y_pred == y_true).mean())
        # Top-2 accuracy
        top2 = float("nan")
        if y_logits is not None and y_logits.ndim == 2:
            top2_idx = np.argsort(-y_logits, axis=1)[:, :2]
            top2 = float(np.mean([yt in row for yt, row in zip(y_true, top2_idx)]))
        metrics = {
            "val_macro_f1": macro,
            "val_weighted_f1": weighted,
            "val_accuracy": acc,
            "val_top2_accuracy": top2,
        }
        return metrics, macro

    def _post_predict(self, raw: np.ndarray) -> np.ndarray:
        return np.argmax(raw, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_tensor(X)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

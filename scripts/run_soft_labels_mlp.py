"""Soft Labels + MLP experiment — ordinal class 2 improvement attempt.

Baseline MLP (test fold):
  class 2 F1 = 0.358  macro-F1 = 0.504  accuracy = 0.642

Hypothesis: Replacing one-hot CE with Gaussian soft label KL-divergence
will improve ordinal accuracy, especially for class 2 (the boundary zone).

We sweep sigma ∈ {0.5, 0.8, 1.0, 1.5} and pick the best val macro-F1.
Uses the same Optuna best_params as baseline MLP for fair comparison.

Usage:
    python scripts/run_soft_labels_mlp.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402 — must precede other imports on Windows
from src.models._mlp_soft import MLPSoftClassifier, MLPSoftConfig  # noqa: E402
from src.training.data import feature_columns, load_features, load_splits  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_METRICS = ROOT / "artifacts" / "runs" / "2026-04-24_mlp_cls_42" / "metrics.json"
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR  = ROOT / "artifacts" / "runs" / f"{RUN_DATE}_mlp_soft_cls_42"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "next_ne_action_bin"

# Sigma sweep values
SIGMAS = [0.5, 0.8, 1.0, 1.5]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading features …")
df = load_features("v1")
splits = load_splits("v1")

train_ids = set(splits["train"].tolist())
val_ids   = set(splits["val"].tolist())
test_ids  = set(splits["test"].tolist())

feat_cols = feature_columns(df)
print(f"  Feature columns: {len(feat_cols)}")

df_train = df[df.stay_id.isin(train_ids)].dropna(subset=feat_cols + [LABEL_COL])
df_val   = df[df.stay_id.isin(val_ids)].dropna(subset=feat_cols + [LABEL_COL])
df_test  = df[df.stay_id.isin(test_ids)].dropna(subset=feat_cols + [LABEL_COL])

X_train = df_train[feat_cols].values.astype(np.float32)
y_train = df_train[LABEL_COL].values.astype(np.int64)
X_val   = df_val[feat_cols].values.astype(np.float32)
y_val   = df_val[LABEL_COL].values.astype(np.int64)
X_test  = df_test[feat_cols].values.astype(np.float32)
y_test  = df_test[LABEL_COL].values.astype(np.int64)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# Z-score normalisation (train stats only).
mean_ = X_train.mean(axis=0)
std_  = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean_) / std_
X_val   = (X_val   - mean_) / std_
X_test  = (X_test  - mean_) / std_

# ---------------------------------------------------------------------------
# Load Optuna best params
# ---------------------------------------------------------------------------
with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

bp = baseline["tuning"]["best_params"]
print(f"\nBaseline MLP best_params: {bp}")

base_hidden: tuple[int, ...] = tuple(
    bp[f"hidden_{i}"] for i in range(int(bp.get("n_layers", 1))) if f"hidden_{i}" in bp
)
if not base_hidden:
    base_hidden = (256,)

# ---------------------------------------------------------------------------
# Sigma sweep on validation fold
# ---------------------------------------------------------------------------
print("\nSweeping sigma on val fold …")
sigma_results: list[dict] = []

for sigma in SIGMAS:
    cfg = MLPSoftConfig(
        hidden_sizes=base_hidden,
        dropout=float(bp.get("dropout", 0.3)),
        learning_rate=float(bp.get("learning_rate", 1e-3)),
        weight_decay=float(bp.get("weight_decay", 1e-4)),
        batch_size=int(bp.get("batch_size", 512)),
        max_epochs=100,
        patience=10,
        use_class_weight=True,
        use_focal=False,
        sigma=sigma,
        random_state=42,
    )
    model = MLPSoftClassifier(config=cfg)
    val_metrics = model.fit(X_train, y_train, X_val, y_val)
    macro = val_metrics.get("val_macro_f1", 0.0)
    print(f"  sigma={sigma:.1f}  val macro-F1={macro:.4f}  "
          f"epoch={val_metrics.get('best_epoch', '?')}")
    sigma_results.append({"sigma": sigma, "val_macro_f1": macro, "model": model})

# Best sigma by val macro-F1.
best_result = max(sigma_results, key=lambda r: r["val_macro_f1"])
best_sigma  = best_result["sigma"]
best_model  = best_result["model"]
print(f"\nBest sigma: {best_sigma}  (val macro-F1={best_result['val_macro_f1']:.4f})")

# ---------------------------------------------------------------------------
# Evaluate best model on val + test
# ---------------------------------------------------------------------------
def evaluate(X: np.ndarray, y: np.ndarray, model: MLPSoftClassifier, split: str) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)

    acc      = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    w_f1     = float(f1_score(y, preds, average="weighted", zero_division=0))
    top2     = float(np.sum(np.argsort(proba, axis=1)[:, -2:] == y[:, None])) / len(y)

    per_class = {}
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    for k in ["0", "1", "2", "3", "4"]:
        if k in report:
            per_class[f"class_{k}"] = {
                "precision": report[k]["precision"],
                "recall":    report[k]["recall"],
                "f1":        report[k]["f1-score"],
                "support":   int(report[k]["support"]),
            }

    cm = confusion_matrix(y, preds, labels=[0, 1, 2, 3, 4]).tolist()

    print(f"\n[{split}] acc={acc:.4f}  macro_f1={macro_f1:.4f}  top2={top2:.4f}")
    print("  Per-class F1: " + "  ".join(
        f"cls{k}={per_class[f'class_{k}']['f1']:.3f}" for k in ["0","1","2","3","4"]
    ))
    return {
        "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": w_f1,
        "top2_accuracy": float(top2), "per_class": per_class, "confusion_matrix": cm,
    }


val_metrics  = evaluate(X_val,  y_val,  best_model, "val")
test_metrics = evaluate(X_test, y_test, best_model, "test")

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
baseline_test = baseline["test"]

print("\n" + "="*65)
print("COMPARISON: Baseline MLP vs Soft-Label MLP (test fold)")
print("="*65)
for key, label in [("macro_f1","Macro-F1"), ("accuracy","Accuracy"), ("top2_accuracy","Top-2 Acc")]:
    bv = baseline_test.get(key, float("nan"))
    nv = test_metrics.get(key, float("nan"))
    print(f"  {label:15s}: baseline={bv:.4f}  soft-label={nv:.4f}  Δ={nv-bv:+.4f}")

print("\nPer-class F1:")
print(f"  {'Class':8s}  {'Baseline':>10s}  {'Soft-Label':>10s}  {'Delta':>8s}")
for k in ["0","1","2","3","4"]:
    ck = f"class_{k}"
    bv = baseline_test["per_class"][ck]["f1"]
    nv = test_metrics["per_class"][ck]["f1"]
    flag = " ← TARGET" if k == "2" else ""
    print(f"  cls_{k}          {bv:>10.4f}  {nv:>10.4f}  {nv-bv:>+8.4f}{flag}")

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
metrics_out = {
    "experiment": "mlp_soft_labels",
    "best_sigma": best_sigma,
    "sigma_sweep": [
        {"sigma": r["sigma"], "val_macro_f1": r["val_macro_f1"]} for r in sigma_results
    ],
    "use_class_weight": True,
    "val":  val_metrics,
    "test": test_metrics,
    "baseline_test": baseline_test,
    "mlp_best_params": bp,
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

best_model.save(OUT_DIR)

# Predictions parquet.
preds_test = best_model.predict(X_test)
proba_test = best_model.predict_proba(X_test)
pred_df = pd.DataFrame({
    "stay_id": df_test.stay_id.values,
    "t_bin":   df_test.t_bin.values,
    "y_true":  y_test,
    "y_pred":  preds_test,
    **{f"proba_{i}": proba_test[:, i] for i in range(5)},
})
pred_df.to_parquet(OUT_DIR / "predictions.parquet", index=False)

config_out = {
    "model": "mlp_soft_labels",
    "best_sigma": best_sigma,
    "feature_cols": feat_cols,
    "n_train": len(X_train),
    "n_val":   len(X_val),
    "n_test":  len(X_test),
    "mlp_best_params": bp,
    "run_date": RUN_DATE,
}
with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config_out, f, indent=2)

print(f"\nArtifacts saved to: {OUT_DIR}")
print("Done.")

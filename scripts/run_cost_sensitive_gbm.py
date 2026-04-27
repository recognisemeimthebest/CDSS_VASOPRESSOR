"""Run cost-sensitive GBM experiment and compare with baseline.

Baseline (standard CE):
  class 2 F1 = 0.356  macro-F1 = 0.529  accuracy = 0.676

This experiment replaces the cross-entropy objective with a cost-sensitive
objective C[i,j] = |i-j|^2 to penalise ordinal mistakes more strongly,
aiming to improve class 2 (8.4-20 mcg/min) which is the weakest bin.

Usage:
    python scripts/run_cost_sensitive_gbm.py
"""
from __future__ import annotations

import json
import sys
import pickle
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

from src.models._gbm_cost import train_cost_sensitive_gbm
from src.training.data import feature_columns, load_features, load_splits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_METRICS = ROOT / "artifacts" / "runs" / "2026-04-24_gbm_cls_42" / "metrics.json"

RUN_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR  = ROOT / "artifacts" / "runs" / f"{RUN_DATE}_gbm_cost_cls_42"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "next_ne_action_bin"


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
print(f"  Feature columns used: {len(feat_cols)}")

df_train = df[df.stay_id.isin(train_ids)].dropna(subset=feat_cols + [LABEL_COL])
df_val   = df[df.stay_id.isin(val_ids)].dropna(subset=feat_cols + [LABEL_COL])
df_test  = df[df.stay_id.isin(test_ids)].dropna(subset=feat_cols + [LABEL_COL])

X_train, y_train = df_train[feat_cols].values, df_train[LABEL_COL].values.astype(int)
X_val,   y_val   = df_val[feat_cols].values,   df_val[LABEL_COL].values.astype(int)
X_test,  y_test  = df_test[feat_cols].values,  df_test[LABEL_COL].values.astype(int)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# Load Optuna best params
# ---------------------------------------------------------------------------
with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

best_params = baseline["tuning"]["best_params"]
print(f"\nOptuna best params: {best_params}")

# ---------------------------------------------------------------------------
# Train cost-sensitive GBM
# ---------------------------------------------------------------------------
print("\nTraining cost-sensitive GBM …")
booster, train_summary, predict_fn, predict_proba_fn = train_cost_sensitive_gbm(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    best_params=best_params,
    n_estimators=2000,
    early_stopping_rounds=50,
    cost_power=2.0,
    use_class_weight=True,
    seed=42,
)

print(f"\nTrain summary: {train_summary}")

# ---------------------------------------------------------------------------
# Evaluate on val + test
# ---------------------------------------------------------------------------
def evaluate(X: np.ndarray, y: np.ndarray, split: str) -> dict:
    preds = predict_fn(X)
    proba = predict_proba_fn(X)

    acc      = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    w_f1     = float(f1_score(y, preds, average="weighted", zero_division=0))

    # top-2 accuracy
    top2 = int(np.sum(np.argsort(proba, axis=1)[:, -2:] == y[:, None])) / len(y)

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

    print(f"\n[{split}] accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  top2={top2:.4f}")
    print(f"  Per-class F1: " + "  ".join(
        f"cls{k}={per_class[f'class_{k}']['f1']:.3f}" for k in ["0","1","2","3","4"]
    ))
    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "weighted_f1": w_f1,
        "top2_accuracy": float(top2),
        "per_class":   per_class,
        "confusion_matrix": cm,
    }


val_metrics  = evaluate(X_val,  y_val,  "val")
test_metrics = evaluate(X_test, y_test, "test")

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
baseline_test = baseline["test"]

print("\n" + "="*60)
print("COMPARISON: Baseline GBM vs Cost-Sensitive GBM (test fold)")
print("="*60)
metrics_compare = [
    ("macro_f1",     "Macro-F1"),
    ("accuracy",     "Accuracy"),
    ("top2_accuracy","Top-2 Acc"),
]
for key, label in metrics_compare:
    base_v = baseline_test.get(key, "N/A")
    new_v  = test_metrics.get(key, "N/A")
    delta  = f"{new_v - base_v:+.4f}" if isinstance(base_v, float) else "N/A"
    print(f"  {label:15s}: baseline={base_v:.4f}  cost-sens={new_v:.4f}  Δ={delta}")

print("\nPer-class F1 comparison:")
print(f"  {'Class':8s}  {'Baseline':>10s}  {'Cost-Sens':>10s}  {'Delta':>8s}")
for k in ["0","1","2","3","4"]:
    ck = f"class_{k}"
    base_f1 = baseline_test["per_class"][ck]["f1"]
    new_f1  = test_metrics["per_class"][ck]["f1"]
    delta   = new_f1 - base_f1
    flag    = " ← TARGET" if k == "2" else ""
    print(f"  cls_{k} ({ck})  {base_f1:>10.4f}  {new_f1:>10.4f}  {delta:>+8.4f}{flag}")

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
metrics_out = {
    "experiment": "gbm_cost_sensitive",
    "cost_power": 2.0,
    "use_class_weight": True,
    "train_summary": train_summary,
    "val":  val_metrics,
    "test": test_metrics,
    "baseline_test": baseline_test,
    "optuna_params": best_params,
}

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

booster.save_model(str(OUT_DIR / "model.lgb"))

# Save predictions parquet
preds_test  = predict_fn(X_test)
proba_test  = predict_proba_fn(X_test)
pred_df = pd.DataFrame({
    "stay_id": df_test.stay_id.values,
    "t_bin":   df_test.t_bin.values,
    "y_true":  y_test,
    "y_pred":  preds_test,
    **{f"proba_{i}": proba_test[:, i] for i in range(5)},
})
pred_df.to_parquet(OUT_DIR / "predictions.parquet", index=False)

config = {
    "model": "gbm_cost_sensitive",
    "cost_power": 2.0,
    "feature_cols": feat_cols,
    "n_train": len(X_train),
    "n_val":   len(X_val),
    "n_test":  len(X_test),
    "best_params": best_params,
    "run_date": RUN_DATE,
}
with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nArtifacts saved to: {OUT_DIR}")
print("Done.")

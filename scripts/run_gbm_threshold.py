"""GBM per-class probability scaling (threshold adjustment).

No retraining. The GBM already outputs calibrated probabilities.
We multiply each class's probability by a scale factor before argmax:

    pred = argmax( proba * scale_vector )

where scale_vector[k] boosts the 'confidence' for class k.
Optimised on val fold → applied to test fold.

Two sweep strategies:
  A) Sweep class-2 scale only (others=1): maximise class 2 F1
  B) Joint grid search on class-1, class-2, class-3 scales: maximise macro-F1

Usage:
    python scripts/run_gbm_threshold.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GBM_PREDS  = ROOT / "artifacts" / "runs" / "2026-04-24_gbm_cls_42" / "predictions.parquet"
GBM_METRICS = ROOT / "artifacts" / "runs" / "2026-04-24_gbm_cls_42" / "metrics.json"

RUN_DATE = datetime.now().strftime("%Y-%m-%d")
OUT_DIR  = ROOT / "artifacts" / "runs" / f"{RUN_DATE}_gbm_threshold_cls_42"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
df = pd.read_parquet(GBM_PREDS)
val  = df[df.fold == "val"].reset_index(drop=True)
test = df[df.fold == "test"].reset_index(drop=True)

proba_val  = val[[f"proba_{i}"  for i in range(5)]].values.astype(np.float64)
proba_test = test[[f"proba_{i}" for i in range(5)]].values.astype(np.float64)
y_val  = val["y_true"].values
y_test = test["y_true"].values

print(f"Val : {len(val):,}  Test: {len(test):,}")

# ---------------------------------------------------------------------------
# Helper: apply scale vector and predict
# ---------------------------------------------------------------------------
def scaled_predict(proba: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """argmax( proba * scales ) - no renormalisation needed for argmax."""
    return np.argmax(proba * scales[None, :], axis=1)


def report_cls2(y_true, y_pred, label=""):
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cls2  = f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    acc   = (y_pred == y_true).mean()
    if label:
        print(f"  {label:35s}  macro={macro:.4f}  cls2={cls2:.4f}  acc={acc:.4f}")
    return macro, cls2


# Baseline (scale=1 everywhere)
baseline_scales = np.ones(5)
base_pred_val  = scaled_predict(proba_val, baseline_scales)
base_pred_test = scaled_predict(proba_test, baseline_scales)
print("\nBaseline:")
report_cls2(y_val,  base_pred_val,  "val")
report_cls2(y_test, base_pred_test, "test")

# ---------------------------------------------------------------------------
# Strategy A: Sweep class-2 scale (maximise class-2 F1 on val)
# ---------------------------------------------------------------------------
print("\nStrategy A - class-2 scale sweep (maximise class-2 F1):")
best_cls2_f1 = -1.0
best_s2_for_cls2 = 1.0
results_a = []

for s2 in np.arange(1.0, 6.1, 0.1):
    scales = np.array([1.0, 1.0, s2, 1.0, 1.0])
    preds = scaled_predict(proba_val, scales)
    macro, cls2 = report_cls2(y_val, preds, f"s2={s2:.1f}")
    results_a.append({"s2": float(s2), "val_cls2_f1": cls2, "val_macro_f1": macro})
    if cls2 > best_cls2_f1:
        best_cls2_f1 = cls2
        best_s2_for_cls2 = float(s2)

print(f"\n  Best s2 for class-2 F1: {best_s2_for_cls2}  (val cls2={best_cls2_f1:.4f})")

# ---------------------------------------------------------------------------
# Strategy B: Sweep class-2 scale (maximise MACRO F1 on val)
# ---------------------------------------------------------------------------
print("\nStrategy B - class-2 scale sweep (maximise macro F1):")
best_macro = -1.0
best_s2_for_macro = 1.0

for s2 in np.arange(1.0, 6.1, 0.1):
    scales = np.array([1.0, 1.0, s2, 1.0, 1.0])
    preds = scaled_predict(proba_val, scales)
    macro, cls2 = f1_score(y_val, preds, average="macro", zero_division=0), \
                  f1_score(y_val, preds, labels=[2], average=None, zero_division=0)[0]
    if macro > best_macro:
        best_macro = macro
        best_s2_for_macro = float(s2)

print(f"  Best s2 for macro F1: {best_s2_for_macro}  (val macro={best_macro:.4f})")

# ---------------------------------------------------------------------------
# Strategy C: Joint sweep cls-1, cls-2, cls-3 (maximise macro F1)
# ---------------------------------------------------------------------------
print("\nStrategy C - joint sweep cls1/cls2/cls3 (maximise macro F1):")
best_macro_joint = best_macro
best_scales_joint = np.array([1.0, 1.0, best_s2_for_macro, 1.0, 1.0])

for s2 in [best_s2_for_macro, best_s2_for_macro * 0.8, best_s2_for_macro * 1.2]:
    for s1 in np.arange(0.8, 2.1, 0.2):
        for s3 in np.arange(0.8, 2.1, 0.2):
            scales = np.array([1.0, s1, s2, s3, 1.0])
            preds = scaled_predict(proba_val, scales)
            macro = f1_score(y_val, preds, average="macro", zero_division=0)
            if macro > best_macro_joint:
                best_macro_joint = macro
                best_scales_joint = scales.copy()

s = best_scales_joint
print(f"  Best scales: cls0=1.0  cls1={s[1]:.2f}  cls2={s[2]:.2f}  cls3={s[3]:.2f}  cls4=1.0")
print(f"  Val macro F1: {best_macro_joint:.4f}")

# ---------------------------------------------------------------------------
# Evaluate all strategies on TEST fold
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("TEST FOLD RESULTS")
print("="*65)

strategies = {
    "Baseline (no scaling)":      np.ones(5),
    f"A - cls2 scale={best_s2_for_cls2:.1f} (max cls2-F1)": np.array([1.0, 1.0, best_s2_for_cls2, 1.0, 1.0]),
    f"B - cls2 scale={best_s2_for_macro:.1f} (max macro-F1)": np.array([1.0, 1.0, best_s2_for_macro, 1.0, 1.0]),
    "C - joint scales (max macro-F1)": best_scales_joint,
}

test_results = {}
for name, scales in strategies.items():
    preds = scaled_predict(proba_test, scales)
    macro, cls2_f1 = report_cls2(y_test, preds, name)
    per_class_f1 = f1_score(y_test, preds, average=None, zero_division=0)
    test_results[name] = {
        "scales": scales.tolist(),
        "macro_f1": float(macro),
        "cls2_f1": float(cls2_f1),
        "per_class_f1": per_class_f1.tolist(),
        "accuracy": float((preds == y_test).mean()),
    }

# Full per-class breakdown for best strategy by cls2 F1
print("\n--- Per-class detail: Strategy A (max cls2 F1) ---")
best_a_scales = np.array([1.0, 1.0, best_s2_for_cls2, 1.0, 1.0])
preds_a = scaled_predict(proba_test, best_a_scales)
rpt = classification_report(y_test, preds_a, zero_division=0, output_dict=True)
for k in ["0","1","2","3","4"]:
    r = rpt.get(k, {})
    print(f"  cls_{k}: prec={r.get('precision',0):.3f}  rec={r.get('recall',0):.3f}  f1={r.get('f1-score',0):.3f}")

# ---------------------------------------------------------------------------
# Load baseline test metrics for Δ
# ---------------------------------------------------------------------------
with open(GBM_METRICS) as f:
    base_m = json.load(f)["test"]

print("\n--- Comparison vs Baseline GBM ---")
print(f"{'Strategy':42s}  {'macro-F1':>8s}  {'cls2-F1':>8s}  {'Δmacro':>8s}  {'Δcls2':>8s}")
bm = base_m["macro_f1"]
bc = base_m["per_class"]["class_2"]["f1"]
for name, r in test_results.items():
    print(f"  {name[:40]:40s}  {r['macro_f1']:>8.4f}  {r['cls2_f1']:>8.4f}  "
          f"{r['macro_f1']-bm:>+8.4f}  {r['cls2_f1']-bc:>+8.4f}")

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
out = {
    "experiment": "gbm_threshold_scaling",
    "strategies": {k: {kk: vv for kk, vv in v.items() if kk != "scales"} | {"scales": v["scales"]}
                   for k, v in test_results.items()},
    "baseline_test": {"macro_f1": bm, "cls2_f1": bc},
    "sigma_sweep_a": results_a,
    "best_for_cls2": {"s2": best_s2_for_cls2, "val_cls2_f1": best_cls2_f1},
    "best_for_macro": {"s2": best_s2_for_macro, "val_macro_f1": best_macro},
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\nArtifacts saved to: {OUT_DIR}")
print("Done.")

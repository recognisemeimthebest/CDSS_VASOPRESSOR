"""Shared data-loading utilities for the Streamlit app.

All DataFrames are cached with st.cache_data to avoid reloading on every
widget interaction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

# Action bin labels (NE-equivalent mcg/min)
ACTION_LABELS = {
    0: "No vasopressor",
    1: "NE ≤ 8.4 mcg/min  (low)",
    2: "NE 8.4–20 mcg/min  (medium)",
    3: "NE 20–50 mcg/min  (high)",
    4: "NE > 50 mcg/min  (max)",
}
ACTION_COLORS = {
    0: "#4CAF50",  # green
    1: "#8BC34A",
    2: "#FFC107",  # amber
    3: "#FF5722",  # deep orange
    4: "#B71C1C",  # dark red
}

VITALS_DISPLAY = {
    "hr_last": "Heart Rate (bpm)",
    "sbp_last": "SBP (mmHg)",
    "mbp_last": "MBP (mmHg)",
    "spo2_last": "SpO₂ (%)",
    "rr_last": "Resp Rate (/min)",
    "temperature_last": "Temp (°C)",
}
LAB_DISPLAY = {
    "lactate_last": "Lactate (mmol/L)",
    "creatinine_last": "Creatinine (mg/dL)",
    "sofa": "SOFA Score",
    "ne_dose_now": "NE dose now (mcg/min)",
}


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features_v1.parquet")


@st.cache_data
def load_cohort() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cohort_v1.parquet")


@st.cache_data
def load_gbm_predictions() -> pd.DataFrame:
    return pd.read_parquet(RUNS_DIR / "2026-04-24_gbm_cls_42" / "predictions.parquet")


@st.cache_data
def load_dbcq_predictions() -> pd.DataFrame:
    return pd.read_parquet(RUNS_DIR / "2026-04-27_rl_dbcq_mlp_42" / "predictions.parquet")


@st.cache_data
def load_bc_predictions() -> pd.DataFrame:
    return pd.read_parquet(RUNS_DIR / "2026-04-27_rl_bc_mlp_42" / "predictions.parquet")


def get_test_stay_ids(gbm: pd.DataFrame) -> list[int]:
    return sorted(gbm.loc[gbm.fold == "test", "stay_id"].unique().tolist())


def patient_timeline(features: pd.DataFrame, stay_id: int) -> pd.DataFrame:
    df = features[features.stay_id == stay_id].sort_values("t_bin").copy()
    df["time_h"] = df["t_bin"] * 4
    return df

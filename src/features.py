"""Time-series feature engineering for vasopressor recommendation (Phase 2).

Pulls derived-table values for each cohort stay and aggregates them into a
4-hour-bin long DataFrame over the first 72 h of ICU stay (max 18 bins).

Output schema (one row per stay x t_bin):

    keys          : stay_id, subject_id, t_bin
    static        : age, gender_M, weight_kg, first_careunit_*
    vitals        : hr/sbp/dbp/mbp/rr/spo2/temperature/glucose_{mean,min,max,last}
    blood gas     : lactate/ph/po2/pco2/baseexcess/bicarbonate _last (+ missing flags)
    chemistry     : sodium/potassium/creatinine/bun/glucose_lab_last
    coagulation   : inr/ptt/platelet _last
    CBC           : wbc/hematocrit/hemoglobin _last
    SOFA          : sofa, respiration, coagulation_score, liver, cardiovascular,
                    cns, renal  (last per-hour value falling inside the bin)
    treatment     : vent_flag (0/1 overlap), urine_4h_ml
    current dose  : ne_dose_now (mcg/kg/min, mean this bin)
    action (next) : next_ne_dose, next_ne_action_bin (0..4)

Missingness policy:
    1. groupby('stay_id').ffill()   --- forward-fill within a stay only
    2. remaining NaN --- cohort mean (stored in FeatureStats; fold-specific
       normalisation is applied downstream using the train fold only)
    3. '_was_missing' 0/1 columns added for sparse lab variables:
       lactate, ph, po2, pco2, baseexcess, lactate chemistry,
       inr, ptt, wbc, sofa.

Public API:
    build_features(cohort, version='v1', use_cache=True) -> (DataFrame, dict)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.cohort import load_cohort
from src.db import get_engine

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BIN_HOURS = 4            # PROJECT_PLAN §2.3
WINDOW_HOURS = 72        # PROJECT_PLAN §2.3
N_BINS = WINDOW_HOURS // BIN_HOURS  # 18

# NE-equivalent 5-bin edges in mcg/min (AI Clinician Komorowski 2018).
# dose_mcg_min = dose_mcg_kg_min * weight_kg
NE_BIN_EDGES_MCG_MIN = (0.0, 8.4, 20.28, 50.0)  # [0], (0, 8.4], (8.4, 20.28], (20.28, 50], (50, inf)

# Variables that deserve a missing-indicator (sparse per 4h bin).
MISSING_FLAG_VARS: tuple[str, ...] = (
    "lactate_last",
    "ph_last",
    "po2_last",
    "pco2_last",
    "baseexcess_last",
    "inr_last",
    "ptt_last",
    "wbc_last",
    "sofa",
)

# Careunit one-hot values we keep.  Anything else becomes first_careunit_OTHER.
CAREUNIT_VALUES: tuple[str, ...] = (
    "Medical Intensive Care Unit (MICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Coronary Care Unit (CCU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Trauma SICU (TSICU)",
    "Neuro Stepdown",
    "Neuro Intermediate",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
)

DEFAULT_VERSION = "v1"

# --- Paths ------------------------------------------------------------------


def _features_path(version: str) -> Path:
    return DATA_DIR / f"features_{version}.parquet"


def _stats_path(version: str) -> Path:
    return ARTIFACTS_DIR / f"feature_stats_{version}.json"


# --- SQL helpers ------------------------------------------------------------


def _chunks(seq: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _read_chunked(
    engine: Engine,
    sql: str,
    stay_ids: list[int] | None,
    subject_ids: list[int] | None = None,
    chunk: int = 3000,
) -> pd.DataFrame:
    """Fetch rows for many stay_ids in chunks, avoid parameter blow-up."""
    dfs: list[pd.DataFrame] = []
    if stay_ids is not None:
        for batch in _chunks(stay_ids, chunk):
            with engine.connect() as conn:
                dfs.append(pd.read_sql(text(sql), conn, params={"ids": batch}))
    elif subject_ids is not None:
        for batch in _chunks(subject_ids, chunk):
            with engine.connect() as conn:
                dfs.append(pd.read_sql(text(sql), conn, params={"ids": batch}))
    else:
        raise ValueError("Need stay_ids or subject_ids")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# --- SQL queries ------------------------------------------------------------
# All queries restrict to ICU intime..intime+72h and attach stay_id so
# Python-side binning is purely arithmetic on hours-since-intime.

_VITALS_SQL = """
SELECT v.stay_id, v.charttime,
       v.heart_rate AS hr, v.sbp, v.dbp, v.mbp,
       v.resp_rate AS rr, v.spo2, v.temperature, v.glucose AS glucose_vital
FROM mimiciv_derived.vitalsign v
JOIN mimiciv_icu.icustays i USING (stay_id)
WHERE v.stay_id = ANY(:ids)
  AND v.charttime >= i.intime
  AND v.charttime <  i.intime + INTERVAL '72 hours'
"""

_BG_SQL = """
SELECT i.stay_id, bg.charttime,
       bg.lactate, bg.ph, bg.po2, bg.pco2,
       bg.baseexcess, bg.bicarbonate AS bg_bicarbonate
FROM mimiciv_derived.bg bg
JOIN mimiciv_icu.icustays i
  ON i.subject_id = bg.subject_id
 AND i.hadm_id    = bg.hadm_id
WHERE i.stay_id = ANY(:ids)
  AND bg.charttime >= i.intime
  AND bg.charttime <  i.intime + INTERVAL '72 hours'
"""

_CHEM_SQL = """
SELECT i.stay_id, c.charttime,
       c.sodium, c.potassium, c.creatinine, c.bun,
       c.glucose AS glucose_lab
FROM mimiciv_derived.chemistry c
JOIN mimiciv_icu.icustays i
  ON i.subject_id = c.subject_id
 AND i.hadm_id    = c.hadm_id
WHERE i.stay_id = ANY(:ids)
  AND c.charttime >= i.intime
  AND c.charttime <  i.intime + INTERVAL '72 hours'
"""

_COAG_SQL = """
SELECT i.stay_id, g.charttime, g.inr, g.ptt
FROM mimiciv_derived.coagulation g
JOIN mimiciv_icu.icustays i
  ON i.subject_id = g.subject_id
 AND i.hadm_id    = g.hadm_id
WHERE i.stay_id = ANY(:ids)
  AND g.charttime >= i.intime
  AND g.charttime <  i.intime + INTERVAL '72 hours'
"""

_CBC_SQL = """
SELECT i.stay_id, b.charttime,
       b.wbc, b.hematocrit, b.hemoglobin, b.platelet
FROM mimiciv_derived.complete_blood_count b
JOIN mimiciv_icu.icustays i
  ON i.subject_id = b.subject_id
 AND i.hadm_id    = b.hadm_id
WHERE i.stay_id = ANY(:ids)
  AND b.charttime >= i.intime
  AND b.charttime <  i.intime + INTERVAL '72 hours'
"""

# SOFA is hourly; align endtime with the bin it falls in.
_SOFA_SQL = """
SELECT s.stay_id, s.endtime AS charttime,
       s.sofa_24hours AS sofa,
       s.respiration_24hours AS respiration,
       s.coagulation_24hours AS coagulation_score,
       s.liver_24hours AS liver,
       s.cardiovascular_24hours AS cardiovascular,
       s.cns_24hours AS cns,
       s.renal_24hours AS renal
FROM mimiciv_derived.sofa s
JOIN mimiciv_icu.icustays i USING (stay_id)
WHERE s.stay_id = ANY(:ids)
  AND s.endtime >  i.intime
  AND s.endtime <= i.intime + INTERVAL '72 hours'
"""

# Ventilation: overlap a bin if any vent episode intersects [bin_start, bin_end).
_VENT_SQL = """
SELECT v.stay_id, v.starttime, v.endtime, v.ventilation_status
FROM mimiciv_derived.ventilation v
JOIN mimiciv_icu.icustays i USING (stay_id)
WHERE v.stay_id = ANY(:ids)
  AND v.endtime   >  i.intime
  AND v.starttime <  i.intime + INTERVAL '72 hours'
  AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy', 'NonInvasiveVent')
"""

# Urine output: total volume per bin.
_UO_SQL = """
SELECT u.stay_id, u.charttime, u.urineoutput AS value
FROM mimiciv_derived.urine_output u
JOIN mimiciv_icu.icustays i USING (stay_id)
WHERE u.stay_id = ANY(:ids)
  AND u.charttime >= i.intime
  AND u.charttime <  i.intime + INTERVAL '72 hours'
  AND u.urineoutput IS NOT NULL
  AND u.urineoutput > 0
"""

# NE-equivalent dose: intervals in mcg/kg/min. Used for both current-bin dose
# and next-bin action label. Pull the full 0..76h window so bin 17 has a
# defined next-action.
_NE_SQL = """
SELECT n.stay_id, n.starttime, n.endtime,
       n.norepinephrine_equivalent_dose AS dose
FROM mimiciv_derived.norepinephrine_equivalent_dose n
JOIN mimiciv_icu.icustays i USING (stay_id)
WHERE n.stay_id = ANY(:ids)
  AND n.endtime   >  i.intime
  AND n.starttime <  i.intime + INTERVAL '76 hours'
  AND n.norepinephrine_equivalent_dose IS NOT NULL
"""

# Weight per stay: take the earliest in-stay weight (AI Clinician style).
_WEIGHT_SQL = """
SELECT stay_id, weight
FROM (
  SELECT w.stay_id, w.weight,
         ROW_NUMBER() OVER (PARTITION BY w.stay_id ORDER BY w.starttime NULLS LAST) AS rn
  FROM mimiciv_derived.weight_durations w
  WHERE w.stay_id = ANY(:ids)
    AND w.weight BETWEEN 20 AND 400
) t WHERE rn = 1
"""


# --- Binning ----------------------------------------------------------------


def _bin_index(charttime: pd.Series, intime: pd.Series) -> pd.Series:
    """Hours since intime -> 4h-bin index (0..N_BINS-1 kept; rest dropped).

    SQL pre-filters charttime >= intime, so the result is always >= 0; NaT
    rows produce NaN which we coerce to -1 so the caller's range filter drops
    them.
    """
    hours = (charttime - intime).dt.total_seconds() / 3600.0
    binned = np.floor(hours / BIN_HOURS).fillna(-1)
    return binned.astype("int64")


def _merge_intime(df: pd.DataFrame, intime_map: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["intime"] = df["stay_id"].map(intime_map)
    return df


def _aggregate_vitals(vitals: pd.DataFrame, intime_map: pd.Series) -> pd.DataFrame:
    if vitals.empty:
        return pd.DataFrame(columns=["stay_id", "t_bin"])
    df = _merge_intime(vitals, intime_map)
    df["t_bin"] = _bin_index(df["charttime"], df["intime"])
    df = df[(df["t_bin"] >= 0) & (df["t_bin"] < N_BINS)]
    value_cols = ["hr", "sbp", "dbp", "mbp", "rr", "spo2", "temperature", "glucose_vital"]
    df = df.sort_values(["stay_id", "t_bin", "charttime"])
    # as_index=True + reset_index to get a predictable flat column set after .agg().
    grouped = df.groupby(["stay_id", "t_bin"])
    agg = grouped[value_cols].agg(["mean", "min", "max", "last"])
    # agg.columns is a MultiIndex (value_col, stat); flatten it.
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    return agg


def _aggregate_lab_last(
    labs: pd.DataFrame,
    intime_map: pd.Series,
    cols: list[str],
    suffix: str = "_last",
) -> pd.DataFrame:
    if labs.empty:
        return pd.DataFrame(columns=["stay_id", "t_bin"])
    df = _merge_intime(labs, intime_map)
    df["t_bin"] = _bin_index(df["charttime"], df["intime"])
    df = df[(df["t_bin"] >= 0) & (df["t_bin"] < N_BINS)]
    df = df.sort_values(["stay_id", "t_bin", "charttime"])
    # last non-null per bin.
    agg = (
        df.groupby(["stay_id", "t_bin"], as_index=False)[cols]
        .last()  # last row in the window
    )
    agg = agg.rename(columns={c: f"{c}{suffix}" for c in cols})
    return agg


def _aggregate_sofa(sofa: pd.DataFrame, intime_map: pd.Series) -> pd.DataFrame:
    if sofa.empty:
        return pd.DataFrame(columns=["stay_id", "t_bin"])
    df = _merge_intime(sofa, intime_map)
    df["t_bin"] = _bin_index(df["charttime"], df["intime"])
    df = df[(df["t_bin"] >= 0) & (df["t_bin"] < N_BINS)]
    df = df.sort_values(["stay_id", "t_bin", "charttime"])
    cols = ["sofa", "respiration", "coagulation_score", "liver", "cardiovascular", "cns", "renal"]
    return df.groupby(["stay_id", "t_bin"], as_index=False)[cols].last()


def _ventilation_overlap(
    vent: pd.DataFrame,
    skeleton: pd.DataFrame,
    intime_map: pd.Series,
) -> pd.DataFrame:
    """Set vent_flag=1 for any bin overlapping a vent interval."""
    if vent.empty:
        return skeleton.assign(vent_flag=0)[["stay_id", "t_bin", "vent_flag"]]
    vent = vent.copy()
    vent["intime"] = vent["stay_id"].map(intime_map)
    vent["s_hr"] = (vent["starttime"] - vent["intime"]).dt.total_seconds() / 3600.0
    vent["e_hr"] = (vent["endtime"] - vent["intime"]).dt.total_seconds() / 3600.0
    # Clip to [0, 72].
    vent["s_hr"] = vent["s_hr"].clip(lower=0, upper=WINDOW_HOURS)
    vent["e_hr"] = vent["e_hr"].clip(lower=0, upper=WINDOW_HOURS)
    vent = vent[vent["e_hr"] > vent["s_hr"]]
    # Emit one row per (stay, bin) that a vent interval touches.
    rows: list[tuple[int, int]] = []
    for stay_id, s_hr, e_hr in zip(vent["stay_id"], vent["s_hr"], vent["e_hr"]):
        first_bin = int(s_hr // BIN_HOURS)
        last_bin = int((e_hr - 1e-9) // BIN_HOURS)
        for b in range(first_bin, last_bin + 1):
            rows.append((stay_id, b))
    if not rows:
        return skeleton.assign(vent_flag=0)[["stay_id", "t_bin", "vent_flag"]]
    vf = pd.DataFrame(rows, columns=["stay_id", "t_bin"]).drop_duplicates()
    vf["vent_flag"] = 1
    return vf


def _urine_per_bin(uo: pd.DataFrame, intime_map: pd.Series) -> pd.DataFrame:
    if uo.empty:
        return pd.DataFrame(columns=["stay_id", "t_bin", "urine_4h_ml"])
    df = _merge_intime(uo, intime_map)
    df["t_bin"] = _bin_index(df["charttime"], df["intime"])
    df = df[(df["t_bin"] >= 0) & (df["t_bin"] < N_BINS)]
    agg = df.groupby(["stay_id", "t_bin"], as_index=False)["value"].sum()
    return agg.rename(columns={"value": "urine_4h_ml"})


def _ne_per_bin(
    ne: pd.DataFrame,
    skeleton: pd.DataFrame,
    intime_map: pd.Series,
    weight_by_stay: pd.Series,
    cohort_mean_weight: float,
) -> pd.DataFrame:
    """For each (stay, bin) return mean NE-equiv dose (mcg/kg/min).

    Integrates dose-time and divides by bin duration (in hours inside the
    window).  This respects intervals that span multiple bins.
    """
    if ne.empty:
        return skeleton.assign(ne_dose=0.0)[["stay_id", "t_bin", "ne_dose"]]
    ne = ne.copy()
    ne["intime"] = ne["stay_id"].map(intime_map)
    ne["s_hr"] = (ne["starttime"] - ne["intime"]).dt.total_seconds() / 3600.0
    ne["e_hr"] = (ne["endtime"] - ne["intime"]).dt.total_seconds() / 3600.0
    ne["s_hr"] = ne["s_hr"].clip(lower=0, upper=N_BINS * BIN_HOURS + BIN_HOURS)
    ne["e_hr"] = ne["e_hr"].clip(lower=0, upper=N_BINS * BIN_HOURS + BIN_HOURS)
    ne = ne[ne["e_hr"] > ne["s_hr"]]

    rows: list[tuple[int, int, float]] = []
    for stay_id, s_hr, e_hr, dose in zip(
        ne["stay_id"], ne["s_hr"], ne["e_hr"], ne["dose"]
    ):
        if pd.isna(dose):
            continue
        first_bin = int(s_hr // BIN_HOURS)
        last_bin = int((e_hr - 1e-9) // BIN_HOURS)
        for b in range(first_bin, last_bin + 1):
            b_start = b * BIN_HOURS
            b_end = b_start + BIN_HOURS
            overlap = max(0.0, min(e_hr, b_end) - max(s_hr, b_start))
            if overlap > 0:
                rows.append((stay_id, b, float(dose) * overlap))
    if not rows:
        return skeleton.assign(ne_dose=0.0)[["stay_id", "t_bin", "ne_dose"]]
    doses = pd.DataFrame(rows, columns=["stay_id", "t_bin", "dose_hours"])
    agg = doses.groupby(["stay_id", "t_bin"], as_index=False)["dose_hours"].sum()
    agg["ne_dose"] = agg["dose_hours"] / BIN_HOURS  # mean mcg/kg/min across the bin
    agg = agg.drop(columns=["dose_hours"])

    # Keep bins beyond 17 to compute next_ne_dose for bin 17.
    return agg


def _ne_action_bin_from_mcg_min(dose_mcg_min: pd.Series) -> pd.Series:
    """Map mcg/min -> {0..4} per PROJECT_PLAN §2.3."""
    bins = pd.Series(0, index=dose_mcg_min.index, dtype="int8")
    d = dose_mcg_min.fillna(0.0)
    bins = bins.mask(d > 0, 1)
    bins = bins.mask(d > NE_BIN_EDGES_MCG_MIN[1], 2)
    bins = bins.mask(d > NE_BIN_EDGES_MCG_MIN[2], 3)
    bins = bins.mask(d > NE_BIN_EDGES_MCG_MIN[3], 4)
    return bins


# --- Missing handling -------------------------------------------------------


def _add_missing_flags(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c.replace('_last', '')}_was_missing"] = df[c].isna().astype("int8")
    return df


def _ffill_and_impute(df: pd.DataFrame, fill_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Forward-fill within stay_id; then fill residual NaN with cohort mean."""
    df = df.copy()
    if fill_cols:
        df[fill_cols] = df.groupby("stay_id")[fill_cols].ffill()
    cohort_mean = df[fill_cols].mean(numeric_only=True)
    df[fill_cols] = df[fill_cols].fillna(cohort_mean)
    return df, cohort_mean.to_dict()


# --- Main driver ------------------------------------------------------------


def build_features(
    cohort: pd.DataFrame | None = None,
    version: str = DEFAULT_VERSION,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the 4h-bin long feature DataFrame for every cohort stay.

    Returns (features_df, stats_dict) and writes both to disk under
    `data/features_{version}.parquet` and `artifacts/feature_stats_{version}.json`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cache = _features_path(version)
    stats_p = _stats_path(version)
    if use_cache and cache.exists() and stats_p.exists():
        logger.info("Loading features cache: %s", cache)
        df = pd.read_parquet(cache)
        stats = json.loads(stats_p.read_text())
        return df, stats

    if cohort is None:
        cohort = load_cohort(use_cache=True)
    stay_ids = cohort["stay_id"].astype(int).tolist()
    intime_map = pd.Series(cohort["intime"].values, index=cohort["stay_id"].values)

    engine = get_engine()
    logger.info("Pulling vitals...")
    vitals = _read_chunked(engine, _VITALS_SQL, stay_ids)
    logger.info("Pulling blood gas...")
    bg = _read_chunked(engine, _BG_SQL, stay_ids)
    logger.info("Pulling chemistry...")
    chem = _read_chunked(engine, _CHEM_SQL, stay_ids)
    logger.info("Pulling coagulation...")
    coag = _read_chunked(engine, _COAG_SQL, stay_ids)
    logger.info("Pulling CBC...")
    cbc = _read_chunked(engine, _CBC_SQL, stay_ids)
    logger.info("Pulling SOFA...")
    sofa = _read_chunked(engine, _SOFA_SQL, stay_ids)
    logger.info("Pulling ventilation...")
    vent = _read_chunked(engine, _VENT_SQL, stay_ids)
    logger.info("Pulling urine output...")
    uo = _read_chunked(engine, _UO_SQL, stay_ids)
    logger.info("Pulling NE-equiv dose...")
    ne = _read_chunked(engine, _NE_SQL, stay_ids)
    logger.info("Pulling weights...")
    weights = _read_chunked(engine, _WEIGHT_SQL, stay_ids)

    weight_by_stay = pd.Series(
        weights["weight"].values if not weights.empty else [],
        index=weights["stay_id"].values if not weights.empty else [],
    )
    cohort_mean_weight = float(weight_by_stay.mean()) if len(weight_by_stay) else 80.0

    # Skeleton: every (stay, t_bin) in [0, N_BINS).
    skeleton = (
        cohort[["stay_id", "subject_id"]]
        .merge(pd.DataFrame({"t_bin": list(range(N_BINS))}), how="cross")
        .sort_values(["stay_id", "t_bin"])
        .reset_index(drop=True)
    )

    vitals_agg = _aggregate_vitals(vitals, intime_map)
    bg_agg = _aggregate_lab_last(
        bg, intime_map,
        ["lactate", "ph", "po2", "pco2", "baseexcess", "bg_bicarbonate"],
    )
    chem_agg = _aggregate_lab_last(
        chem, intime_map,
        ["sodium", "potassium", "creatinine", "bun", "glucose_lab"],
    )
    coag_agg = _aggregate_lab_last(coag, intime_map, ["inr", "ptt"])
    cbc_agg = _aggregate_lab_last(
        cbc, intime_map, ["wbc", "hematocrit", "hemoglobin", "platelet"],
    )
    sofa_agg = _aggregate_sofa(sofa, intime_map)
    vent_agg = _ventilation_overlap(vent, skeleton, intime_map)
    uo_agg = _urine_per_bin(uo, intime_map)

    # NE doses per bin (may contain bins > N_BINS-1 — we need next_bin for bin 17).
    ne_agg = _ne_per_bin(ne, skeleton, intime_map, weight_by_stay, cohort_mean_weight)

    # Merge.
    features = skeleton.copy()
    for tbl in (vitals_agg, bg_agg, chem_agg, coag_agg, cbc_agg, sofa_agg, uo_agg):
        if not tbl.empty:
            features = features.merge(tbl, on=["stay_id", "t_bin"], how="left")
    features = features.merge(vent_agg, on=["stay_id", "t_bin"], how="left")
    features["vent_flag"] = features["vent_flag"].fillna(0).astype("int8")

    # --- Current dose + next-bin action ----
    ne_now = ne_agg[ne_agg["t_bin"] < N_BINS].rename(columns={"ne_dose": "ne_dose_now"})
    features = features.merge(ne_now, on=["stay_id", "t_bin"], how="left")
    features["ne_dose_now"] = features["ne_dose_now"].fillna(0.0)

    ne_next = ne_agg.copy()
    ne_next["t_bin"] = ne_next["t_bin"] - 1  # shift back so label at bin t = dose at bin t+1
    ne_next = ne_next.rename(columns={"ne_dose": "next_ne_dose"})
    ne_next = ne_next[(ne_next["t_bin"] >= 0) & (ne_next["t_bin"] < N_BINS)]
    features = features.merge(ne_next, on=["stay_id", "t_bin"], how="left")
    features["next_ne_dose"] = features["next_ne_dose"].fillna(0.0)

    # --- Weight ----
    features["weight_kg"] = features["stay_id"].map(weight_by_stay)
    features["weight_kg"] = features["weight_kg"].fillna(cohort_mean_weight)

    # Convert mcg/kg/min -> mcg/min for action binning.
    next_mcg_min = features["next_ne_dose"] * features["weight_kg"]
    features["next_ne_action_bin"] = _ne_action_bin_from_mcg_min(next_mcg_min).astype("int8")

    # --- Static demographics ----
    static = cohort[["stay_id", "age", "gender", "first_careunit"]].copy()
    static["gender_M"] = (static["gender"].astype(str).str.upper() == "M").astype("int8")
    for cu in CAREUNIT_VALUES:
        col = f"first_careunit_{_safe_slug(cu)}"
        static[col] = (static["first_careunit"] == cu).astype("int8")
    static["first_careunit_OTHER"] = (~static["first_careunit"].isin(CAREUNIT_VALUES)).astype(
        "int8"
    )
    static = static.drop(columns=["gender", "first_careunit"])
    features = features.merge(static, on="stay_id", how="left")

    # --- Missing flags BEFORE imputation ----
    features = _add_missing_flags(features, MISSING_FLAG_VARS)

    # --- Impute: ffill within stay, then cohort mean ----
    non_fill = {
        "stay_id", "subject_id", "t_bin", "vent_flag",
        "ne_dose_now", "next_ne_dose", "next_ne_action_bin",
        "weight_kg", "age", "gender_M",
        "urine_4h_ml",  # missing urine => treat as 0 ml
    }
    non_fill.update({c for c in features.columns if c.startswith("first_careunit_")})
    non_fill.update({c for c in features.columns if c.endswith("_was_missing")})
    fill_cols = [
        c for c in features.columns
        if c not in non_fill and pd.api.types.is_numeric_dtype(features[c])
    ]
    features, cohort_mean = _ffill_and_impute(features, fill_cols)
    features["urine_4h_ml"] = features["urine_4h_ml"].fillna(0.0)

    # Reorder: keys, action, static, vitals, labs, sofa, treatment, dose.
    key_cols = ["stay_id", "subject_id", "t_bin"]
    action_cols = ["next_ne_dose", "next_ne_action_bin", "ne_dose_now"]
    other = [c for c in features.columns if c not in key_cols + action_cols]
    features = features[key_cols + action_cols + other]

    # --- Trim bins beyond the patient's last observed data (optional cap) ----
    features = features.sort_values(["stay_id", "t_bin"]).reset_index(drop=True)

    # --- Stats ----
    stats = _compute_stats(features, fill_cols, cohort_mean)

    # --- Persist ----
    features.to_parquet(cache, index=False)
    stats_p.write_text(json.dumps(stats, indent=2, default=float))
    logger.info("Wrote %d rows to %s", len(features), cache)
    logger.info("Wrote stats to %s", stats_p)

    return features, stats


def _safe_slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in " /-()":
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _compute_stats(
    df: pd.DataFrame, fill_cols: list[str], cohort_mean: dict
) -> dict:
    stats: dict = {
        "version": DEFAULT_VERSION,
        "n_rows": int(len(df)),
        "n_stays": int(df["stay_id"].nunique()),
        "bin_hours": BIN_HOURS,
        "window_hours": WINDOW_HOURS,
        "ne_bin_edges_mcg_min": list(NE_BIN_EDGES_MCG_MIN),
        "variables": {},
    }
    for c in fill_cols:
        col = df[c]
        stats["variables"][c] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "median": float(col.median()),
            "cohort_mean_fill": float(cohort_mean.get(c, np.nan)),
            "missing_rate_raw": None,  # post-impute columns have 0 NaN here
        }
    # Action distribution.
    stats["action_bin_counts"] = (
        df["next_ne_action_bin"].value_counts().sort_index().to_dict()
    )
    stats["action_bin_counts"] = {int(k): int(v) for k, v in stats["action_bin_counts"].items()}
    # Pre-impute missingness — read from the _was_missing flags.
    miss_cols = [c for c in df.columns if c.endswith("_was_missing")]
    stats["raw_missing_rate"] = {
        c: float(df[c].mean()) for c in miss_cols
    }
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df, stats = build_features(use_cache=False)
    print("\n=== Features shape ===")
    print(df.shape)
    print("=== Action bin distribution ===")
    print(stats["action_bin_counts"])
    print("=== First columns ===")
    print(df.columns[:20].tolist())


if __name__ == "__main__":
    main()

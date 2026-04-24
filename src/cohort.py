"""Sepsis-3 cohort extraction for vasopressor recommendation task.

Implements the cohort definition from PROJECT_PLAN §2.1 (13,071 stays):
  1. sepsis-3 (mimiciv_derived.sepsis3)
  2. adult (age >= 18)
  3. first ICU stay per subject
  4. ICU LOS >= 24h
  5. received NE-equivalent at any time (> 0 mcg/kg/min)
  6. NE first start time strictly after intime (no external-transfer carryovers)
  7. cumulative NE-equivalent time >= 1h (drops one-off boluses)

Public API:
    load_cohort(version: str = "v1", use_cache: bool = True) -> pd.DataFrame
    compute_consort_flow() -> pd.DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine

logger = logging.getLogger(__name__)

# Paths are resolved relative to the project root (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Cohort version tag — bump if the SQL definition changes.
DEFAULT_VERSION = "v1"

# NE-equivalent dose column name in mimiciv_derived.norepinephrine_equivalent_dose
_NE_DOSE_COL = "norepinephrine_equivalent_dose"

# Core cohort SQL — mirrors scripts/quick_consort_eb.sql F-step (13,071 stays).
# Enriched with demographics, mortality, first_day_sofa.
_COHORT_SQL = """
WITH base AS (
    SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.los,
           i.first_careunit,
           a.age
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_derived.age a USING (hadm_id)
    WHERE a.age >= 18 AND i.los >= 1.0
      AND EXISTS (SELECT 1 FROM mimiciv_derived.sepsis3 s WHERE s.stay_id = i.stay_id)
),
first_stay AS (
    SELECT stay_id, subject_id, hadm_id, intime, outtime, los, first_careunit, age
    FROM (
        SELECT b.*, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS sn
        FROM base b
    ) x
    WHERE sn = 1
),
ne_per_stay AS (
    SELECT n.stay_id,
           MIN(n.starttime) AS first_ne_time,
           MAX(n.endtime)   AS last_ne_time,
           SUM(EXTRACT(EPOCH FROM (n.endtime - n.starttime)))/3600.0 AS cum_ne_hours,
           MAX(n.norepinephrine_equivalent_dose) AS peak_ne_equiv
    FROM mimiciv_derived.norepinephrine_equivalent_dose n
    WHERE n.norepinephrine_equivalent_dose > 0
    GROUP BY n.stay_id
),
sep_onset AS (
    SELECT stay_id, MIN(sofa_time) AS sepsis3_onset_time
    FROM mimiciv_derived.sepsis3
    GROUP BY stay_id
)
SELECT
    f.subject_id,
    f.stay_id,
    f.hadm_id,
    f.intime,
    f.outtime,
    f.los,
    f.first_careunit,
    f.age,
    p.gender,
    p.dod,
    adm.race,
    adm.hospital_expire_flag,
    adm.deathtime,
    adm.admittime,
    adm.dischtime,
    so.sepsis3_onset_time,
    ne.first_ne_time,
    ne.last_ne_time,
    ne.cum_ne_hours,
    ne.peak_ne_equiv,
    fds.sofa AS first_day_sofa
FROM first_stay f
JOIN ne_per_stay ne USING (stay_id)
LEFT JOIN sep_onset so USING (stay_id)
LEFT JOIN mimiciv_hosp.patients p ON p.subject_id = f.subject_id
LEFT JOIN mimiciv_hosp.admissions adm ON adm.hadm_id = f.hadm_id
LEFT JOIN mimiciv_derived.first_day_sofa fds ON fds.stay_id = f.stay_id
WHERE ne.first_ne_time > f.intime
  AND ne.cum_ne_hours >= 1.0
ORDER BY f.subject_id, f.intime
"""

# Consort flow SQL — reproduces the 7-step inclusion funnel.
_CONSORT_SQL = """
WITH
icu_all AS (
    SELECT i.stay_id, i.subject_id, i.intime, i.los, a.age
    FROM mimiciv_icu.icustays i
    LEFT JOIN mimiciv_derived.age a USING (hadm_id)
),
sepsis_all AS (
    SELECT ia.*
    FROM icu_all ia
    WHERE EXISTS (SELECT 1 FROM mimiciv_derived.sepsis3 s WHERE s.stay_id = ia.stay_id)
),
sepsis_adult AS (
    SELECT * FROM sepsis_all WHERE age >= 18
),
sepsis_adult_los AS (
    SELECT * FROM sepsis_adult WHERE los >= 1.0
),
first_stay AS (
    SELECT * FROM (
        SELECT s.*, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS sn
        FROM sepsis_adult_los s
    ) x WHERE sn = 1
),
ne_per_stay AS (
    SELECT n.stay_id,
           MIN(n.starttime) AS first_ne_time,
           SUM(EXTRACT(EPOCH FROM (n.endtime - n.starttime)))/3600.0 AS cum_hours
    FROM mimiciv_derived.norepinephrine_equivalent_dose n
    WHERE n.norepinephrine_equivalent_dose > 0
    GROUP BY n.stay_id
),
joined AS (
    SELECT f.*, ne.first_ne_time, ne.cum_hours
    FROM first_stay f
    LEFT JOIN ne_per_stay ne USING (stay_id)
)
SELECT step, n_stays, n_subjects FROM (
    SELECT 1 AS ord, '1_sepsis3_all' AS step,
           COUNT(DISTINCT stay_id)::bigint AS n_stays,
           COUNT(DISTINCT subject_id)::bigint AS n_subjects
    FROM sepsis_all
    UNION ALL SELECT 2, '2_adult_ge18',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id) FROM sepsis_adult
    UNION ALL SELECT 3, '3_plus_los_ge_24h',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id) FROM sepsis_adult_los
    UNION ALL SELECT 4, '4_first_icu_stay',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id) FROM first_stay
    UNION ALL SELECT 5, '5_got_ne_ever',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id)
           FROM joined WHERE first_ne_time IS NOT NULL
    UNION ALL SELECT 6, '6_ne_start_after_intime',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id)
           FROM joined WHERE first_ne_time > intime
    UNION ALL SELECT 7, '7_cum_ne_hours_ge_1',
           COUNT(DISTINCT stay_id), COUNT(DISTINCT subject_id)
           FROM joined WHERE first_ne_time > intime AND cum_hours >= 1.0
) t
ORDER BY ord
"""


def _cache_path(version: str) -> Path:
    return DATA_DIR / f"cohort_{version}.parquet"


def _consort_path(version: str) -> Path:
    return DATA_DIR / f"cohort_{version}_consort.csv"


def _query_cohort() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(_COHORT_SQL), conn)
    return df


def load_cohort(version: str = DEFAULT_VERSION, use_cache: bool = True) -> pd.DataFrame:
    """Load sepsis-3 NE cohort. Uses parquet cache if available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(version)
    if use_cache and cache.exists():
        logger.info("Loading cohort from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Querying cohort from DB (version=%s)", version)
    df = _query_cohort()
    df.to_parquet(cache, index=False)
    logger.info("Cached %d rows to %s", len(df), cache)
    return df


def compute_consort_flow() -> pd.DataFrame:
    """Return the 7-step consort flow table as a DataFrame (step, n_stays, n_subjects)."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(_CONSORT_SQL), conn)
    return df


def save_consort_flow(version: str = DEFAULT_VERSION) -> Path:
    """Compute consort flow and persist to CSV. Returns path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = compute_consort_flow()
    out = _consort_path(version)
    df.to_csv(out, index=False)
    logger.info("Saved consort flow (%d steps) to %s", len(df), out)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_cohort(use_cache=False)
    n_stays = df["stay_id"].nunique()
    n_subjects = df["subject_id"].nunique()
    logger.info("Cohort: %d stays, %d subjects (expected 13,071)", n_stays, n_subjects)

    consort = compute_consort_flow()
    consort.to_csv(_consort_path(DEFAULT_VERSION), index=False)
    # Print aggregated counts only (no PHI).
    print("\n=== Consort flow ===")
    print(consort.to_string(index=False))
    print("\n=== Summary ===")
    print(f"Final cohort: {n_stays} stays / {n_subjects} subjects")
    print(f"Age  mean={df['age'].mean():.1f}  median={df['age'].median():.1f}")
    if "gender" in df.columns:
        g = df["gender"].value_counts(dropna=False)
        print(f"Gender: {g.to_dict()}")
    if "hospital_expire_flag" in df.columns:
        mort = df["hospital_expire_flag"].mean()
        print(f"In-hospital mortality: {mort:.3f}")


if __name__ == "__main__":
    main()

"""Smoke tests for features & splits modules.

Tests marked with `@pytest.mark.db` hit the PostgreSQL derived schema and are
skipped in CI/offline environments.  The pure-logic checks (splits disjointness,
NE bin edges, slug helper) run without a database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    N_BINS,
    NE_BIN_EDGES_MCG_MIN,
    _ne_action_bin_from_mcg_min,
    _safe_slug,
)
from src.splits import _assert_disjoint, make_splits


# --- Pure-logic tests -------------------------------------------------------


def test_ne_bin_edges_match_plan() -> None:
    assert NE_BIN_EDGES_MCG_MIN == (0.0, 8.4, 20.28, 50.0)


def test_ne_bin_classification() -> None:
    doses = pd.Series([0.0, 0.01, 8.4, 8.41, 20.28, 20.29, 50.0, 50.01, 120.0])
    bins = _ne_action_bin_from_mcg_min(doses).tolist()
    # 0 -> 0, just-above-0 -> 1, boundary inclusive on the lower side.
    assert bins == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_safe_slug() -> None:
    assert _safe_slug("Medical Intensive Care Unit (MICU)") == "Medical_Intensive_Care_Unit_MICU"
    assert _safe_slug("A/B-C") == "A_B_C"


def _fake_cohort(n_subj: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subj = np.arange(1_000_000, 1_000_000 + n_subj)
    stays = subj + 500  # 1:1 mapping
    intime = pd.Timestamp("2150-01-01") + pd.to_timedelta(rng.integers(0, 365 * 10, n_subj), "D")
    return pd.DataFrame(
        {
            "subject_id": subj,
            "stay_id": stays,
            "intime": intime,
            "age": rng.normal(65, 15, n_subj),
            "gender": rng.choice(["M", "F"], n_subj),
            "hospital_expire_flag": rng.choice([0, 1], n_subj, p=[0.77, 0.23]),
            "peak_ne_equiv": rng.uniform(0, 0.5, n_subj),
        }
    )


def test_subject_random_split_disjoint_and_ratios() -> None:
    cohort = _fake_cohort(n_subj=2000, seed=42)
    splits = make_splits(cohort, strategy="subject_random", seed=42)
    _assert_disjoint(splits)

    total = sum(len(v) for v in splits.values())
    assert total == len(cohort)
    assert 0.65 <= len(splits["train"]) / total <= 0.75
    assert 0.12 <= len(splits["val"]) / total <= 0.18
    assert 0.12 <= len(splits["test"]) / total <= 0.18


def test_time_split_disjoint() -> None:
    cohort = _fake_cohort(n_subj=1500, seed=1)
    splits = make_splits(cohort, strategy="time")
    _assert_disjoint(splits)
    # Train should be before val before test in mean intime.
    intimes = cohort.set_index("stay_id")["intime"]
    means = {k: intimes.loc[v].mean() for k, v in splits.items()}
    assert means["train"] <= means["val"] <= means["test"]


def test_split_rejects_bad_fractions() -> None:
    cohort = _fake_cohort(100)
    with pytest.raises(ValueError):
        make_splits(cohort, val_frac=0.0)
    with pytest.raises(ValueError):
        make_splits(cohort, val_frac=0.5, test_frac=0.6)
    with pytest.raises(ValueError):
        make_splits(cohort, strategy="bogus")


# --- Integration (DB-backed) ------------------------------------------------


@pytest.mark.db
def test_build_features_shape_matches_cohort() -> None:
    """Cache-backed smoke test — requires a prior build."""
    from src.features import build_features

    from src.cohort import load_cohort

    cohort = load_cohort()
    features, stats = build_features(cohort=cohort, use_cache=True)

    # Every cohort stay must appear.
    assert features["stay_id"].nunique() == cohort["stay_id"].nunique()
    # t_bin range.
    assert features["t_bin"].min() == 0
    assert features["t_bin"].max() == N_BINS - 1
    # Expected row count: n_stays * N_BINS (skeleton-driven, no trimming).
    assert len(features) == cohort["stay_id"].nunique() * N_BINS
    # Every action bin value in {0..4}.
    assert set(features["next_ne_action_bin"].unique()).issubset({0, 1, 2, 3, 4})
    # Stats carry variable mean/std.
    assert "variables" in stats
    assert "ne_bin_edges_mcg_min" in stats

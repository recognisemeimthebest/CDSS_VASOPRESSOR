"""Train/validation/test split helpers for the NE cohort.

Two strategies:
  * subject_random (default): random split at the subject level so that no
    subject appears in more than one fold (prevents patient-level leakage).
    For our cohort, 1 subject = 1 stay (first-stay-only), so this reduces
    to a random stay_id split — but we keep the subject abstraction for
    future-proofing.
  * time: chronological split on `intime`.  MIMIC-IV's anchor_year is only
    a 3-year group, so the boundaries are approximate.

Usage:
    from src.splits import make_splits
    splits = make_splits(cohort)  # {'train': [...], 'val': [...], 'test': [...]}

Writes `data/splits_{version}.json` with stay_ids per fold and a small meta block.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VERSION = "v1"

SplitStrategy = Literal["subject_random", "time"]


def _splits_path(version: str) -> Path:
    return DATA_DIR / f"splits_{version}.json"


def make_splits(
    cohort: pd.DataFrame,
    strategy: SplitStrategy = "subject_random",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return a dict of fold -> np.ndarray[stay_id].

    Guarantees no subject_id leakage across folds.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("val_frac and test_frac must lie in (0, 1)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")
    if strategy not in ("subject_random", "time"):
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if strategy == "subject_random":
        return _subject_random_split(cohort, val_frac, test_frac, seed)
    return _time_split(cohort, val_frac, test_frac)


def _subject_random_split(
    cohort: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> dict[str, np.ndarray]:
    subjects = cohort["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects)

    n = len(perm)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_subj = set(perm[:n_test].tolist())
    val_subj = set(perm[n_test : n_test + n_val].tolist())
    train_subj = set(perm[n_test + n_val :].tolist())

    def _stays(subj_set: set[int]) -> np.ndarray:
        return (
            cohort.loc[cohort["subject_id"].isin(subj_set), "stay_id"]
            .to_numpy()
            .astype(np.int64)
        )

    splits = {
        "train": _stays(train_subj),
        "val": _stays(val_subj),
        "test": _stays(test_subj),
    }
    _assert_disjoint(splits)
    return splits


def _time_split(
    cohort: pd.DataFrame, val_frac: float, test_frac: float
) -> dict[str, np.ndarray]:
    if "intime" not in cohort.columns:
        raise ValueError("time split requires 'intime' column on cohort")
    ordered = cohort.sort_values("intime").reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    train_end = n - n_test - n_val
    val_end = train_end + n_val

    # Respect subject boundary: if the same subject would straddle a boundary,
    # push all of their stays into the later fold.  In practice, first-stay
    # cohort means 1 subject = 1 stay, so this is a no-op here, but we keep
    # it safe.
    def _bucket(idx: int) -> str:
        return "train" if idx < train_end else ("val" if idx < val_end else "test")

    ordered["_fold"] = [_bucket(i) for i in range(n)]
    # Resolve any subject appearing in multiple folds -> collapse to latest.
    fold_order = {"train": 0, "val": 1, "test": 2}
    latest = (
        ordered.groupby("subject_id")["_fold"]
        .agg(lambda s: max(s, key=lambda f: fold_order[f]))
        .to_dict()
    )
    ordered["_fold"] = ordered["subject_id"].map(latest)
    splits = {
        fold: ordered.loc[ordered["_fold"] == fold, "stay_id"]
        .to_numpy()
        .astype(np.int64)
        for fold in ("train", "val", "test")
    }
    _assert_disjoint(splits)
    return splits


def _assert_disjoint(splits: dict[str, np.ndarray]) -> None:
    sets = {k: set(v.tolist()) for k, v in splits.items()}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                inter = sets[a] & sets[b]
                if inter:
                    raise AssertionError(f"Split leak {a}<->{b}: {len(inter)} overlap")


def save_splits(
    splits: dict[str, np.ndarray],
    cohort: pd.DataFrame,
    strategy: str,
    version: str = DEFAULT_VERSION,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Include subject_id list so leakage check is obvious from the JSON.
    stay_to_subj = dict(zip(cohort["stay_id"].astype(int), cohort["subject_id"].astype(int)))
    payload = {
        "version": version,
        "strategy": strategy,
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "train": [int(x) for x in splits["train"].tolist()],
        "val": [int(x) for x in splits["val"].tolist()],
        "test": [int(x) for x in splits["test"].tolist()],
        "subject_sizes": {
            k: len({stay_to_subj.get(int(sid)) for sid in v.tolist()})
            for k, v in splits.items()
        },
    }
    out = _splits_path(version)
    out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Saved splits to %s (train=%d val=%d test=%d)",
        out,
        payload["sizes"]["train"],
        payload["sizes"]["val"],
        payload["sizes"]["test"],
    )
    return out


def load_splits(version: str = DEFAULT_VERSION) -> dict[str, np.ndarray]:
    p = _splits_path(version)
    if not p.exists():
        raise FileNotFoundError(f"No splits file at {p}; run save_splits first")
    payload = json.loads(p.read_text())
    return {k: np.asarray(payload[k], dtype=np.int64) for k in ("train", "val", "test")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.cohort import load_cohort

    cohort = load_cohort()
    splits = make_splits(cohort, strategy="subject_random", seed=42)
    save_splits(splits, cohort, strategy="subject_random")
    for k, v in splits.items():
        print(f"{k:5s}: {len(v):>6d} stays")


if __name__ == "__main__":
    main()

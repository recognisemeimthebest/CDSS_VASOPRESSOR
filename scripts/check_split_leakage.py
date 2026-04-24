"""Quick check: subject_id leakage between train/val/test splits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cohort import load_cohort
from src.splits import load_splits


def main() -> None:
    cohort = load_cohort()
    splits = load_splits()
    subj = {k: set(cohort[cohort["stay_id"].isin(v)]["subject_id"]) for k, v in splits.items()}
    print(f"train: {len(subj['train'])} subjects, {len(splits['train'])} stays")
    print(f"val  : {len(subj['val'])} subjects, {len(splits['val'])} stays")
    print(f"test : {len(subj['test'])} subjects, {len(splits['test'])} stays")
    print(f"\nLeakage check (should all be 0):")
    print(f"  train ∩ val  = {len(subj['train'] & subj['val'])}")
    print(f"  train ∩ test = {len(subj['train'] & subj['test'])}")
    print(f"  val   ∩ test = {len(subj['val'] & subj['test'])}")


if __name__ == "__main__":
    main()

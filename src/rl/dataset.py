"""Stay -> trajectory conversion + reward shaping for offline RL.

Each ICU stay becomes one episode of length T (<=18). A transition is:

    (s_t, a_t, r_t, s_{t+1}, done, stay_id, t)

where

    * s_t is the z-scored feature vector at bin t.
    * a_t = ``next_ne_action_bin`` at bin t (the action the clinician took for
      the NEXT 4h window). Our models recommend the same quantity, so at the
      last bin (t = len(stay)-1) the label still describes the immediate-next
      4h period.
    * r_t = intermediate (SOFA/lactate deltas) + terminal (+-15 on last bin).
    * done is 1 only at the last bin of a stay.

Reward design (Raghu 2017 / AI Clinician style, honesty-guarded):

    r_t = -c_sofa * (SOFA_{t+1} - SOFA_t) - c_lactate * (lac_{t+1} - lac_t)
        + (TERMINAL_REWARD if alive else -TERMINAL_REWARD) if done else 0

    Defaults: c_sofa = 0.025, c_lactate = 0.05, TERMINAL_REWARD = 15.0
    gamma = 0.99
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training.data import (
    FeatureMatrix,
    ScalerState,
    build_xy,
    feature_columns,
    load_features,
    load_splits,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# --- Reward defaults (Raghu 2017 Table 2) -----------------------------------
DEFAULT_C_SOFA: float = 0.025
DEFAULT_C_LACTATE: float = 0.05
DEFAULT_TERMINAL_REWARD: float = 15.0
DEFAULT_GAMMA: float = 0.99

N_ACTIONS: int = 5


# -----------------------------------------------------------------------------
# Public dataclasses
# -----------------------------------------------------------------------------


@dataclass
class RewardConfig:
    c_sofa: float = DEFAULT_C_SOFA
    c_lactate: float = DEFAULT_C_LACTATE
    terminal_reward: float = DEFAULT_TERMINAL_REWARD
    gamma: float = DEFAULT_GAMMA
    # Which terminal label to use. "hospital" = in-hospital mortality flag (always
    # present); "90day" = dod within 90 days of dischtime (optional extra).
    terminal_target: str = "hospital"


@dataclass
class TrajectoryDataset:
    """Flat arrays of transitions + per-stay metadata.

    Shape conventions:
        states, next_states -> (N, F) float32
        actions, next_actions -> (N,) int64 in [0, N_ACTIONS)
        rewards -> (N,) float32
        dones  -> (N,) uint8, 1 at last bin of a stay
        stay_ids, t_bins -> (N,) int64
        is_terminal_mask -> (N,) uint8 (alias of dones)
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_actions: np.ndarray
    dones: np.ndarray
    stay_ids: np.ndarray
    t_bins: np.ndarray
    feature_names: list[str]
    fold: str
    # Per-stay arrays (length = n_stays).
    stay_lengths: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    stay_id_order: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    terminal_alive: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    @property
    def n_features(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_transitions(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_stays(self) -> int:
        return int(self.stay_id_order.shape[0])


@dataclass
class RLData:
    """Train/val/test trajectory bundles + scaler + global config."""

    train: TrajectoryDataset
    val: TrajectoryDataset
    test: TrajectoryDataset
    scaler: ScalerState
    reward_config: RewardConfig
    feature_names: list[str]
    n_actions: int = N_ACTIONS


# -----------------------------------------------------------------------------
# Cohort mortality loading
# -----------------------------------------------------------------------------


def _load_terminal_labels(
    cohort_path: Path, terminal_target: str
) -> dict[int, int]:
    """Return ``{stay_id: alive_flag (1=alive, 0=dead)}`` for each stay."""
    cohort = pd.read_parquet(cohort_path)
    if terminal_target == "hospital":
        dead = cohort["hospital_expire_flag"].fillna(0).astype(int)
    elif terminal_target == "90day":
        if "dod" not in cohort.columns or "dischtime" not in cohort.columns:
            raise KeyError("cohort missing dod/dischtime for 90day target")
        # Dead within 90 days of discharge.
        dod = pd.to_datetime(cohort["dod"], errors="coerce")
        disch = pd.to_datetime(cohort["dischtime"], errors="coerce")
        delta_days = (dod - disch).dt.total_seconds() / 86400.0
        dead = ((delta_days >= 0) & (delta_days <= 90)).astype(int)
        # Hospital deaths without dod -> still dead.
        dead = dead | cohort["hospital_expire_flag"].fillna(0).astype(int)
    else:
        raise ValueError(f"Unknown terminal_target={terminal_target!r}")
    alive = 1 - dead
    return dict(zip(cohort["stay_id"].astype(int), alive.astype(int)))


# -----------------------------------------------------------------------------
# Trajectory building
# -----------------------------------------------------------------------------


def _stay_index_map(stay_ids: np.ndarray, t_bins: np.ndarray) -> np.ndarray:
    """Return the argsort that orders rows by (stay_id, t_bin).

    ``build_xy(..., flatten='last_bin')`` already sorts this way, but we
    re-sort defensively.
    """
    order = np.lexsort((t_bins, stay_ids))
    return order


def _compute_next_state_indices(
    stay_ids: np.ndarray, t_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each row i return the row j that is (stay_ids[i], t_bins[i]+1).

    If no such row exists (last bin of the stay), we set j = i and mark the
    row as terminal via the returned ``done`` array.
    """
    # Assumes rows are sorted by (stay_id, t_bin).
    n = stay_ids.shape[0]
    # Same-stay neighbour check.
    same_stay = np.zeros(n, dtype=bool)
    same_stay[:-1] = (stay_ids[:-1] == stay_ids[1:]) & (t_bins[1:] == t_bins[:-1] + 1)
    next_idx = np.where(same_stay, np.arange(n) + 1, np.arange(n))
    done = (~same_stay).astype(np.uint8)
    return next_idx.astype(np.int64), done


def _cohort_fill(series: pd.Series, cohort_mean: float) -> pd.Series:
    return series.fillna(cohort_mean).astype(np.float32)


def _build_trajectory_for_fold(
    features_df: pd.DataFrame,
    fm: FeatureMatrix,
    fold_name: str,
    terminal_alive: dict[int, int],
    reward_cfg: RewardConfig,
) -> TrajectoryDataset:
    """Construct a TrajectoryDataset from a per-bin FeatureMatrix.

    We need raw (unscaled) SOFA + lactate values to compute reward deltas, so
    we pull them from ``features_df`` using (stay_id, t_bin).
    """
    # fm.X is z-scored. Rows already sorted by stay_id,t_bin (build_xy sorts).
    order = _stay_index_map(fm.stay_ids, fm.t_bins)
    X = fm.X[order]
    stays = fm.stay_ids[order]
    tbins = fm.t_bins[order]
    actions = fm.y[order].astype(np.int64) if fm.task == "classification" else None
    if actions is None:
        # dataset.py only called with task='cls'; action bins live in the label.
        raise ValueError("TrajectoryDataset expects classification labels (action bins)")

    # Pull raw SOFA / lactate aligned to (stay_id, t_bin).
    keys = pd.DataFrame({"stay_id": stays, "t_bin": tbins})
    raw_cols = ["stay_id", "t_bin", "sofa", "lactate_last"]
    available = [c for c in raw_cols if c in features_df.columns]
    raw = features_df[available].copy()
    if "lactate_last" not in raw.columns:
        raw["lactate_last"] = np.nan
    raw = keys.merge(raw, on=["stay_id", "t_bin"], how="left")
    sofa_mean = float(features_df["sofa"].mean())
    lac_mean = float(features_df["lactate_last"].mean()) if "lactate_last" in features_df.columns else 2.0
    raw["sofa"] = _cohort_fill(raw["sofa"], sofa_mean)
    raw["lactate_last"] = _cohort_fill(raw["lactate_last"], lac_mean)

    next_idx, done = _compute_next_state_indices(stays, tbins)

    # Next state/action come from the next row unless terminal.
    next_states = X[next_idx]
    next_actions = actions[next_idx]

    sofa = raw["sofa"].to_numpy(dtype=np.float32)
    lac = raw["lactate_last"].to_numpy(dtype=np.float32)

    # Intermediate reward: negative change (improvement == positive reward).
    delta_sofa = np.where(done == 0, sofa[next_idx] - sofa, 0.0)
    delta_lac = np.where(done == 0, lac[next_idx] - lac, 0.0)
    intermediate = -reward_cfg.c_sofa * delta_sofa - reward_cfg.c_lactate * delta_lac

    # Terminal reward: alive=+R, dead=-R.
    alive_arr = np.asarray(
        [terminal_alive.get(int(s), 1) for s in stays], dtype=np.int8
    )
    terminal = np.where(
        done == 1,
        np.where(alive_arr == 1, reward_cfg.terminal_reward, -reward_cfg.terminal_reward),
        0.0,
    ).astype(np.float32)
    rewards = (intermediate + terminal).astype(np.float32)

    # Per-stay metadata.
    stay_id_order, counts = np.unique(stays, return_counts=True)
    # Preserve stay order (np.unique sorts, that's fine).
    stay_alive = np.asarray(
        [terminal_alive.get(int(s), 1) for s in stay_id_order], dtype=np.int8
    )

    return TrajectoryDataset(
        states=X.astype(np.float32),
        actions=actions.astype(np.int64),
        rewards=rewards,
        next_states=next_states.astype(np.float32),
        next_actions=next_actions.astype(np.int64),
        dones=done.astype(np.uint8),
        stay_ids=stays.astype(np.int64),
        t_bins=tbins.astype(np.int64),
        feature_names=list(fm.feature_names),
        fold=fold_name,
        stay_lengths=counts.astype(np.int64),
        stay_id_order=stay_id_order.astype(np.int64),
        terminal_alive=stay_alive,
    )


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------


def load_trajectories(
    dataset_version: str = "v1",
    reward_config: RewardConfig | None = None,
    features_df: pd.DataFrame | None = None,
    splits: dict[str, np.ndarray] | None = None,
    cohort_path: Path | None = None,
) -> RLData:
    """Top-level factory: features parquet + splits + cohort -> RLData.

    Scaling uses the **train** fold only (same ScalerState as supervised).
    """
    reward_config = reward_config or RewardConfig()

    if features_df is None:
        features_df = load_features(version=dataset_version)
    if splits is None:
        splits = load_splits(version=dataset_version)
    if cohort_path is None:
        cohort_path = DATA_DIR / f"cohort_{dataset_version}.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Missing cohort parquet at {cohort_path}. Run src.cohort first."
        )

    terminal_alive = _load_terminal_labels(cohort_path, reward_config.terminal_target)

    folds, scaler = build_xy(
        features_df, splits, task="cls", flatten="last_bin"
    )

    bundle: dict[str, TrajectoryDataset] = {}
    for fold_name, fm in folds.items():
        bundle[fold_name] = _build_trajectory_for_fold(
            features_df, fm, fold_name, terminal_alive, reward_config
        )
    return RLData(
        train=bundle["train"],
        val=bundle["val"],
        test=bundle["test"],
        scaler=scaler,
        reward_config=reward_config,
        feature_names=list(scaler.feature_names),
    )


# -----------------------------------------------------------------------------
# Batch iteration helpers
# -----------------------------------------------------------------------------


def iter_transition_batches(
    ds: TrajectoryDataset,
    batch_size: int,
    seed: int,
    n_batches: int | None = None,
    shuffle: bool = True,
) -> Any:
    """Yield dicts of torch-ready numpy arrays with uniform transition sampling.

    ``n_batches=None`` loops forever (caller decides); otherwise yields that
    many batches.
    """
    import itertools

    rng = np.random.default_rng(seed)
    n = ds.n_transitions
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    iterator = itertools.cycle(np.array_split(idx, max(1, n // batch_size)))
    count = 0
    for chunk in iterator:
        if n_batches is not None and count >= n_batches:
            return
        yield {
            "state": ds.states[chunk],
            "action": ds.actions[chunk],
            "reward": ds.rewards[chunk],
            "next_state": ds.next_states[chunk],
            "next_action": ds.next_actions[chunk],
            "done": ds.dones[chunk],
        }
        count += 1

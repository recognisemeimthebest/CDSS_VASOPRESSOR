"""Tests for Phase 4 offline RL package.

Pure-logic only (no DB, no trained models required). Uses a small synthetic
trajectory dataset so every test runs in seconds on CPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import torch  # noqa: F401  (ensure torch loads before pandas downstream)

from src.rl import ope as rl_ope
from src.rl.algorithms.bc import BehaviorCloning
from src.rl.dataset import (
    N_ACTIONS,
    RewardConfig,
    TrajectoryDataset,
    _build_trajectory_for_fold,
    _compute_next_state_indices,
)
from src.training.data import FeatureMatrix


# -----------------------------------------------------------------------------
# Synthetic features + FeatureMatrix fixture
# -----------------------------------------------------------------------------


def _synth_features(n_stays: int = 10, n_bins: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stays = np.repeat(np.arange(1000, 1000 + n_stays), n_bins)
    t_bins = np.tile(np.arange(n_bins), n_stays)
    n = n_stays * n_bins
    sofa = rng.integers(0, 15, n).astype(np.float32)
    lactate = rng.normal(2.0, 1.0, n).astype(np.float32)
    hr = rng.normal(80, 10, n).astype(np.float32)
    actions = rng.integers(0, N_ACTIONS, n).astype(np.int64)
    return pd.DataFrame(
        {
            "stay_id": stays,
            "subject_id": stays,
            "t_bin": t_bins,
            "sofa": sofa,
            "lactate_last": lactate,
            "hr_mean": hr,
            "vent_flag": rng.integers(0, 2, n).astype(np.int8),
            "next_ne_dose": np.zeros(n, dtype=np.float32),
            "next_ne_action_bin": actions,
            "ne_dose_now": np.zeros(n, dtype=np.float32),
        }
    )


def _synth_feature_matrix(df: pd.DataFrame) -> FeatureMatrix:
    feat_cols = ["sofa", "lactate_last", "hr_mean", "vent_flag"]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["next_ne_action_bin"].to_numpy(dtype=np.int64)
    return FeatureMatrix(
        X=X,
        y=y,
        stay_ids=df["stay_id"].to_numpy(dtype=np.int64),
        t_bins=df["t_bin"].to_numpy(dtype=np.int64),
        feature_names=feat_cols,
        fold="train",
        task="classification",
        flatten="last_bin",
    )


def _alive_map(n_stays: int) -> dict[int, int]:
    # Alternate alive/dead.
    return {1000 + i: i % 2 for i in range(n_stays)}


# -----------------------------------------------------------------------------
# Trajectory construction
# -----------------------------------------------------------------------------


def test_next_state_index_on_contiguous_stay() -> None:
    stays = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    t_bins = np.array([0, 1, 2, 0, 1, 0], dtype=np.int64)
    nxt, done = _compute_next_state_indices(stays, t_bins)
    # Last bin of each stay -> done=1, next==self.
    assert list(done) == [0, 0, 1, 0, 1, 1]
    assert list(nxt) == [1, 2, 2, 4, 4, 5]


def test_trajectory_terminal_reward_sign() -> None:
    df = _synth_features(n_stays=4, n_bins=3)
    fm = _synth_feature_matrix(df)
    alive = {1000: 1, 1001: 0, 1002: 1, 1003: 0}
    cfg = RewardConfig(c_sofa=0.0, c_lactate=0.0, terminal_reward=15.0)
    td = _build_trajectory_for_fold(df, fm, "train", alive, cfg)

    # Only last bin of each stay carries a non-zero reward.
    for stay, is_alive in alive.items():
        mask = (td.stay_ids == stay)
        # done flag is 1 exactly at last bin.
        assert td.dones[mask].sum() == 1
        last_row = np.where((td.stay_ids == stay) & (td.dones == 1))[0][0]
        expected = 15.0 if is_alive else -15.0
        assert td.rewards[last_row] == pytest.approx(expected, rel=1e-5)
        # Non-terminal rewards zero because c_sofa = c_lactate = 0.
        intermediate = td.rewards[(td.stay_ids == stay) & (td.dones == 0)]
        assert np.all(intermediate == 0.0)


def test_trajectory_intermediate_signs() -> None:
    # Handcrafted: rising SOFA should produce negative intermediate reward.
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "subject_id": [1, 1, 1],
            "t_bin": [0, 1, 2],
            "sofa": [2.0, 4.0, 6.0],
            "lactate_last": [1.0, 1.0, 1.0],
            "hr_mean": [80.0, 80.0, 80.0],
            "vent_flag": [0, 0, 0],
            "next_ne_dose": [0.0, 0.0, 0.0],
            "next_ne_action_bin": [0, 0, 0],
            "ne_dose_now": [0.0, 0.0, 0.0],
        }
    )
    fm = _synth_feature_matrix(df)
    alive = {1: 1}
    cfg = RewardConfig(c_sofa=0.1, c_lactate=0.0, terminal_reward=10.0)
    td = _build_trajectory_for_fold(df, fm, "train", alive, cfg)

    # At t=0 -> t=1 SOFA 2->4 (delta=+2) -> reward = -0.1 * 2 = -0.2
    assert td.rewards[0] == pytest.approx(-0.2, rel=1e-4)
    # At t=1 -> t=2 SOFA 4->6 (delta=+2) -> reward = -0.1 * 2 = -0.2 (non-terminal)
    assert td.rewards[1] == pytest.approx(-0.2, rel=1e-4)
    # At t=2 terminal -> +10 (alive).
    assert td.rewards[2] == pytest.approx(10.0, rel=1e-4)
    # Done pattern.
    assert list(td.dones) == [0, 0, 1]
    # Trajectory length matches stay length.
    assert td.n_transitions == 3


def test_trajectory_stay_lengths_match() -> None:
    df = _synth_features(n_stays=6, n_bins=4)
    fm = _synth_feature_matrix(df)
    alive = _alive_map(6)
    cfg = RewardConfig()
    td = _build_trajectory_for_fold(df, fm, "train", alive, cfg)
    assert td.n_stays == 6
    assert int(td.stay_lengths.sum()) == td.n_transitions
    assert np.all(td.stay_lengths == 4)


# -----------------------------------------------------------------------------
# OPE sanity
# -----------------------------------------------------------------------------


def _trivial_dataset() -> TrajectoryDataset:
    # 4 stays, 3 bins each; action = 0 always; reward = +1 at terminal only.
    n_stays = 4
    n_bins = 3
    rng = np.random.default_rng(0)
    total = n_stays * n_bins
    states = rng.normal(size=(total, 4)).astype(np.float32)
    actions = np.zeros(total, dtype=np.int64)
    rewards = np.zeros(total, dtype=np.float32)
    dones = np.zeros(total, dtype=np.uint8)
    stay_ids = np.repeat(np.arange(n_stays), n_bins).astype(np.int64)
    t_bins = np.tile(np.arange(n_bins), n_stays).astype(np.int64)
    next_states = states.copy()
    # Set rewards + dones on last bin.
    for s in range(n_stays):
        idx = np.where(stay_ids == s)[0][-1]
        rewards[idx] = 1.0
        dones[idx] = 1
    next_states[:-1] = states[1:]  # rough; terminal rows unchanged
    return TrajectoryDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_actions=actions.copy(),
        dones=dones,
        stay_ids=stay_ids,
        t_bins=t_bins,
        feature_names=[f"f{i}" for i in range(4)],
        fold="val",
        stay_lengths=np.full(n_stays, n_bins, dtype=np.int64),
        stay_id_order=np.arange(n_stays, dtype=np.int64),
        terminal_alive=np.ones(n_stays, dtype=np.int8),
    )


def test_wis_matches_behaviour_when_policies_equal() -> None:
    ds = _trivial_dataset()
    # Behaviour always picks action 0 with probability 1; policy too.
    p = np.eye(N_ACTIONS, dtype=np.float32)[ds.actions]
    wis = rl_ope.weighted_importance_sampling(
        ds, policy_proba=p, behaviour_proba=p, gamma=0.99
    )
    # All IS ratios = 1, so WIS == mean trajectory return.
    expected = float(np.mean([0.99 ** (3 - 1) * 1.0] * 4))
    assert wis.value == pytest.approx(expected, rel=1e-3)
    # ESS equals N_traj when all weights are equal.
    assert wis.ess_frac == pytest.approx(1.0, rel=1e-3)


def test_null_policy_zero_has_valid_shape() -> None:
    ds = _trivial_dataset()
    p = rl_ope.null_policy_proba(ds, "zero")
    assert p.shape == (ds.n_transitions, N_ACTIONS)
    # Action 0 dominates.
    assert np.all(p[:, 0] >= p[:, 1])
    # Rows sum close to 1 after softening.
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)


def test_ood_rate_uses_threshold() -> None:
    ds = _trivial_dataset()
    behav = np.full((ds.n_transitions, N_ACTIONS), 0.2, dtype=np.float32)
    # Policy picks action 4 always — behav=0.2 >= 0.05, so OOD should be 0.
    pol = np.eye(N_ACTIONS, dtype=np.float32)[np.full(ds.n_transitions, 4)]
    pol_actions = np.argmax(pol, axis=1)
    ood = rl_ope.ood_rate(pol_actions, behav, threshold=0.05)
    assert ood == 0.0
    ood_strict = rl_ope.ood_rate(pol_actions, behav, threshold=0.5)
    assert ood_strict == 1.0


# -----------------------------------------------------------------------------
# BC fit/predict shape
# -----------------------------------------------------------------------------


def test_bc_fit_predict_shapes(tmp_path) -> None:
    ds = _trivial_dataset()
    # Force CPU for the test so we don't need CUDA.
    device = torch.device("cpu")
    bc = BehaviorCloning.from_train(
        in_features=ds.n_features,
        n_actions=N_ACTIONS,
        y_train=ds.actions,
        encoder="mlp",
        learning_rate=1e-2,
        device=device,
    )
    batch = {
        "state": torch.from_numpy(ds.states).to(device),
        "action": torch.from_numpy(ds.actions).to(device),
        "reward": torch.from_numpy(ds.rewards).to(device),
        "next_state": torch.from_numpy(ds.next_states).to(device),
        "next_action": torch.from_numpy(ds.next_actions).to(device),
        "done": torch.from_numpy(ds.dones.astype(np.float32)).to(device),
    }
    # BatchNorm(1 layer in MLPEncoder) needs > 1 sample to compute running stats.
    bc.train()
    info = bc.update(batch)
    assert "loss" in info and np.isfinite(info["loss"])

    bc.eval()
    proba = bc.policy_proba(ds.states)
    assert proba.shape == (ds.n_transitions, N_ACTIONS)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    actions = bc.policy_action(ds.states)
    assert actions.shape == (ds.n_transitions,)
    assert actions.dtype == np.int64

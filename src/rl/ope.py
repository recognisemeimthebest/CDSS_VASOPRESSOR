"""Off-policy evaluation for offline RL policies.

Implements (ch07 mandate):

    * Weighted Importance Sampling (WIS) with per-step IS clipping.
    * Effective Sample Size (ESS).
    * Fitted Q-Evaluation (FQE) via a simple MLP Q-critic.
    * Bootstrap 95% CI on both estimators.
    * Null-policy comparisons: zero / uniform / constant-dose.
    * Clinician match rate, OOD rate.

Honesty guard: results are OPE estimates, NOT evidence of clinical benefit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from src.rl.dataset import N_ACTIONS, RewardConfig, TrajectoryDataset

logger = logging.getLogger(__name__)

# IS ratio clip bounds (Komorowski 2018 / Gottesman 2019).
DEFAULT_IS_CLIP = 1e3
DEFAULT_BEHAVIOUR_FLOOR = 1e-3  # lower bound for clinician proba to avoid div/0
DEFAULT_POLICY_FLOOR = 1e-3     # lower bound for policy proba in soft-epsilon


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _group_by_stay(ds: TrajectoryDataset) -> dict[int, np.ndarray]:
    """Return {stay_id -> sorted row indices} using t_bin ascending."""
    stays = ds.stay_ids
    order = np.lexsort((ds.t_bins, stays))
    out: dict[int, list[int]] = {}
    for i in order:
        out.setdefault(int(stays[i]), []).append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _soften_policy(
    p_policy: np.ndarray, floor: float = DEFAULT_POLICY_FLOOR
) -> np.ndarray:
    """Clip then renormalise to keep IS ratios finite."""
    p = np.clip(p_policy, floor, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def behaviour_proba_from_counts(
    actions: np.ndarray, n_actions: int
) -> np.ndarray:
    """Global empirical frequency (constant per state). Used as null baseline."""
    probs = np.bincount(actions, minlength=n_actions).astype(np.float64)
    probs /= max(probs.sum(), 1.0)
    return np.tile(probs.astype(np.float32), (len(actions), 1))


# -----------------------------------------------------------------------------
# WIS
# -----------------------------------------------------------------------------


@dataclass
class WISResult:
    value: float
    ess: float
    ess_frac: float
    n_trajectories: int
    weights: np.ndarray
    returns: np.ndarray


def weighted_importance_sampling(
    ds: TrajectoryDataset,
    policy_proba: np.ndarray,
    behaviour_proba: np.ndarray,
    gamma: float,
    is_clip: float = DEFAULT_IS_CLIP,
) -> WISResult:
    """Per-trajectory WIS following Raghu 2017 eq.(4).

    Returns WIS value + effective sample size. If ESS < 5% of N_traj the
    caller should treat the estimate as unreliable (ch07).
    """
    grouped = _group_by_stay(ds)
    n_traj = len(grouped)

    # Soft-clip both policies.
    p_policy = _soften_policy(policy_proba, floor=DEFAULT_POLICY_FLOOR)
    p_behav = np.clip(behaviour_proba, DEFAULT_BEHAVIOUR_FLOOR, 1.0)
    p_behav = p_behav / p_behav.sum(axis=1, keepdims=True)

    rho_traj = np.empty(n_traj, dtype=np.float64)
    returns_traj = np.empty(n_traj, dtype=np.float64)

    for k, (_stay_id, rows) in enumerate(grouped.items()):
        acts = ds.actions[rows]
        rews = ds.rewards[rows]
        pp = p_policy[rows, acts]
        pb = p_behav[rows, acts]
        ratios = pp / pb
        rho = float(np.clip(np.prod(ratios), 0.0, is_clip))
        rho_traj[k] = rho
        # Discounted return.
        t = np.arange(len(rews))
        disc = gamma ** t
        returns_traj[k] = float(np.sum(disc * rews))

    denom = rho_traj.sum() + 1e-12
    wis_value = float(np.sum(rho_traj * returns_traj) / denom)

    # ESS for self-normalised WIS.
    ess = float((rho_traj.sum() ** 2) / (np.sum(rho_traj ** 2) + 1e-12))
    ess_frac = ess / max(n_traj, 1)

    return WISResult(
        value=wis_value,
        ess=ess,
        ess_frac=ess_frac,
        n_trajectories=n_traj,
        weights=rho_traj,
        returns=returns_traj,
    )


def bootstrap_wis_ci(
    ds: TrajectoryDataset,
    policy_proba: np.ndarray,
    behaviour_proba: np.ndarray,
    gamma: float,
    n_boot: int = 1000,
    seed: int = 42,
    is_clip: float = DEFAULT_IS_CLIP,
) -> tuple[float, tuple[float, float], WISResult]:
    """Bootstrap CI on WIS by resampling *trajectories*."""
    full = weighted_importance_sampling(
        ds, policy_proba, behaviour_proba, gamma, is_clip=is_clip
    )
    rng = np.random.default_rng(seed)
    n = len(full.weights)
    values = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w = full.weights[idx]
        r = full.returns[idx]
        denom = w.sum() + 1e-12
        values[b] = float(np.sum(w * r) / denom)
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return full.value, (lo, hi), full


# -----------------------------------------------------------------------------
# FQE
# -----------------------------------------------------------------------------


class _FQEQNet(nn.Module):
    def __init__(self, in_features: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FQEResult:
    value: float
    q_init_mean: float
    n_iterations: int


def fitted_q_evaluation(
    ds: TrajectoryDataset,
    policy_proba: np.ndarray,
    gamma: float,
    n_iterations: int = 60,
    lr: float = 5e-4,
    batch_size: int = 1024,
    device: torch.device | None = None,
    seed: int = 42,
) -> FQEResult:
    """Per-policy fitted Q evaluation.

    Trains Q_pi(s, a) via TD(0) targets with the evaluation policy's action
    distribution at next state. Returns V^pi = E_s0[sum_a pi(a|s0) Q(s0, a)],
    evaluated over the stay starting states (t_bin == 0).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    in_features = ds.states.shape[1]
    n_actions = int(policy_proba.shape[1])
    q_net = _FQEQNet(in_features, n_actions).to(device)
    target_net = _FQEQNet(in_features, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-build tensors.
    s = torch.from_numpy(ds.states).to(device)
    ns = torch.from_numpy(ds.next_states).to(device)
    a = torch.from_numpy(ds.actions.astype(np.int64)).to(device)
    r = torch.from_numpy(ds.rewards.astype(np.float32)).to(device)
    d = torch.from_numpy(ds.dones.astype(np.float32)).to(device)
    pp_next = torch.from_numpy(
        _soften_policy(policy_proba).astype(np.float32)
    ).to(device)

    n = s.shape[0]
    rng = np.random.default_rng(seed)

    for it in range(n_iterations):
        idx = rng.permutation(n)
        # Multiple minibatches per iteration.
        for start in range(0, n, batch_size):
            chunk = idx[start : start + batch_size]
            bs = torch.as_tensor(chunk, dtype=torch.long, device=device)
            with torch.no_grad():
                q_next = target_net(ns.index_select(0, bs))
                pi_next = pp_next.index_select(0, bs)
                v_next = (q_next * pi_next).sum(dim=-1)
                target = r.index_select(0, bs) + (1.0 - d.index_select(0, bs)) * gamma * v_next
            q_now = q_net(s.index_select(0, bs))
            q_taken = q_now.gather(1, a.index_select(0, bs).unsqueeze(1)).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(q_taken, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            opt.step()
        # Periodic target sync.
        if (it + 1) % 5 == 0:
            target_net.load_state_dict(q_net.state_dict())

    # Evaluate V^pi at start states (t_bin == 0).
    start_mask = ds.t_bins == 0
    start_idx = np.where(start_mask)[0]
    q_net.eval()
    with torch.no_grad():
        q_start = q_net(s[torch.as_tensor(start_idx, dtype=torch.long, device=device)])
        pi_start = torch.from_numpy(
            _soften_policy(policy_proba[start_idx]).astype(np.float32)
        ).to(device)
        v = (q_start * pi_start).sum(dim=-1)
        value = float(v.mean().item())
        q_init_mean = float(q_start.mean().item())
    return FQEResult(value=value, q_init_mean=q_init_mean, n_iterations=n_iterations)


def bootstrap_fqe_ci(
    ds: TrajectoryDataset,
    policy_proba: np.ndarray,
    gamma: float,
    n_boot: int = 200,
    seed: int = 42,
    n_iterations: int = 30,
    batch_size: int = 1024,
) -> tuple[float, tuple[float, float]]:
    """Trajectory-bootstrap CI on FQE (fewer boots than WIS; critic is expensive).

    We refit the critic each bootstrap on a trajectory-resampled dataset. 200
    boots balances cost vs stability; reduce n_iterations (default 30) to keep
    runtime reasonable.
    """
    grouped = _group_by_stay(ds)
    stay_ids = list(grouped.keys())
    rng = np.random.default_rng(seed)

    full = fitted_q_evaluation(
        ds, policy_proba, gamma=gamma, n_iterations=n_iterations, batch_size=batch_size,
        seed=seed,
    )

    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_keys = [stay_ids[i] for i in rng.integers(0, len(stay_ids), size=len(stay_ids))]
        rows = np.concatenate([grouped[k] for k in sample_keys])
        boot_ds = TrajectoryDataset(
            states=ds.states[rows],
            actions=ds.actions[rows],
            rewards=ds.rewards[rows],
            next_states=ds.next_states[rows],
            next_actions=ds.next_actions[rows],
            dones=ds.dones[rows],
            stay_ids=ds.stay_ids[rows],
            t_bins=ds.t_bins[rows],
            feature_names=ds.feature_names,
            fold=ds.fold + "_boot",
        )
        boot_policy = policy_proba[rows]
        # Shorter FQE per boot.
        res = fitted_q_evaluation(
            boot_ds, boot_policy, gamma=gamma,
            n_iterations=max(5, n_iterations // 3),
            batch_size=batch_size, seed=seed + b + 1,
        )
        vals[b] = res.value

    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return full.value, (lo, hi)


# -----------------------------------------------------------------------------
# Null / reference policies
# -----------------------------------------------------------------------------


def null_policy_proba(
    ds: TrajectoryDataset, kind: str, n_actions: int = N_ACTIONS
) -> np.ndarray:
    """Return a fixed-behaviour policy probability matrix.

    kind:
        - "zero"           : always action 0 (no vasopressor)
        - "uniform"        : 1/A everywhere
        - "const_low"      : always action 1 (NE 0 < d <= 8.4 mcg/min)
        - "const_mid"      : always action 2 (NE 8.4-20.28)
        - "const_high"     : always action 3
    """
    n = ds.n_transitions
    if kind == "uniform":
        return np.full((n, n_actions), 1.0 / n_actions, dtype=np.float32)
    mapping = {"zero": 0, "const_low": 1, "const_mid": 2, "const_high": 3, "const_max": 4}
    if kind not in mapping:
        raise ValueError(f"Unknown null policy kind={kind!r}")
    a = mapping[kind]
    out = np.full((n, n_actions), DEFAULT_POLICY_FLOOR, dtype=np.float32)
    out[:, a] = 1.0 - (n_actions - 1) * DEFAULT_POLICY_FLOOR
    return out


# -----------------------------------------------------------------------------
# Clinician match + OOD
# -----------------------------------------------------------------------------


def clinician_match_rate(
    policy_actions: np.ndarray, clinician_actions: np.ndarray
) -> float:
    return float((policy_actions == clinician_actions).mean())


def ood_rate(
    policy_actions: np.ndarray,
    behaviour_proba: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Share of policy actions whose behaviour probability is below threshold."""
    idx = np.arange(len(policy_actions))
    behav_at_a = behaviour_proba[idx, policy_actions]
    return float((behav_at_a < threshold).mean())


# -----------------------------------------------------------------------------
# Bundle runner
# -----------------------------------------------------------------------------


@dataclass
class OPEReport:
    policy_name: str
    wis: float
    wis_ci_lo: float
    wis_ci_hi: float
    ess: float
    ess_frac: float
    fqe: float
    fqe_ci_lo: float
    fqe_ci_hi: float
    match_rate: float
    ood_rate: float
    n_trajectories: int
    n_transitions: int
    warnings: list[str]


def evaluate_policy(
    ds: TrajectoryDataset,
    policy_proba: np.ndarray,
    behaviour_proba: np.ndarray,
    clinician_actions: np.ndarray,
    gamma: float,
    policy_name: str,
    n_boot_wis: int = 1000,
    n_boot_fqe: int = 100,
    fqe_iterations: int = 40,
    seed: int = 42,
) -> OPEReport:
    """Run the full OPE bundle and collect warnings."""
    warnings: list[str] = []

    wis_val, (wis_lo, wis_hi), wis_full = bootstrap_wis_ci(
        ds, policy_proba, behaviour_proba, gamma=gamma, n_boot=n_boot_wis, seed=seed
    )
    if wis_full.ess_frac < 0.05:
        warnings.append(
            f"ESS fraction {wis_full.ess_frac:.3f} < 5% — WIS estimate unreliable"
        )

    fqe_val, (fqe_lo, fqe_hi) = bootstrap_fqe_ci(
        ds, policy_proba, gamma=gamma, n_boot=n_boot_fqe, n_iterations=fqe_iterations,
        seed=seed,
    )

    policy_actions = np.argmax(policy_proba, axis=1).astype(np.int64)
    match = clinician_match_rate(policy_actions, clinician_actions)
    ood = ood_rate(policy_actions, behaviour_proba)

    if ood > 0.10:
        warnings.append(
            f"OOD rate {ood:.3f} > 10% — tighten dBCQ threshold or CQL alpha"
        )

    return OPEReport(
        policy_name=policy_name,
        wis=wis_val,
        wis_ci_lo=wis_lo,
        wis_ci_hi=wis_hi,
        ess=wis_full.ess,
        ess_frac=wis_full.ess_frac,
        fqe=fqe_val,
        fqe_ci_lo=fqe_lo,
        fqe_ci_hi=fqe_hi,
        match_rate=match,
        ood_rate=ood,
        n_trajectories=wis_full.n_trajectories,
        n_transitions=ds.n_transitions,
        warnings=warnings,
    )


def format_report_table(reports: list[OPEReport]) -> str:
    """Format a markdown-ish table for ope_report.md."""
    hdr = (
        "| Policy | WIS (95% CI) | FQE (95% CI) | ESS% | Match% | OOD% | N traj |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [hdr]
    for r in reports:
        rows.append(
            "| {name} | {wis:.3f} [{wl:.3f},{wh:.3f}] | {fqe:.3f} [{fl:.3f},{fh:.3f}] "
            "| {ess:.1f} | {mr:.1f} | {ood:.1f} | {n} |".format(
                name=r.policy_name,
                wis=r.wis, wl=r.wis_ci_lo, wh=r.wis_ci_hi,
                fqe=r.fqe, fl=r.fqe_ci_lo, fh=r.fqe_ci_hi,
                ess=100 * r.ess_frac,
                mr=100 * r.match_rate,
                ood=100 * r.ood_rate,
                n=r.n_trajectories,
            )
        )
    return "\n".join(rows)

"""Offline RL algorithm implementations.

Every algorithm exposes:
    - ``update(batch)`` -> dict (must include 'loss')
    - ``policy_action(states)`` -> (N,) action ids
    - ``policy_proba(states)`` -> (N, A) probabilities
    - ``q_values(states)`` -> (N, A) Q-values  (Q-based agents; BC returns logits-as-Q)
    - ``state_snapshot()`` / ``load_snapshot(state)`` — checkpointing
    - ``to(device)``, ``train()``, ``eval()`` — nn.Module pass-through
"""
from __future__ import annotations

from src.rl.algorithms.bc import BehaviorCloning
from src.rl.algorithms.dqn import DuelingDDQN
from src.rl.algorithms.dbcq import DiscreteBCQ
from src.rl.algorithms.cql import ConservativeQLearning

__all__ = [
    "BehaviorCloning",
    "DuelingDDQN",
    "DiscreteBCQ",
    "ConservativeQLearning",
]


def build_agent(algo: str, **kwargs):  # type: ignore[no-untyped-def]
    algo = algo.lower()
    if algo == "bc":
        return BehaviorCloning(**kwargs)
    if algo == "ddqn":
        return DuelingDDQN(**kwargs)
    if algo == "dbcq":
        return DiscreteBCQ(**kwargs)
    if algo == "cql":
        return ConservativeQLearning(**kwargs)
    raise ValueError(f"Unknown algo={algo!r}")

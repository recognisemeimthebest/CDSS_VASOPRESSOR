"""Offline reinforcement learning package for vasopressor recommendation.

Modules
-------
* :mod:`src.rl.dataset` — trajectory building + reward shaping.
* :mod:`src.rl.networks` — Dueling Q-networks + behaviour-policy classifier.
* :mod:`src.rl.algorithms` — Behaviour Cloning / DDQN / dBCQ / CQL agents.
* :mod:`src.rl.training` — shared training loop (GPU, early stop on FQE).
* :mod:`src.rl.ope` — off-policy evaluation (WIS, FQE, ESS, bootstrap CI).
* :mod:`src.rl.run_rl` — CLI entry point.

Honesty guard (ch05/ch07):
    This package trains **offline** policies and reports **OPE** estimates.
    OPE numbers are NOT evidence of clinical benefit. Prospective validation
    is required before any real-world use.
"""
from __future__ import annotations

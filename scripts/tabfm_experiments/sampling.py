"""Row sampling for poisoning trials and the independent seed streams."""

from __future__ import annotations

import math

import numpy as np


def k_from_percent(n: int, row_percent: float) -> int:
    """``max(1, round(n * row_percent / 100))`` with half-up rounding (not banker's)."""
    if n < 1:
        raise ValueError("context must have at least one row")
    return int(min(n, max(1, math.floor(n * row_percent / 100.0 + 0.5))))


def eligible_indices(y: np.ndarray, attack_class: int | None) -> np.ndarray:
    y = np.asarray(y)
    if attack_class is None:
        return np.arange(len(y))
    if attack_class not in (0, 1):
        raise ValueError(f"attack_class must be 0, 1 or None, got {attack_class}")
    return np.flatnonzero(y.astype(int) == attack_class)


def sample_row_indices(
    y: np.ndarray, k: int, *, attack_class: int | None, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``k`` distinct context rows (sorted). Uses all eligible rows if fewer than ``k``."""
    elig = eligible_indices(y, attack_class)
    if len(elig) == 0:
        raise ValueError(f"no rows with y == {attack_class} in the context")
    if len(elig) <= k:
        return np.sort(elig)
    return np.sort(rng.choice(elig, size=k, replace=False))


def row_rng(row_seed: int, subsample_id: int) -> np.random.Generator:
    """Row-subset stream for subsample ``s``: ``Generator(row_seed + s)``."""
    return np.random.default_rng(row_seed + subsample_id)


def attack_seed_for(attack_seed: int, run_id: int) -> int:
    """CAPGD random-start stream for run ``r``."""
    return attack_seed + run_id


def random_flip_rng(row_seed: int, run_id: int, subsample_id: int = 0) -> np.random.Generator:
    """Row stream for the test-agnostic random label flip: a fresh draw per (run, subsample).

    Unlike ``row_rng`` (which x-capgd shares across runs so that runs differ only in CAPGD
    randomness), every run here must flip a *different* random subset.
    """
    return np.random.default_rng([row_seed, run_id, subsample_id])

#!/usr/bin/env python
"""Correctness tests for scripts/beam_filter.py (no GPU, no TabPFN), with pass functions whose
answers are known:

  A. parsing   specs parse; bad specs raise
  B. branches  m = 1: the X lowest; m > 1: child 0 is the m lowest, the others are m distinct
               rows, and a tiny tau makes them the greedy batch too
  C. greedy    K1:X1 reproduces iterative_filter.grow with the same fixed step, for a scorer that
               depends on T (distance to the nearest trusted point)
  D. search    a trap row (lowest score, but accepting it ruins the criterion) is taken by the
               greedy and avoided by a beam of width 2; children with equal trusted sets are
               merged (fewer passes than K x X per step)
  E. path      with static scores the path criterion's best beam is the greedy one

    python scripts/test_beam_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from beam_filter import beam_search, branches, parse_beam  # noqa: E402
from iterative_filter import grow, parse_variant  # noqa: E402


def test_parsing() -> None:
    b = parse_beam("beam:K4:X3:m50:hold")
    assert (b.K, b.X, b.m, b.criterion, b.prior, b.tau, b.uses_holdout) == (4, 3, 50, "hold", False, 0.1, True)
    b = parse_beam("beam:K2:X2:m1:path:prior:tau0.3")
    assert (b.K, b.X, b.m, b.criterion, b.prior, b.tau, b.uses_holdout) == (2, 2, 1, "path", True, 0.3, False)
    b = parse_beam("beam:K4:X4:p10:vcv")
    assert (b.m, b.pct, b.criterion, b.step(4950), b.step(407)) == (0, 10.0, "vcv", 495, 41) and not b.uses_holdout
    for bad in ("beam:K4:X3:m50", "beam:K0:X3:m5:hold", "grow:K4:X3:m5:hold", "beam:K4:X3:m5:hold:nope", "beam:K4:X3:hold"):
        try:
            parse_beam(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    print("A. parsing ok")


def test_branches() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(size=40)
    order = np.argsort(s, kind="stable")
    br = branches(s, 1, 3, rng, 0.1)
    assert [int(b[0]) for b in br] == order[:3].tolist()
    br = branches(s, 5, 4, rng, 0.1)
    assert np.array_equal(br[0], order[:5]) and all(len(set(b.tolist())) == 5 for b in br)
    br = branches(s, 5, 4, rng, 1e-6)  # tiny tau: the Gumbel noise cannot reorder distinct scores
    assert all(set(b.tolist()) == set(order[:5].tolist()) for b in br)
    print("B. branches ok")


def test_greedy() -> None:
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(60, 2)) * np.array([4.0, 1.0])
    n, k = 60, 15

    def dist(acc, q):
        trusted = np.concatenate([np.zeros((1, 2)), pts[acc]])
        return np.linalg.norm(pts[q][:, None, :] - trusted[None, :, :], axis=2).min(1)

    def pass_fn(acc):
        P = np.setdiff1d(np.arange(n), acc)
        return P, dist(acc, P), {"hold": 0.0, "holdent": 0.0, "entP": 0.0}

    for m in (1, 4):
        keep_b, step_b, info = beam_search(pass_fn, n, k, parse_beam(f"beam:K1:X1:m{m}:entP"), rng)
        keep_g, step_g, _, steps = grow(dist, n, k, 1, parse_variant(f"grow:m{m}"))
        assert np.array_equal(keep_b, keep_g) and np.array_equal(step_b, step_g), m
        assert info["steps"] == steps and info["passes"] == steps + 1
    print("C. greedy ok")


def test_search() -> None:
    n, k = 10, 3
    static = np.arange(n) * 0.1  # row 0 is the most plausible ...
    calls = []

    def pass_fn(acc):  # ... but accepting it ruins the criterion
        calls.append(len(acc))
        P = np.setdiff1d(np.arange(n), acc)
        return P, static[P], {"hold": 10.0 if 0 in acc else 0.01 * len(acc), "holdent": 0.0, "entP": 0.0}

    rng = np.random.default_rng(2)
    keep_g, _, _ = beam_search(pass_fn, n, k, parse_beam("beam:K1:X1:m1:hold"), rng)
    assert keep_g[0]  # the greedy takes the trap
    calls.clear()
    keep_b, _, info = beam_search(pass_fn, n, k, parse_beam("beam:K2:X2:m1:hold"), rng)
    assert not keep_b[0] and keep_b.sum() == n - k  # the beam avoids it
    # step 1: 2 children from one beam; later steps: at most 4, fewer when two parents meet
    assert info["passes"] == len(calls) and info["passes"] < 1 + 2 + 4 * (info["steps"] - 1)
    print("D. search ok")


def test_path() -> None:
    rng = np.random.default_rng(3)
    n, k = 30, 6
    static = rng.normal(size=n)

    def pass_fn(acc):
        P = np.setdiff1d(np.arange(n), acc)
        return P, static[P], {"hold": 0.0, "holdent": 0.0, "entP": 0.0}

    keep_b, _, _ = beam_search(pass_fn, n, k, parse_beam("beam:K3:X3:m1:path"), rng)
    want = np.ones(n, dtype=bool)
    want[np.argsort(-static, kind="stable")[:k]] = False
    assert np.array_equal(keep_b, want)
    print("E. path ok")


if __name__ == "__main__":
    test_parsing()
    test_branches()
    test_greedy()
    test_search()
    test_path()
    print("all ok")

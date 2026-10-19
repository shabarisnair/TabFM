#!/usr/bin/env python
"""Correctness tests for scripts/iterative_filter.py (no GPU, no TabPFN).

Checks against properties that must hold exactly, with scorers whose answer is known:

  A. parsing      variant specs parse to the right schedule, score, options, refine rounds;
                  bad specs raise
  B. quotas       sum to m, never exceed a class's rows, follow the proportions
  C. growth       every schedule removes exactly k rows; with a scorer that ignores T
                  (static scores) every schedule removes the k highest scores, i.e. the
                  V-filter; m = 1 takes n - k steps and accepts in score order; with a scorer
                  that depends on T (distance to the nearest trusted point on a line) the
                  one-row greedy grows outward from V and differs from the one-shot filter in
                  the predicted way; balanced growth keeps every step's per-class quotas
  D. refine       with static scores, one round removes the k highest scores and the next
                  round repeats it, so it stops after two; fold scoring never lets a row's own
                  fold into its context

    python scripts/test_iterative_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from iterative_filter import Variant, blend, grow, parse_variant, quotas, refine  # noqa: E402


def test_parsing() -> None:
    assert parse_variant("oneshot") == Variant("oneshot", "oneshot", 0.0)
    v = parse_variant("grow:pct1")
    assert (v.schedule, v.value, v.score, v.prior, v.balance, v.refine) == ("pct", 1.0, "ce", False, False, 0)
    v = parse_variant("grow:frac0.1:margin:bal:prior+refine3")
    assert (v.schedule, v.value, v.score, v.prior, v.balance, v.refine) == ("frac", 0.1, "margin", True, True, 3)
    v = parse_variant("grow:m1")
    assert (v.schedule, v.value) == ("m", 1.0)
    assert parse_variant("oneshot+refine2").refine == 2
    v = parse_variant("grow:pct2:anchor")
    assert v.anchor and not v.prior and v.score == "ce"
    assert parse_variant("grow:pct1:conf0.9").conf == 0.9 and parse_variant("grow:pct1").conf == 0.0
    v = parse_variant("grow:m1:cprior")
    assert v.cprior and not v.prior
    assert parse_variant("grow:pct1:prior:hold0.25").hold == 0.25 and parse_variant("grow:pct1").hold == -1.0
    assert parse_variant("oneshot:hold0").hold == 0.0 and parse_variant("oneshot").hold == -1.0
    v = parse_variant("grow:pct1:prior:holdr0.25")
    assert (v.hold, v.hold_mode, v.prior) == (0.25, "remove", True) and parse_variant("oneshot:hold0.5").hold_mode == "add"
    assert (parse_variant("grow:pct1:prior:holdp0.25").hold_mode, parse_variant("grow:pct1:prior:holdp0.25").hold) == ("penalty", 0.25)
    for bad in ("grow:pct1:conf1.5", "grow:pct1:conf0.9:margin", "grow:pct1:prior:cprior", "grow:pct1:hold1.5",
                "oneshot:prior"):
        try:
            parse_variant(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    for bad in ("grow", "grow:x1", "grow:pct0", "grow:m5:nope", "shrink:m1"):
        try:
            parse_variant(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse")
    print("A. parsing ok")


def test_blend() -> None:
    rng = np.random.default_rng(4)
    score, harm = rng.normal(size=60), rng.normal(size=60)
    assert np.array_equal(np.argsort(blend(score, harm, 0.0)), np.argsort(score))  # w = 0: the score's order
    assert np.array_equal(np.argsort(blend(score, harm, 1.0)), np.argsort(harm))  # w = 1: the harm's order
    b = blend(score, np.zeros(60), 0.5)  # a constant harm adds the same rank to every row
    assert np.array_equal(np.argsort(b), np.argsort(score))
    print("B0. blend ok")


def test_quotas() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        counts = rng.integers(0, 40, size=rng.integers(2, 6))
        if counts.sum() == 0:
            continue
        m = int(rng.integers(1, counts.sum() + 1))
        q = quotas(m, counts)
        assert q.sum() == m and (q <= counts).all() and (q >= 0).all(), (m, counts, q)
    assert quotas(10, np.array([50, 50])).tolist() == [5, 5]
    assert quotas(10, np.array([75, 25])).tolist() == [8, 2]  # 7.5 / 2.5: tie to the first class
    assert quotas(6, np.array([1, 100])).tolist() == [0, 6]  # a 0.06 share rounds to 0
    assert quotas(5, np.array([1, 9])).tolist() == [1, 4]  # 0.5 / 4.5: tie to the first class
    print("B. quotas ok")


def test_growth() -> None:
    rng = np.random.default_rng(1)
    n, k = 200, 40
    static = rng.normal(size=n)
    calls = []

    def static_fn(acc, q):
        calls.append((acc.copy(), q.copy()))
        assert np.intersect1d(acc, q).size == 0 and len(acc) + len(q) == n
        return static[q]

    want = np.ones(n, dtype=bool)
    want[np.argsort(-static, kind="stable")[:k]] = False
    for spec in ("oneshot", "grow:m1", "grow:m7", "grow:pct3", "grow:frac0.25", "grow:frac2"):
        calls.clear()
        keep, step_of, score_at, steps = grow(static_fn, n, k, 10, parse_variant(spec))
        assert (~keep).sum() == k, spec
        assert np.array_equal(keep, want), spec
        assert ((step_of == -1) == ~keep).all() and np.isnan(score_at[~keep]).all(), spec
        assert steps == len(calls), spec
    assert grow(static_fn, n, k, 10, parse_variant("oneshot"))[3] == 1
    logged = []
    _, _, _, steps = grow(static_fn, n, k, 10, parse_variant("grow:m7"), log_fn=lambda st, T: logged.append((st, int(T.sum()))))
    assert [st for st, _ in logged] == list(range(1, steps + 1)) and logged[-1][1] == n - k
    assert all(t == min(7 * st, n - k) for st, t in logged)
    keep, step_of, _, steps = grow(static_fn, n, k, 10, parse_variant("grow:m1"))
    assert steps == n - k
    order = np.argsort(step_of[keep])  # accepted in increasing score order
    assert (np.diff(static[np.flatnonzero(keep)][order]) >= 0).all()
    # conf: rows with score <= -log(t) join m at a time; then the rest of the n - k at once
    t = 0.6
    n_ok = int((static <= -np.log(t)).sum())
    keep, step_of, _, steps = grow(static_fn, n, k, 10, parse_variant(f"grow:m7:conf{t}"))
    assert np.array_equal(keep, want)  # static scores: same rows as the one-shot filter
    sizes = np.bincount(step_of[keep])
    assert steps == int(np.ceil(min(n_ok, n - k) / 7)) + (n_ok < n - k), (steps, n_ok)
    assert n_ok < n - k and sizes[-1] == n - k - n_ok  # the confident rows first, then the rest at once
    assert (static[step_of == steps - 1] > -np.log(t)).all() and (static[(step_of >= 0) & (step_of < steps - 1)] <= -np.log(t)).all()
    # shrink / both: m = ceil(f * rows still to accept) / ceil(f * min(|T|, rows still to accept))
    for spec, f in (("grow:shrink0.2", lambda T, r: int(np.ceil(0.2 * r))), ("grow:both0.3", lambda T, r: int(np.ceil(0.3 * min(T, r))))):
        keep, step_of, _, steps = grow(static_fn, n, k, 10, parse_variant(spec))
        assert np.array_equal(keep, want), spec
        sizes, T, r = np.bincount(step_of[keep], minlength=steps), 10, n - k
        for m in sizes:
            assert m == max(1, min(f(T, r), r)), (spec, m, T, r)
            T, r = T + m, r - m
        assert sizes[-1] == 1  # the end-game goes one row at a time
    # frac schedule: m = ceil(f * |T|) with |T| = n_v + accepted
    keep, step_of, _, steps = grow(static_fn, n, k, 10, parse_variant("grow:frac0.5"))
    sizes = np.bincount(step_of[keep])
    T = 10
    for m in sizes[:-1]:
        assert m == int(np.ceil(0.5 * T)), (m, T)
        T += m

    # a scorer that depends on T: score = distance to the nearest trusted point (V at the
    # origin, plus the accepted rows). A chain of 96 points runs from (1, 0) to (20, 0), 0.2
    # apart; a cluster of 24 points sits at (0, 8). The one-shot filter trusts only V, so it
    # removes the 24 chain points farthest from the origin (all beyond 8); the greedy walks
    # along the chain (each next point is 0.2 from a trusted one) and removes the cluster.
    pts = np.concatenate([np.stack([np.linspace(1, 20, 96), np.zeros(96)], 1),
                          np.stack([rng.normal(0, 0.05, 24), 8 + rng.normal(0, 0.05, 24)], 1)])

    def dist_fn(acc, q):
        trusted = np.concatenate([np.zeros((1, 2)), pts[acc]])
        return np.linalg.norm(pts[q][:, None, :] - trusted[None, :, :], axis=2).min(1)

    keep_os = grow(dist_fn, 120, 24, 1, parse_variant("oneshot"))[0]
    assert np.array_equal(np.flatnonzero(~keep_os), np.arange(72, 96))
    for spec in ("grow:m1", "grow:m4"):
        keep_g = grow(dist_fn, 120, 24, 1, parse_variant(spec))[0]
        assert np.array_equal(np.flatnonzero(~keep_g), np.arange(96, 120)), spec

    # balanced growth: every step's accepted rows follow the per-class quotas of the rows left
    y = rng.integers(0, 3, size=n)
    keepb, step_of_b, _, steps_b = grow(static_fn, n, k, 10, parse_variant("grow:pct10:bal"), y_given=y)
    assert (~keepb).sum() == k
    left = np.ones(n, dtype=bool)
    for st in range(steps_b):
        acc = step_of_b == st
        m = int(acc.sum())
        q = quotas(m, np.bincount(y[left], minlength=3))
        assert np.array_equal(np.bincount(y[acc], minlength=3), q), st
        for c in range(3):  # within a class, the lowest scores of the rows left go first
            cand = np.flatnonzero(left & (y == c))
            assert set(np.flatnonzero(acc & (y == c))) == set(cand[np.argsort(static[cand], kind="stable")[:q[c]]])
        left &= ~acc
    print("C. growth ok")


def test_refine() -> None:
    rng = np.random.default_rng(2)
    n, k = 120, 24
    static = rng.normal(size=n)
    seen = []

    def fn(acc, q):
        seen.append((acc.copy(), q.copy()))
        return static[q]

    start = np.ones(n, dtype=bool)
    start[rng.permutation(n)[:k]] = False
    keep, rounds = refine(fn, start, k, rounds=5, folds=5, seed=3)
    want = np.ones(n, dtype=bool)
    want[np.argsort(-static, kind="stable")[:k]] = False
    assert np.array_equal(keep, want) and rounds == 2, rounds
    fold = np.random.default_rng(3).permutation(n) % 5
    for acc, q in seen:
        f = fold[q[0]]
        assert (fold[q] == f).all() and (fold[acc] != f).all()  # a fold never sees itself
    keep, rounds = refine(fn, want, k, rounds=5, folds=5, seed=3)
    assert rounds == 1 and np.array_equal(keep, want)
    print("D. refine ok")


if __name__ == "__main__":
    test_parsing()
    test_blend()
    test_quotas()
    test_growth()
    test_refine()
    print("all ok")

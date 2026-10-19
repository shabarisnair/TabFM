#!/usr/bin/env python
"""Correctness tests for scripts/small_v_methods.py (CPU, no TabPFN).

  A. knn_shapley   equals the brute-force Shapley value (all subsets) of the K-NN likelihood utility
                   on a tiny problem, averaged over the clean rows
  B. label_spread  the closed form equals the fixed point of the iteration F <- alpha S F + (1 - alpha) Y0
  C. knn_sanitize  relabels a row to its neighbours' majority exactly when the share reaches eta
  D. sanity        on two separated Gaussian blobs with 20% of the labels flipped at random and three
                   clean rows per class, every method's k most suspicious rows are mostly the flipped
                   ones (precision >= 0.7; chance is 0.2), relabelling methods recover most labels, and
                   the training methods' classifiers are accurate on fresh rows

    python scripts/test_small_v_methods.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import small_v_methods as M  # noqa: E402


def _utility(S, D_row, yp, y_test, K):
    if not S:
        return 0.0
    near = sorted(S, key=lambda i: (D_row[i], i))[:K]
    return sum(yp[i] == y_test for i in near) / K


def test_knn_shapley() -> None:
    rng = np.random.default_rng(0)
    n, m, K = 7, 3, 2
    Xp, Xv = torch.tensor(rng.normal(size=(n, 2))), torch.tensor(rng.normal(size=(m, 2)))
    yp, yv = rng.integers(0, 2, n), rng.integers(0, 2, m)
    d = M.Draw(Xp, yp, Xv, yv, K=2, k=2, seed=0)
    got = -M.knn_shapley(d, K_nn=K).score
    D = torch.cdist(Xv, Xp).numpy()
    want = np.zeros(n)
    for j in range(m):
        for i in range(n):
            others = [t for t in range(n) if t != i]
            for size in range(n):
                for S in itertools.combinations(others, size):
                    w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    want[i] += w * (_utility(list(S) + [i], D[j], yp, yv[j], K) - _utility(list(S), D[j], yp, yv[j], K)) / m
    assert np.allclose(got, want, atol=1e-10), (got, want)
    assert np.isclose(got.sum(), np.mean([_utility(list(range(n)), D[j], yp, yv[j], K) for j in range(m)]))  # efficiency


def test_label_spread() -> None:
    rng = np.random.default_rng(1)
    Xp, Xv = torch.tensor(rng.normal(size=(30, 3))), torch.tensor(rng.normal(size=(4, 3)))
    yp, yv = rng.integers(0, 2, 30), np.array([0, 1, 0, 1])
    d = M.Draw(Xp, yp, Xv, yv, K=2, k=6, seed=0)
    for alpha, beta in ((0.5, 0.0), (0.9, 0.3)):
        res = M.label_spread(d, n_neighbors=5, alpha=alpha, beta=beta)
        Z = torch.cat([Xp, Xv]).double()          # the iteration, built the same way
        idx, dist = M._knn(Z, Z, 5, exclude_self=True)
        W = torch.zeros(34, 34, dtype=torch.float64).scatter_(1, idx, torch.exp(-(dist / dist.median()) ** 2))
        W = torch.maximum(W, W.T)
        S = W / W.sum(1).sqrt()[:, None] / W.sum(1).sqrt()[None, :]
        Y0 = torch.cat([beta * M._onehot(yp, 2, "cpu"), M._onehot(yv, 2, "cpu")]).double()
        Fm = Y0.clone()
        for _ in range(3000):
            Fm = alpha * S @ Fm + (1 - alpha) * Y0
        P = Fm[:30] / Fm[:30].sum(1, keepdim=True)
        assert np.allclose(res.score, 1 - P.numpy()[np.arange(30), yp], atol=1e-8)


def test_knn_sanitize() -> None:
    # rows on a line; 0..4 labelled 0 except row 2 (labelled 1); 5..9 labelled 1; V = one row per class far away
    Xp = torch.tensor([[float(i)] for i in range(10)])
    yp = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    Xv, yv = torch.tensor([[-50.0], [60.0]]), np.array([0, 1])
    d = M.Draw(Xp, yp, Xv, yv, K=2, k=1, seed=0)
    res = M.knn_sanitize(d, k_nn=4, eta=0.75)
    # row 2: neighbours 1, 3, 0, 4 all labelled 0 -> 0; row 4: neighbours 3, 5, 2, 6 labelled 0, 1, 1, 1 -> 1
    # (share 0.75); rows 3 and 5 see a 2-2 tie and keep their labels
    assert res.relabel[2] == 0 and res.relabel[4] == 1
    assert np.array_equal(np.delete(res.relabel, [2, 4]), np.delete(yp, [2, 4]))
    assert np.argmax(res.score) == 2              # after relabelling, 3 of row 2's 4 neighbours disagree with it
    res = M.knn_sanitize(d, k_nn=4, eta=1.0)      # only unanimous neighbourhoods relabel
    assert res.relabel[2] == 0 and np.array_equal(np.delete(res.relabel, 2), np.delete(yp, 2))


def _blobs(seed: int, n: int = 400, flip: float = 0.2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 5)) + 3.0 * y[:, None]
    yv = np.array([0, 0, 0, 1, 1, 1])
    Xv = rng.normal(size=(6, 5)) + 3.0 * yv[:, None]
    pois = rng.permutation(n)[:int(flip * n)]
    yp = y.copy()
    yp[pois] = 1 - yp[pois]
    Xt_y = rng.integers(0, 2, 500)
    Xt = rng.normal(size=(500, 5)) + 3.0 * Xt_y[:, None]
    mu, sd = np.concatenate([X, Xv]).mean(0), np.concatenate([X, Xv]).std(0)
    z = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731
    return z(X), yp, z(Xv), yv, y, pois, z(Xt), Xt_y


def test_sanity() -> None:
    Xp, yp, Xv, yv, y, pois, Xt, yt = _blobs(0)
    k = len(pois)

    def knn_probs(tr, te):  # stands in for TabPFN in cleanlab's out-of-fold step
        D = torch.cdist(Xp[te], torch.cat([Xp[tr], Xv]))
        lab = np.concatenate([yp[tr], yv])[torch.topk(D, 15, largest=False).indices.numpy()]
        p1 = (lab == 1).mean(1)
        return np.stack([1 - p1, p1], 1)

    d = M.Draw(Xp, yp, Xv, yv, K=2, k=k, seed=0, tabpfn_probs=knn_probs)
    is_pois = np.zeros(len(yp), bool)
    is_pois[pois] = True
    fast = {"glc": dict(epochs=20), "mwnet": dict(epochs=20), "emlc": dict(epochs=20), "ieg": dict(epochs=20),
            "data_oob": dict(n_trees=100, n_jobs=4)}
    for name, fn in M.METHODS.items():
        res = fn(d, **fast.get(name, {}))
        prec = is_pois[np.argsort(-res.score, kind="stable")[:k]].mean()
        assert prec >= 0.7, (name, prec)
        if res.relabel is not None:
            assert (res.relabel == y).mean() >= 0.85, (name, (res.relabel == y).mean())
        if res.predict is not None:
            assert (res.predict(Xt).argmax(1) == yt).mean() >= 0.9, name
        print(f"  {name}: precision {prec:.2f}" + (f", relabel acc {(res.relabel == y).mean():.2f}" if res.relabel is not None else ""))


if __name__ == "__main__":
    for t in (test_knn_shapley, test_label_spread, test_knn_sanitize, test_sanity):
        t()
        print(f"{t.__name__}: ok")

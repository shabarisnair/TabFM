import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")

from tabfm_experiments.attacks import run_label_flip_ga  # noqa: E402
from tabfm_experiments.ga import GAConfig  # noqa: E402
from tabfm_experiments.metrics import binary_metrics  # noqa: E402
from tabfm_experiments.runtime import batched_label_probs, build_tabpfn_v2, evaluate_context  # noqa: E402


def _toy(seed=0, n=40, d=3, n_test=12):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = (X[:, 0] > 0).float()
    Xt = torch.randn(n_test, d, generator=g)
    return X, y, Xt, (Xt[:, 0] > 0).long()


def test_batched_label_probs_match_evaluate_context():
    clf = build_tabpfn_v2("cpu", 0)
    X, y, Xt, yt = _toy()
    ys = []
    for idx in ([], [0, 5, 7], list(range(10, 25))):
        yb = y.clone()
        yb[idx] = 1 - yb[idx]
        ys.append(yb)
    P = batched_label_probs(clf, X, torch.stack(ys, 1), Xt)
    assert P.shape == (12, 3, 2)
    for b, yb in enumerate(ys):
        _, _, ref = evaluate_context(clf, X, yb, Xt, yt, need_grad=False)
        assert float((P[:, b] - ref).abs().max()) < 1e-4


def test_batched_probs_independent_of_batch_composition():
    # Justifies the GA cache: a mask's fitness must not depend on which other masks share its batch.
    clf = build_tabpfn_v2("cpu", 0)
    X, y, Xt, _ = _toy(seed=3)
    m = y.clone()
    m[[1, 4, 9]] = 1 - m[[1, 4, 9]]
    alone = batched_label_probs(clf, X, m[:, None], Xt)[:, 0]
    others = [y.clone() for _ in range(5)]
    batch = torch.stack(others[:2] + [m] + others[2:], 1)
    ingroup = batched_label_probs(clf, X, batch, Xt)[:, 2]
    # Only float-kernel jitter from the batched matmul (~5e-6), not a real dependence.
    assert float((alone - ingroup).abs().max()) < 1e-4


def test_recompute_layers_same_loss_and_grad():
    X, y, Xt, yt = _toy(seed=1)
    base = build_tabpfn_v2("cpu", 0)
    rc = build_tabpfn_v2("cpu", 0, recompute_layers=True)
    l0, g0, p0 = evaluate_context(base, X, y, Xt, yt, need_grad=True)
    l1, g1, p1 = evaluate_context(rc, X, y, Xt, yt, need_grad=True)
    assert float(l0) == pytest.approx(float(l1), abs=1e-6)
    assert torch.allclose(g0, g1, atol=1e-5)
    _, _, q1 = evaluate_context(rc, X, y, Xt, yt, need_grad=False)
    assert torch.allclose(p0, q1, atol=1e-5)


def test_label_flip_ga_trial_cpu():
    clf = build_tabpfn_v2("cpu", 0)
    X, y, Xt, yt = _toy(seed=2)
    _, _, p = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    clean = binary_metrics(p.numpy(), yt.numpy())
    pool = np.arange(len(y))
    k = 8
    cfg = GAConfig(population=6, generations=3, batch_size=4)
    res = run_label_flip_ga(clf, X, y, Xt, yt, pool, k, cfg=cfg, run_id=0, subsample_id=0, ga_seed=5,
                            clean_metrics=clean)
    flipped = res.extra["flipped_indices"]
    assert set(flipped) <= set(pool) and 1 <= len(flipped) <= k  # budget respected
    assert res.extra["batched_check_abs_dce"] < 1e-4
    # elitism + influence seeding: the winner is at least as good (fitness rows) as influence top-k
    infl_ce = res.extra["influence_topk"]["poisoned"]["ce"]
    assert res.poisoned.ce >= infl_ce - 1e-4
    changed = np.flatnonzero(res.y_poisoned != y.numpy().astype(int))
    assert np.array_equal(np.sort(changed), np.sort(flipped))

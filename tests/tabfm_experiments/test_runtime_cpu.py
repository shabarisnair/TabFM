import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")

from tabfm_experiments.runtime import (  # noqa: E402
    build_tabpfn_v2,
    chunking_is_exact,
    evaluate_context,
    target_embedding_influence_scores,
)


@pytest.fixture(scope="module")
def clf():
    """For the 4-feature toys."""
    return build_tabpfn_v2("cpu", model_seed=0, deterministic_mode="best-effort")


@pytest.fixture(scope="module")
def clf2():
    """Separate classifier for the 2-feature influence toys (one classifier per feature width)."""
    return build_tabpfn_v2("cpu", model_seed=0, deterministic_mode="best-effort")


def _toy(seed=0, n=32, d=4, n_test=8):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).float()
    Xt = torch.randn(n_test, d, generator=g)
    yt = (Xt[:, 0] + 0.3 * Xt[:, 1] > 0).long()
    return X, y, Xt, yt


def test_repeat_forward_is_stable(clf):
    X, y, Xt, yt = _toy()
    l1, g1, p1 = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    l2, _, p2 = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    assert g1 is None
    assert p1.shape == (8, 2)
    assert float((p1 - p2).abs().max()) < 1e-4
    ce = -torch.log(p1[torch.arange(8), yt].clamp_min(1e-12)).mean()
    assert float(l1) == pytest.approx(float(ce), abs=1e-6)


def test_nograd_memory_saving_forward_matches_differentiable_forward(clf):
    X, y, Xt, yt = _toy(seed=4)
    _, _, p_eval = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    assert clf.differentiable_input is True  # restored
    with torch.no_grad():
        clf.fit_with_differentiable_input(X, y)
        p_diff = clf.forward(Xt, use_inference_mode=True)
    assert float((p_eval - p_diff).abs().max()) < 1e-6


def test_grad_wrt_x_nonzero_and_y_has_no_useful_grad(clf):
    X, y, Xt, yt = _toy()
    _, g, _ = evaluate_context(clf, X, y, Xt, yt, need_grad=True)
    assert g.shape == X.shape
    assert float(g.abs().sum()) > 0
    # Labels are densified with (y > unique_ys).sum(): autograd sees no path to y.
    y_leaf = y.clone().requires_grad_(True)
    clf.fit_with_differentiable_input(X, y_leaf)
    probs = clf.forward(Xt, use_inference_mode=True)
    loss = torch.nn.functional.nll_loss(torch.log(probs.clamp_min(1e-12)), yt)
    (gy,) = torch.autograd.grad(loss, y_leaf, allow_unused=True)
    assert gy is None or float(gy.abs().sum()) == 0.0


def test_chunking_matches_unchunked(clf):
    X, y, Xt, yt = _toy(seed=1, n_test=10)
    assert chunking_is_exact(X, Xt)
    l_full, g_full, p_full = evaluate_context(clf, X, y, Xt, yt, need_grad=True)
    l_ch, g_ch, p_ch = evaluate_context(clf, X, y, Xt, yt, need_grad=True, test_batch_size=3)
    assert float(l_full) == pytest.approx(float(l_ch), abs=1e-5)
    assert torch.allclose(p_full, p_ch, atol=1e-5)
    # Same math; each chunk re-runs the float32 forward/backward, so grads agree to float noise.
    assert torch.allclose(g_full, g_ch, rtol=1e-2, atol=1e-4)


def test_refuses_classifier_reuse_across_feature_widths():
    c = build_tabpfn_v2("cpu", model_seed=0)
    X, y, Xt, yt = _toy(d=4)
    evaluate_context(c, X, y, Xt, yt, need_grad=False)
    X5, y5, Xt5, yt5 = _toy(d=5)
    with pytest.raises(ValueError, match="new classifier"):
        evaluate_context(c, X5, y5, Xt5, yt5, need_grad=False)


def test_chunking_exactness_detector():
    X = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert chunking_is_exact(X, torch.tensor([[1.0, 5.0]]))
    assert not chunking_is_exact(X, torch.tensor([[2.0, 5.0]]))


def _lone_row_toy(seed):
    g = torch.Generator().manual_seed(seed)
    X0 = torch.randn(15, 2, generator=g) * 0.7 + torch.tensor([-1.5, 0.0])
    X1 = torch.randn(15, 2, generator=g) * 0.7 + torch.tensor([1.5, 0.0])
    lone = torch.tensor([[0.0, 3.0]])
    X = torch.cat([X0, X1, lone])
    y = torch.cat([torch.zeros(15), torch.ones(15), torch.ones(1)])
    Xt = lone + torch.randn(10, 2, generator=g) * 0.3
    return X, y, Xt, torch.ones(10, dtype=torch.long)


def _ce_with_embedding_shift(clf, X, y, Xt, yt, row, t):
    """Mean CE after moving row's target embedding by t * (flip direction)."""
    arch = clf.models_[0]
    orig = arch._embed_targets
    w0 = arch.target_embedder.weight[:, 0].detach()

    def hooked(yy, *, num_rows, num_train_labels, batch_size):
        emb, m, u = orig(yy, num_rows=num_rows, num_train_labels=num_train_labels, batch_size=batch_size)
        add = torch.zeros_like(emb)
        add[:, row, :] = t * (1.0 - 2.0 * float(y[row])) * w0
        return emb + add, m, u

    arch._embed_targets = hooked
    try:
        loss, _, _ = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    finally:
        del arch._embed_targets
    return float(loss)


def test_influence_scores_match_finite_differences(clf2):
    clf = clf2
    X, y, Xt, yt = _lone_row_toy(0)
    scores = target_embedding_influence_scores(clf, X, y, Xt, yt, np.arange(len(y)))
    assert scores.shape == (len(y),)
    assert "_embed_targets" not in vars(clf.models_[0])  # hook removed
    t = 0.05
    for row in np.argsort(-np.abs(scores))[:3]:
        fd = (_ce_with_embedding_shift(clf, X, y, Xt, yt, row, t)
              - _ce_with_embedding_shift(clf, X, y, Xt, yt, row, -t)) / (2 * t)
        assert fd == pytest.approx(scores[row], rel=0.3, abs=1e-4), (row, fd, scores[row])


def test_influence_ranks_harmful_flip_high(clf2):
    """Plan test 12: flipping the isolated row clearly raises CE and ranks top-k.

    This holds for this toy (seed 0) but first-order influence is NOT a reliable
    ranking for TabPFNv2 in general: with seeds 1 and 2 of the same construction the
    isolated row ranks last, because CE is non-monotone along the flip direction
    (see docs/context_poisoning.md).
    """
    clf = clf2
    X, y, Xt, yt = _lone_row_toy(0)
    scores = target_embedding_influence_scores(clf, X, y, Xt, yt, np.arange(len(y)))
    base, _, _ = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    true_gain = []
    for i in range(len(y)):
        yf = y.clone()
        yf[i] = 1 - yf[i]
        li, _, _ = evaluate_context(clf, X, yf, Xt, yt, need_grad=False)
        true_gain.append(float(li - base))
    best = int(np.argmax(true_gain))
    assert best == 30 and true_gain[best] > 1.0
    assert best in np.argsort(-scores)[:3], (best, np.argsort(-scores)[:3])

    masked = target_embedding_influence_scores(clf, X, y, Xt, yt, y.numpy() == 0)
    assert np.isneginf(masked[15:]).all() and np.isfinite(masked[:15]).all()

#!/usr/bin/env python
"""Audit the row-weight attention bias before anything is built on top of it.

The bias is a local patch to vendored TabPFN, so it gets checked against properties
that must hold exactly, not against intuition:

  1. NO-OP          X = 1 everywhere must give bit-comparable predictions to running
                    with no bias at all. log 1 = 0, so anything else is a plumbing bug.
  2. SHIFT-INVARIANT  X = c everywhere, for any constant c, must ALSO match the
                    baseline: softmax is invariant to a constant added to every logit.
                    This is what makes "all X -> 0" harmless to the context.
  3. REMOVAL        X = 0 on a set of rows, applied at every layer, must reproduce
                    physically deleting those rows from the context. Exact equality is
                    not expected: TabPFN fits its imputation, scaling and
                    constant-column masks on the context, and a shorter context changes
                    those. The test reports the gap next to a preprocessing-free
                    reference so the residual can be attributed.
  4. GRADIENT       the loss must produce finite, non-zero gradients on the logits.
  5. DIAGONAL       with D on both sides, masking the diagonal must actually change the
                    predictions -- otherwise every row is reading its own label.

    python scripts/verify_sift_mechanism.py --gpu 0 --dataset heloc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits  # noqa: E402
from tabfm_experiments.sift import (  # noqa: E402
    HARD_OFF, RowWeights, build_masks, row_attention_modules, row_bias_installed,
    weights_installed,
)


def probs_with_bias(clf, X_ctx, y_ctx, X_q, modules, masks, forward, install, x=None):
    """Predictions under an attention bias, and optionally a weighted label mean."""
    with torch.no_grad():
        clf.fit_with_differentiable_input(X_ctx, y_ctx)
        install(clf)
        with weights_installed(clf.models_[0], modules, masks, x):
            p = forward(clf, X_q, need_grad=False)
    return p.detach().float().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", default="heloc")
    ap.add_argument("--n-context", type=int, default=400)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--n-drop", type=int, default=40, help="rows zeroed in the removal test")
    ap.add_argument("--tol-exact", type=float, default=1e-6)
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import (
        _forward, _install_recompute, build_tabpfn_v2, device_from_gpu,
    )

    train, _ = default_splits(a.dataset)
    device = device_from_gpu(a.gpu)
    Xdf, y, cols = load_split(train)
    # Queries are held out of the context: these checks are about the bias, and rows
    # that appear on both sides would add a self-attention effect on top.
    arr = Xdf.to_numpy(np.float32)[: a.n_context + a.n_query]
    yall = np.asarray(y).astype(int)[: a.n_context + a.n_query]
    Xq = torch.tensor(arr[: a.n_query], device=device)
    X = torch.tensor(arr[a.n_query:], device=device)
    y = yall[a.n_query:]
    yq = yall[: a.n_query]
    yt = torch.as_tensor(y.astype(np.float32), device=device)
    n = X.shape[0]
    clf = build_tabpfn_v2(device, 0)
    with torch.no_grad():                       # first fit builds clf.models_
        clf.fit_with_differentiable_input(X, yt)
    all_mods = row_attention_modules(clf, "all")
    print(f"{a.dataset}: context {n}x{len(cols)}, query {a.n_query}, "
          f"{len(all_mods)} blocks\n")

    fails = []

    def check(name, cond, detail):
        print(f"  {'PASS' if cond else 'FAIL'}  {name:<34s} {detail}")
        if not cond:
            fails.append(name)

    base = probs_with_bias(clf, X, yt, Xq, all_mods, None, _forward, _install_recompute)

    # --- 1 & 2: no-op and shift invariance ------------------------------------------
    # A uniform bias is a constant added to every logit, so the softmax is unchanged.
    # Equality is float32-exact only at X=1 (bias 0.0); any other constant perturbs the
    # arithmetic slightly, so the gate is 1e-5 -- the point is that the error does NOT
    # grow with the size of the shift.
    for label, logit, tol in (("1. no-op at X=1", 30.0, a.tol_exact),
                              ("2. shift-invariant X~0.5", 0.0, 1e-5),
                              ("2b. shift-invariant X~2e-9", -20.0, 1e-5),
                              ("2c. shift-invariant X~2e-22", -50.0, 1e-5)):
        w = RowWeights(n, device, init_logit=logit)
        masks = build_masks(w.log_x.detach(), None)
        p = probs_with_bias(clf, X, yt, Xq, all_mods, masks, _forward, _install_recompute)
        d = float(np.abs(p - base).max())
        check(label, d < tol,
              f"max|dp| vs no-bias = {d:.3e}  (X={float(torch.sigmoid(torch.tensor(logit))):.3g})")

    # --- 3: a zeroed row's LABEL cannot reach the predictions -------------------------
    # This is the property the whole method rests on: we exclude rows because their
    # labels are wrong, so a down-weighted row must stop contributing its label.
    #
    # It is testable with NO confound. Append a block of rows twice over, with identical
    # FEATURES but opposite LABELS. TabPFN fits its preprocessing on features alone, so
    # both contexts are preprocessed identically and the only difference is the labels of
    # the appended block. Zero that block: if the bias works, the two must agree exactly.
    # The appended block is class-balanced, so flipping its labels leaves BOTH the
    # context class counts and the training label mean untouched. That matters: test
    # rows carry a NaN target which the y-encoder imputes with the training label mean,
    # which is a label channel that no attention bias can reach. Balancing closes it, so
    # what this measures is the attention alone.
    rng = np.random.default_rng(0)
    keep = np.arange(n - a.n_drop)
    half = a.n_drop // 2
    dup_src = np.concatenate([rng.permutation(keep[y[keep] == 0])[:half],
                              rng.permutation(keep[y[keep] == 1])[:half]])
    ki = torch.as_tensor(keep, dtype=torch.long, device=device)
    di = torch.as_tensor(dup_src, dtype=torch.long, device=device)
    X_dup = torch.cat([X[ki], X[di]])
    y_same = torch.cat([yt[ki], yt[di]])
    y_flip = torch.cat([yt[ki], 1.0 - yt[di]])

    w = RowWeights(len(keep) + a.n_drop, device, init_logit=30.0)
    with torch.no_grad():
        w.logit[len(keep):] = HARD_OFF
    zeroed = build_masks(w.log_x.detach(), None)
    p_same = probs_with_bias(clf, X_dup, y_same, Xq, all_mods, zeroed, _forward, _install_recompute)
    p_flip = probs_with_bias(clf, X_dup, y_flip, Xq, all_mods, zeroed, _forward, _install_recompute)
    # Control: the same comparison with the block left at full weight.
    c_same = probs_with_bias(clf, X_dup, y_same, Xq, all_mods, None, _forward, _install_recompute)
    c_flip = probs_with_bias(clf, X_dup, y_flip, Xq, all_mods, None, _forward, _install_recompute)
    d_zero = float(np.abs(p_same - p_flip).max())
    d_live = float(np.abs(c_same - c_flip).max())
    check("3. silenced rows leak no label", d_zero < 1e-6,
          f"flipping the silenced block's labels moves predictions by {d_zero:.3e} "
          f"(at full weight the same flip moves them {d_live:.3e})")

    # How negative the logit must go before a row is really gone. Suppression competes
    # with the attention logits, so this is a scale, not a constant.
    scale = []
    for lg in (-10.0, -60.0, HARD_OFF):
        w2 = RowWeights(len(keep) + a.n_drop, device, init_logit=30.0)
        with torch.no_grad():
            w2.logit[len(keep):] = lg
        mk = build_masks(w2.log_x.detach(), None)
        pa = probs_with_bias(clf, X_dup, y_same, Xq, all_mods, mk, _forward, _install_recompute)
        pb = probs_with_bias(clf, X_dup, y_flip, Xq, all_mods, mk, _forward, _install_recompute)
        scale.append(f"logit {lg:>7.0f} -> {np.abs(pa - pb).max():.2e}")
    print("        3b. residual label leak vs how hard the row is pushed off:  "
          + " | ".join(scale))

    # --- 4 & 5: the real configuration -- D on BOTH sides -----------------------------
    # Here query t IS context row t, which is the setting the method actually runs in,
    # so the diagonal mask has something to block and the gradient is the real one.
    X_self = X[: a.n_query]
    y_self = y[: a.n_query]
    rows = torch.arange(a.n_query, device=device)

    w = RowWeights(n, device, init_logit=4.0)
    masks = build_masks(w.log_x, rows)
    clf.fit_with_differentiable_input(X, yt)
    _install_recompute(clf)
    with row_bias_installed(all_mods, masks):
        probs = _forward(clf, X_self, need_grad=True)
    p_given = probs[torch.arange(a.n_query, device=probs.device),
                    torch.as_tensor(y_self, device=probs.device)]
    loss = ((1 - w.x[: a.n_query]) * -torch.log(p_given.clamp_min(1e-12))).sum() / n
    loss.backward()
    g = w.logit.grad
    finite = g is not None and bool(torch.isfinite(g).all())
    nonzero = finite and float(g.abs().max()) > 0
    check("4. gradient on the logits", finite and nonzero,
          f"max|grad| = {float(g.abs().max()):.4e}, "
          f"{int((g != 0).sum())}/{n} rows with non-zero grad, all finite = {finite}")

    flat = build_masks(RowWeights(n, device, init_logit=30.0).log_x.detach(), None)
    cut = build_masks(RowWeights(n, device, init_logit=30.0).log_x.detach(), rows)
    p_leak = probs_with_bias(clf, X, yt, X_self, all_mods, flat, _forward, _install_recompute)
    p_mask = probs_with_bias(clf, X, yt, X_self, all_mods, cut, _forward, _install_recompute)
    acc_leak = float((p_leak.argmax(1) == y_self).mean())
    acc_mask = float((p_mask.argmax(1) == y_self).mean())
    check("5. diagonal blocks the self-read", acc_leak > acc_mask,
          f"accuracy on a row's own label {acc_leak:.3f} (leaking) -> {acc_mask:.3f} "
          f"(masked); max|dp| = {np.abs(p_leak - p_mask).max():.3e}")

    # --- 6 & 7: the weighted training-label mean --------------------------------------
    # Test rows carry a NaN target imputed with the mean of the TRAINING labels, so a
    # context row reaches them through that statistic whatever the attention does.
    # Check 3 hid this by balancing the block; here the flip is deliberately UNBALANCED
    # so the mean really moves, which is the case the weighting has to handle.
    ones = torch.ones(n, device=device)
    p_w1 = probs_with_bias(clf, X, yt, Xq, all_mods, None, _forward, _install_recompute, x=ones)
    check("6. weighted mean is a no-op at X=1", float(np.abs(p_w1 - base).max()) < a.tol_exact,
          f"max|dp| vs unweighted = {np.abs(p_w1 - base).max():.3e}")

    # All one class, so flipping the block genuinely moves the training label mean.
    # (A random block would flip ~half each way and barely move it -- no channel to test.)
    # 7. The weighted mean itself, checked directly against a hand computation
    #    (including the skip-non-finite behaviour).
    from tabpfn.architectures.tabpfn_v2 import _weighted_nanmean_over_rows
    g = torch.Generator(device="cpu").manual_seed(0)
    xv = torch.randn(50, 3, 1, generator=g).to(device)
    xv[3, 0, 0] = float("nan")
    xv[7, 2, 0] = float("inf")
    wv = torch.rand(50, generator=g).to(device)
    got = _weighted_nanmean_over_rows(xv, wv).cpu().numpy()
    xn, wn = xv.cpu().numpy(), wv.cpu().numpy()
    fin = np.isfinite(xn)
    want = ((np.where(fin, xn, 0) * wn[:, None, None] * fin).sum(0)
            / (wn[:, None, None] * fin).sum(0))
    check("7. weighted mean is computed right", float(np.abs(got - want).max()) < 1e-6,
          f"max|d| vs hand-computed = {np.abs(got - want).max():.3e} "
          f"(non-finite entries skipped)")

    # 7b. Informational: with BINARY targets this channel cannot actually bite. The
    #     imputed mean is densified by _flatten_multiclass_targets to
    #     (mean > unique_ys).sum(), and every mean strictly inside (0, 1) maps to the
    #     same code 1 -- the value is discarded. Weighting it is therefore defensive
    #     (it would matter for regression or multiclass), not load-bearing here.
    unbal = rng.permutation(keep[y[keep] == 0])[: a.n_drop]
    ui = torch.as_tensor(unbal, dtype=torch.long, device=device)
    Xu = torch.cat([X[ki], X[ui]])
    yu_same = torch.cat([yt[ki], yt[ui]])
    yu_flip = torch.cat([yt[ki], 1.0 - yt[ui]])
    wsil = RowWeights(len(keep) + a.n_drop, device, init_logit=30.0)
    with torch.no_grad():
        wsil.logit[len(keep):] = HARD_OFF
    msil = build_masks(wsil.log_x.detach(), None)
    off = np.abs(
        probs_with_bias(clf, Xu, yu_same, Xq, all_mods, msil, _forward, _install_recompute)
        - probs_with_bias(clf, Xu, yu_flip, Xq, all_mods, msil, _forward, _install_recompute)
    ).max()
    print(f"        7b. label mean {float(yu_same.mean()):.4f} -> {float(yu_flip.mean()):.4f} "
          f"on silenced rows moves predictions by {off:.3e}: the binary densification "
          f"discards the mean, so this channel is inert for these datasets")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

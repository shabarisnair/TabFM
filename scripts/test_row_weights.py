#!/usr/bin/env python
"""Correctness of the context-row weighting (scripts/tabfm_experiments/row_weights.py) on real TabPFNv2.

Per dataset (binary credit-approval, 3-class students_dropout, 6-class gas-drift: a random
context and query set from the context file), with one classifier reused across fits as in
every experiment, "deleted" meaning the stock model fitted on the context without the rows:

  A. w = 1 everywhere   == the stock model with no weights installed
  B. w = 0 on rows S    == S deleted, for: a random 25% of the rows; a single row; the rows
                           whose deletion leaves a feature constant (the discrete masks
                           change); every row of one class (the class list changes; 3+ classes)
  C1. statistics, w = eps  the label mean and z-norms weighted by eps (the attention given
                           the exact 0) -> S deleted as eps -> 0
  C2. all, log w = L    -> S deleted as L -> -inf, for every case of B. The attention converges
                           only once L is far below minus the logit gaps of E2 (hundreds in
                           blocks 7-9): w = 1e-12 is not yet near deletion, L = -1000 is
  D. channels           which channels w = 0 needs to equal deletion: attention only;
                           + label mean (the older --gate attn); + the encoder statistics;
                           + the external preprocessing (everything)
  E. coverage           in one forward the bias reaches all 12 blocks for both query kinds,
                           test-row queries attend through one key/value head, and the
                           between-rows attention runs over every column, the label column
                           included
  E2. logit gaps        (reported) per block, how far a weighted-out key's attention logit
                           can exceed the log-sum-exp of the kept keys' logits
  F. gradient           d CE / d log w is finite (also with exact zeros) and matches central
                           finite differences; activation checkpointing gives the same gradient;
                           chunked queries give the same loss and gradient

    python scripts/test_row_weights.py [--device cpu] [--datasets credit-approval ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

TOL = 2e-5          # max |probability difference| accepted as "the same"


def setup(ds: str, device: str, n_ctx: int, n_q: int, seed: int = 0, recompute: bool = False):
    import gate_context as G
    from eval_vfilter_attacks import CONTEXT
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2
    X, y = G.load(Path(CONTEXT[ds]), device)
    K = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(y))
    c_rows, q_rows = np.sort(rows[:n_ctx]), np.sort(rows[n_ctx:n_ctx + n_q])
    idx = lambda r: torch.as_tensor(r, device=device)  # noqa: E731
    clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=recompute, n_classes=K)
    return clf, X[idx(c_rows)], y[c_rows], X[idx(q_rows)], y[q_rows], K


def stock_probs(clf, Xc, yc, Xq) -> np.ndarray:
    from tabfm_experiments.runtime import _forward
    with torch.no_grad():
        clf.fit_with_differentiable_input(Xc, torch.as_tensor(yc, dtype=torch.float32, device=Xc.device))
        return _forward(clf, Xq, need_grad=False).float().cpu().numpy()


def weighted(clf, Xc, yc, Xq, log_w) -> np.ndarray:
    from tabfm_experiments.row_weights import probs_weighted
    return probs_weighted(clf, Xc, yc, Xq, log_w)


def lw_with(n: int, rows, value: float, device) -> torch.Tensor:
    lw = torch.zeros(n, device=device)
    lw[torch.as_tensor(np.asarray(rows), device=device)] = value
    return lw


def deleted(clf, Xc, yc, Xq, rows) -> np.ndarray:
    keep = np.setdiff1d(np.arange(len(yc)), rows)
    return stock_probs(clf, Xc[torch.as_tensor(keep, device=Xc.device)], yc[keep], Xq)


def constant_making_rows(Xc: torch.Tensor):
    """(feature, rows): the smallest set of rows whose deletion leaves some non-constant
    feature constant -- every row off the feature's most common value -- if one exists
    with at most a third of the rows."""
    X = Xc.cpu().numpy()
    best = None
    for f in range(X.shape[1]):
        vals, counts = np.unique(X[:, f], return_counts=True)
        if len(vals) < 2:
            continue
        rows = np.flatnonzero(X[:, f] != vals[np.argmax(counts)])
        if len(rows) <= len(X) // 3 and (best is None or len(rows) < len(best[1])):
            best = (f, rows)
    return best


def check(name: str, diff: float, tol: float = TOL) -> None:
    ok = diff <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max |dp| = {diff:.2e}" + ("" if ok else f" > {tol:.0e}"), flush=True)
    if not ok:
        raise AssertionError(name)


def test_dataset(ds: str, device: str, n_ctx: int, n_q: int) -> None:
    print(f"\n=== {ds} (context {n_ctx}, queries {n_q}) ===", flush=True)
    clf, Xc, yc, Xq, yq, K = setup(ds, device, n_ctx, n_q)
    n = len(yc)
    full = stock_probs(clf, Xc, yc, Xq)      # first fit on the full context fixes the inferred schema
    # A; the float32 noise of the weighted path at w = 1 sets this dataset's tolerance
    noise = float(np.abs(weighted(clf, Xc, yc, Xq, torch.zeros(n, device=device)) - full).max())
    check("A  w = 1 everywhere == stock", noise)
    tol = max(TOL, 3 * noise)
    print(f"       tolerance for B-D: {tol:.1e} (3 x the float noise of A, at least {TOL:.0e})")
    # B
    rng = np.random.default_rng(1)
    S_rand = np.sort(rng.choice(n, n // 4, replace=False))
    cases = [("random 25%", S_rand), ("one row", np.array([int(rng.integers(n))]))]
    cm = constant_making_rows(Xc)
    if cm is not None:
        cases.append((f"feature {cm[0]} left constant ({len(cm[1])} rows)", cm[1]))
    else:
        print("  (no feature can be made constant by deleting <= 1/3 of the rows: case skipped)")
    if K >= 3:
        c_small = int(np.argmin([np.sum(yc == c) if np.any(yc == c) else n for c in range(K)]))
        cases.append((f"all of class {c_small} ({int(np.sum(yc == c_small))} rows)", np.flatnonzero(yc == c_small)))
    for name, S in cases:
        check(f"B  w = 0 on {name} == deleted", np.abs(weighted(clf, Xc, yc, Xq, lw_with(n, S, float("-inf"), device))
                                                     - deleted(clf, Xc, yc, Xq, S)).max(), tol)
    # C1: the statistics channels (label mean, z-norms) are continuous in w: eps there, the
    # attention given the exact 0, approaches the deletion as eps -> 0
    ref = deleted(clf, Xc, yc, Xq, S_rand)
    zero = lw_with(n, S_rand, float("-inf"), device)
    diffs = {eps: float(np.abs(partial(clf, Xc, yc, Xq, zero, lw_with(n, S_rand, float(np.log(eps)), device)) - ref).max())
             for eps in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12)}
    ok = diffs[1e-8] <= tol and diffs[1e-12] <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} C1 statistics channels at w = eps (attention 0) vs deleted, random 25%: "
          + "  ".join(f"{eps:.0e}: {d:.1e}" for eps, d in diffs.items()))
    if not ok:
        raise AssertionError("C1")
    # C2: every channel at log w = L approaches the deletion as L -> -inf. The attention needs
    # L far below minus the largest gap between a weighted-out key's logit and the kept keys'
    # (hundreds in blocks 7-9, see E2), so w = 1e-12 (L = -27.6) is not yet near deletion there
    for name, S in cases:
        ref = deleted(clf, Xc, yc, Xq, S)
        diffs = {L: float(np.abs(weighted(clf, Xc, yc, Xq, lw_with(n, S, L, device)) - ref).max())
                 for L in (-3.0, -10.0, -30.0, -100.0, -300.0, -1000.0, -3000.0)}
        ok = diffs[-1000.0] <= tol and diffs[-3000.0] <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} C2 log w = L on {name} vs deleted: "
              + "  ".join(f"{L:.0f}: {d:.1e}" for L, d in diffs.items()))
        if not ok:
            raise AssertionError(f"C2 {name}")
    # D
    test_channels(clf, Xc, yc, Xq, S_rand, tol)
    # E
    if ds == "credit-approval":
        test_coverage(clf, Xc, yc, Xq)
    logit_gaps(clf, Xc, yc, Xq, yq, S_rand)
    # F
    if ds == "credit-approval":
        test_gradient(ds, device, n_ctx, n_q, clf, Xc, yc, Xq, yq)


def partial(clf, Xc, yc, Xq, lw_attn, lw_stats, *, label_mean=True, encoder=True, external=True) -> np.ndarray:
    """Probabilities with the attention biased by lw_attn and the chosen statistics channels
    weighted by exp(lw_stats): the label mean, the encoder's statistics and masks, the
    external preprocessing (z-norm, constant features)."""
    from tabpfn.preprocessing.steps.context_row_weights import context_row_weights_installed
    from tabfm_experiments.row_weights import attention_bias
    from tabfm_experiments.runtime import _forward
    from tabfm_experiments.sift import row_attention_modules
    w = lw_stats.exp()
    yc_t = torch.as_tensor(yc, dtype=torch.float32, device=Xc.device)
    with torch.no_grad(), context_row_weights_installed(w if external else None):
        clf.fit_with_differentiable_input(Xc, yc_t)
        arch, mods = clf.models_[0], row_attention_modules(clf, "all")
        try:
            for m in mods:
                m.row_attn_bias = attention_bias(lw_attn)
            arch.row_weights = w if label_mean else None
            arch.row_weights_stats = w if encoder else None
            return _forward(clf, Xq, need_grad=False).float().cpu().numpy()
        finally:
            for m in mods:
                m.row_attn_bias = None
            arch.row_weights = arch.row_weights_stats = None


def test_channels(clf, Xc, yc, Xq, S, tol: float) -> None:
    """Deletion mismatch with the channels switched on one by one."""
    n = len(yc)
    lw = lw_with(n, S, float("-inf"), Xc.device)
    ref = deleted(clf, Xc, yc, Xq, S)

    def run(label_mean: bool, encoder: bool, external: bool) -> float:
        p = partial(clf, Xc, yc, Xq, lw, lw, label_mean=label_mean, encoder=encoder, external=external)
        return float(np.abs(p - ref).max())

    d = {"attention only": run(False, False, False), "+ label mean (older --gate attn)": run(True, False, False),
         "+ encoder statistics": run(True, True, False), "+ external preprocessing (all)": run(True, True, True)}
    print("  info D  w = 0 on the random 25% vs deleted, by channels: " + "; ".join(f"{k} {v:.1e}" for k, v in d.items()))
    if not (d["+ external preprocessing (all)"] <= tol and d["+ label mean (older --gate attn)"] > 1e-3):
        raise AssertionError("D")
    print("  ok   D  only the full set matches; attention + label mean alone does not")


def test_coverage(clf, Xc, yc, Xq) -> None:
    """Record every attention call of one forward."""
    import tabpfn.architectures.tabpfn_v2 as V2
    from tabfm_experiments.row_weights import evaluate_weighted
    calls = []
    orig = V2.scaled_dot_product_attention

    def spy(q, k, v, *args, attn_mask=None, **kw):
        calls.append((tuple(q.shape), tuple(k.shape), None if attn_mask is None else tuple(attn_mask.shape),
                      None if attn_mask is None else float(attn_mask.detach().min())))
        return orig(q, k, v, *args, attn_mask=attn_mask, **kw)

    V2.scaled_dot_product_attention = spy
    try:
        lw = torch.zeros(len(yc), device=Xc.device)
        lw[0] = -5.0                                        # a visible entry
        # the gradient forward: one unchunked attention call per block (the gradient-free
        # one may split a call over the columns to save memory)
        evaluate_weighted(clf, Xc, yc, Xq, np.zeros(len(Xq), dtype=int), lw.requires_grad_(True), need_grad=True)
    finally:
        V2.scaled_dot_product_attention = orig
    n, m = len(yc), len(Xq)
    feat = [c for c in calls if c[2] is None]               # between features (within a row): no bias
    between = [c for c in calls if c[2] is not None]
    train_q = [c for c in between if c[0][1] == n]          # queries = the context rows
    test_q = [c for c in between if c[0][1] == m]           # queries = the test rows
    n_cols = feat[0][0][1]                                  # columns = feature groups + the label column
    ok = (len(train_q) == 12 and len(test_q) == 12 and len(feat) == 12
          and all(c[1][2] == 6 for c in train_q) and all(c[1][2] == 1 for c in test_q)
          and all(c[2] == (1, 1, 1, n) and c[3] == -5.0 for c in between)
          and all(c[0][0] == n_cols for c in between))
    print(f"  {'ok  ' if ok else 'FAIL'} E  12 blocks x (context-row queries over 6 key/value heads, test-row queries over "
          f"head 0), bias over all {n} context keys, every one of the {n_cols} columns (label column included)")
    if not ok:
        raise AssertionError(f"E {calls[:6]}")


def logit_gaps(clf, Xc, yc, Xq, yq, S) -> None:
    """Report, per block, the largest gap between a weighted-out key's attention logit and the
    log-sum-exp of the kept keys' logits, over the queries that matter (kept context rows,
    test rows): a weight w removes a row only once log w is well below minus this gap."""
    import tabpfn.architectures.tabpfn_v2 as V2
    from tabfm_experiments.row_weights import evaluate_weighted
    n = len(yc)
    keep = torch.ones(n, dtype=torch.bool)
    keep[torch.as_tensor(S)] = False
    gaps = {"context": [], "test": []}
    orig = V2.scaled_dot_product_attention

    def spy(q, k, v, *args, attn_mask=None, **kw):
        if attn_mask is not None:
            qq, kk = q.detach().permute(0, 2, 1, 3).cpu(), k.detach().permute(0, 2, 1, 3).cpu()
            kk = kk.expand(-1, qq.shape[1], -1, -1) if kk.shape[1] == 1 else kk
            s = qq @ kk.transpose(-1, -2) / qq.shape[-1] ** 0.5
            g = s[..., ~keep].amax(-1) - torch.logsumexp(s[..., keep], -1)
            gaps["context" if q.shape[1] == n else "test"].append(float((g[..., keep] if q.shape[1] == n else g).max()))
        return orig(q, k, v, *args, attn_mask=attn_mask, **kw)

    V2.scaled_dot_product_attention = spy
    try:
        evaluate_weighted(clf, Xc, yc, Xq, yq, lw_with(n, S, -30.0, Xc.device).requires_grad_(True), need_grad=True)
    finally:
        V2.scaled_dot_product_attention = orig
    print("  info E2 largest logit gap (weighted-out key vs log-sum-exp of the kept keys) per block, at log w = -30: "
          + "; ".join(f"{k} queries " + " ".join(f"{x:.0f}" for x in v) for k, v in gaps.items()))


def test_gradient(ds, device, n_ctx, n_q, clf, Xc, yc, Xq, yq) -> None:
    from tabfm_experiments.row_weights import evaluate_weighted
    n = len(yc)
    lw0 = torch.full((n,), float(np.log(0.5)), device=device)
    lw = lw0.clone().requires_grad_(True)
    loss, g, _ = evaluate_weighted(clf, Xc, yc, Xq, yq, lw, need_grad=True)
    g = g.cpu().numpy()
    assert np.isfinite(g).all() and np.abs(g).max() > 0, "F gradient not finite or all zero"
    rows = np.argsort(-np.abs(g))[:3].tolist() + [int(np.argmin(np.abs(g)))]
    h = 0.05
    rel = []
    for r in rows:
        f = []
        for sgn in (1, -1):
            l2 = lw0.clone()
            l2[r] += sgn * h
            f.append(evaluate_weighted(clf, Xc, yc, Xq, yq, l2, need_grad=False)[0])
        num = (f[0] - f[1]) / (2 * h)
        rel.append(abs(num - g[r]) / max(abs(num), abs(g[r]), 1e-6))
        print(f"       row {r}: analytic {g[r]:+.3e}  finite difference {num:+.3e}")
    ok = max(rel[:3]) < 0.05
    print(f"  {'ok  ' if ok else 'FAIL'} F  gradient vs central differences (h = {h} in log w): max rel. error {max(rel[:3]):.1e} on the 3 largest")
    if not ok:
        raise AssertionError("F finite differences")
    # exact zeros: finite gradients everywhere
    lz = lw0.clone()
    lz[: n // 4] = float("-inf")
    lz.requires_grad_(True)
    _, gz, _ = evaluate_weighted(clf, Xc, yc, Xq, yq, lz, need_grad=True)
    ok = bool(torch.isfinite(gz).all()) and float(gz[: n // 4].abs().max()) == 0.0
    print(f"  {'ok  ' if ok else 'FAIL'} F  with w = 0 on a quarter of the rows: gradients finite, exactly 0 on the zeroed rows")
    if not ok:
        raise AssertionError("F zeros")
    # chunking and activation checkpointing
    l_c, g_c, _ = evaluate_weighted(clf, Xc, yc, Xq, yq, lw, need_grad=True, batch_size=max(1, len(yq) // 3))
    clf_r = setup(ds, device, n_ctx, n_q, recompute=True)[0]
    stock_probs(clf_r, Xc, yc, Xq)
    l_r, g_r, _ = evaluate_weighted(clf_r, Xc, yc, Xq, yq, lw, need_grad=True)
    d_c = max(abs(l_c - loss), float((g_c.cpu().numpy() - g).__abs__().max()))
    d_r = max(abs(l_r - loss), float((g_r.cpu().numpy() - g).__abs__().max()))
    ok = d_c < 1e-5 and d_r < 1e-5
    print(f"  {'ok  ' if ok else 'FAIL'} F  chunked queries: max diff {d_c:.1e}; activation checkpointing: max diff {d_r:.1e}")
    if not ok:
        raise AssertionError("F chunking / recompute")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--datasets", nargs="+", default=["credit-approval", "students_dropout", "gas-drift"])
    ap.add_argument("--n-context", type=int, default=160)
    ap.add_argument("--n-query", type=int, default=60)
    a = ap.parse_args(argv)
    torch.set_num_threads(16)
    t0 = time.time()
    for ds in a.datasets:
        test_dataset(ds, a.device, a.n_context, a.n_query)
    print(f"\nall checks passed ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

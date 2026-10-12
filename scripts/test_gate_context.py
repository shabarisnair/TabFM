#!/usr/bin/env python
"""Correctness tests for gate_context.py (feature-only context gating).

Every check is against an INDEPENDENT reference: a hand-computed value, a second
implementation, a stored result from an earlier experiment, finite differences, or a
property that must hold exactly. None compares the code under test with itself.

  A. update rules  pgd_step / adam_step / gate_stats / lowest_gated against
                   hand-computed values
  A2. validation   draw_validation: hand-computed class counts, nested draws for a fixed
      draw         seed, random within a class; on the real context its class counts
                   equal prepare_multiclass_splits.stratified_exact's (a second
                   implementation of the same quota rule)
  A3. holdout and  split_holdout: hand-computed class counts and rounding, disjoint,
      stopping     seeded; selection_rules on a hand-made log: the iteration each
      rules        stopping rule picks and the test scores it reports
  B. the gate      u_null equals nan_oracle_experiment's independent capture;
                   g = 1 and g = None are bit-identical to no hook (no-grad and
                   gradient paths);
                   the gated encoder input is exactly g*u + (1-g)*u_null row by row;
                   query rows and the label encoder input are untouched;
                   encoder rows follow context rows (swap test);
                   g = 0 on the poisoned rows reproduces the oracle experiment's stored
                   feat-gate probabilities, and g = 1 its poisoned ones;
                   the hook refuses a mismatched layout
  C. gradient      central finite differences (small steps: the loss is strongly curved
                   in g) along two directions and on the five largest coordinates;
                   chunked == unchunked gradient
  D. end-to-end    main() with pgd and adam (W&B offline); check_run() then checks:
                   - every output exists;
                   - log.csv, the top-k files and the test metrics are consistent;
                   - test scores appear on exactly the test iterations;
                   - the poisoned rows match poison();
                   - iteration 0 matches runtime.evaluate_context on the validation set;
                   - the iteration-0 hard evaluation is the random removal;
                   - the best gates' hard evaluation matches runtime.evaluate_context
                     on independently chosen kept rows;
                   - re-scoring the saved best gates reproduces the saved probabilities;
                   - the validation-set-as-context and poisoned-plus-validation
                     references match runtime.evaluate_context on those contexts;
                   - the W&B summary was written.
                   Also, the references match stored results: ungated vs the
                   multiclass grid; oracle gate and oracle removal vs
                   nan_oracle_experiment.
  E. validation    --val-from-context: refused with --val, when the draw would miss a
     from the      class, and when it would take the whole context; a short run whose
     context       split is disjoint and covers the context file, whose poison is
                   poison() on the remaining rows, and which passes every check of D
                   rebuilt from the file and the saved split; the W&B run carries its
                   group and the draw's size and seed. --val-holdout: refused without test
                   iterations or with nothing left to fit; a short run whose 50 validation
                   rows split 40 fitted / 10 held out, whose held-out scores sit exactly on
                   the test iterations and match runtime.evaluate_context, whose best
                   iteration is the held-out accuracy rule's, and which passes every check
                   of D with the baselines on all 50 rows.

    python scripts/test_gate_context.py --gpu 1          (about 9 min)
    python scripts/test_gate_context.py --gpu 1 --sections draw valsplit
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name:<66s} {detail}")
    if not cond:
        FAILS.append(name)


# ======================================================================== A. update rules
def test_update_rules():
    print("\nA. update rules")
    # pinned: row 1 (g=1, wants up) and row 4 (g=0, wants down); free |grad| max = 0.2
    g = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.0])
    grad = torch.tensor([0.2, -0.4, 0.1, 0.05, 0.3])
    G.pgd_step(g, grad, 0.05, "linf")
    want = torch.tensor([0.95, 1.0, 0.975, 0.4875, 0.0])
    check("pgd linf: hand-computed step, pinned rows ignored by the scale",
          torch.allclose(g, want, atol=1e-7), f"{g.tolist()}")

    g = torch.tensor([1.0, 0.6, 0.02])
    G.pgd_step(g, torch.tensor([0.5, -0.1, 0.1]), 0.2, "none")
    check("pgd none: g - lr*grad, clipped to [0, 1]",
          torch.allclose(g, torch.tensor([0.9, 0.62, 0.0]), atol=1e-7), f"{g.tolist()}")

    g = torch.tensor([1.0, 0.0, 0.3])
    G.pgd_step(g, torch.tensor([-1.0, 2.0, 0.0]), 0.05, "linf")
    check("pgd linf: no movable gradient -> no change",
          torch.equal(g, torch.tensor([1.0, 0.0, 0.3])))

    g = torch.ones(4, requires_grad=True)
    opt = torch.optim.Adam([g], lr=0.05)
    G.adam_step(g, torch.tensor([1e-4, -1e-4, 3e-6, 0.0]), opt)
    # Adam's first step is lr * sign(grad) (up to eps); the clip holds row 1 at 1.
    check("adam: first step = lr*sign(grad), then clipped",
          torch.allclose(g.detach(), torch.tensor([0.95, 1.0, 0.95, 1.0]), atol=2e-4),
          f"{g.detach().tolist()}")
    for _ in range(100):
        G.adam_step(g, torch.tensor([1.0, -1.0, 1.0, 1.0]), opt)
    check("adam: gates stay in [0, 1]", bool(((g >= 0) & (g <= 1)).all()), f"{g.detach().tolist()}")

    # gate_stats: AUROC against a brute-force pair count
    rng = np.random.default_rng(0)
    gv = rng.uniform(size=200).round(1)  # with ties
    pois = rng.uniform(size=200) < 0.3
    removed = np.array([0, 5, 7, 9])
    s = G.gate_stats(gv, pois, removed)
    sp, sn = 1 - gv[pois], 1 - gv[~pois]
    auc = ((sp[:, None] > sn[None]).sum() + 0.5 * (sp[:, None] == sn[None]).sum()) / (len(sp) * len(sn))
    check("gate_stats: AUROC = brute-force pair count (with ties)",
          abs(s["auroc_poisoned"] - auc) < 1e-12, f"{s['auroc_poisoned']:.6f} vs {auc:.6f}")
    hits = sum(bool(pois[i]) for i in removed)
    check("gate_stats: means, share below 0.5, precision/recall of the removed rows",
          abs(s["g_mean_poisoned"] - gv[pois].mean()) < 1e-12
          and abs(s["frac_g_below_0.5"] - (gv < 0.5).mean()) < 1e-12
          and s["precision_at_R"] == hits / 4 and s["recall_at_R"] == hits / pois.sum(),
          f"precision {s['precision_at_R']:.3f} recall {s['recall_at_R']:.4f}")
    s1 = G.gate_stats(np.ones(50), np.arange(50) < 10, np.arange(3))
    check("gate_stats: all gates equal -> AUROC 0.5, AP = poisoned share",
          s1["auroc_poisoned"] == 0.5 and abs(s1["ap_poisoned"] - 0.2) < 1e-12)
    check("gate_stats: no poisoned rows -> no detection keys",
          "auroc_poisoned" not in G.gate_stats(np.ones(5), np.zeros(5, bool), np.arange(2)))

    # lowest_gated: the k smallest gates; ties go to the smaller tie key
    gv = np.array([0.9, 0.2, 1.0, 0.2, 0.5, 1.0], dtype=np.float32)
    tie = np.array([5, 4, 3, 2, 1, 0])
    check("lowest_gated: smallest gates, ties by tie order (hand-computed)",
          G.lowest_gated(gv, 3, tie).tolist() == [1, 3, 4]
          and G.lowest_gated(gv, 5, tie).tolist() == [0, 1, 3, 4, 5], f"{G.lowest_gated(gv, 5, tie).tolist()}")
    tie = np.random.default_rng(1).permutation(1000)
    check("lowest_gated: all gates equal -> the rows whose tie key is < k",
          np.array_equal(G.lowest_gated(np.ones(1000, np.float32), 50, tie), np.flatnonzero(tie < 50)))
    check("lowest_gated: k = 0 -> nothing removed", len(G.lowest_gated(gv, 0, np.arange(6))) == 0)


# ====================================================== A2. validation drawn from the context
def test_draw_validation():
    print("\nA2. validation drawn from the context")
    # 100 rows, classes 0/1/2 with shares 0.5/0.3/0.2, interleaved as in a real file
    y = np.repeat([0, 1, 2], [50, 30, 20])[np.random.default_rng(7).permutation(100)]
    # n = 10: shares 5/3/2 exactly.  n = 12: shares 6/3.6/2.4 -> floors 6/3/2, and the one
    # leftover row goes to the largest remainder, class 1 (0.6).
    for n, want in ((10, [5, 3, 2]), (12, [6, 4, 2])):
        v = G.draw_validation(y, n, seed=0)
        got = np.bincount(y[v], minlength=3).tolist()
        check(f"draw_validation n={n}: class counts {want} (hand-computed)", got == want, f"{got}")
        check(f"draw_validation n={n}: {n} distinct sorted row indices in range",
              len(v) == n and bool(np.all(np.diff(v) > 0)) and v.min() >= 0 and v.max() < 100)
    draws = [set(G.draw_validation(y, n, seed=0).tolist()) for n in (10, 20, 40, 80)]
    check("draw_validation: one seed gives nested draws (10 in 20 in 40 in 80)",
          all(a <= b for a, b in zip(draws, draws[1:])))
    check("draw_validation: another seed gives another draw",
          set(G.draw_validation(y, 20, seed=1).tolist()) != draws[1])
    first = {int(i) for c, q in ((0, 5), (1, 3), (2, 2)) for i in np.flatnonzero(y == c)[:q]}
    check("draw_validation: random rows within a class, not the first ones in the file",
          draws[0] != first)

    # the real gas-drift context: same class counts as stratified_exact, nested 100..500
    from prepare_multiclass_splits import stratified_exact
    from tabfm_experiments.data import load_split
    yc = np.asarray(load_split(G.find_split("gas-drift", "context_"))[1]).astype(int)
    same, sets = True, []
    for n in (100, 200, 300, 400, 500):
        v = G.draw_validation(yc, n, seed=0)
        ref = stratified_exact(np.arange(len(yc)), yc, n, np.random.default_rng(0))
        same &= np.array_equal(np.bincount(yc[v], minlength=6), np.bincount(yc[ref], minlength=6))
        sets.append(set(v.tolist()))
    check("draw_validation on gas-drift: class counts == stratified_exact's, n = 100..500", bool(same))
    check("draw_validation on gas-drift: nested, n = 100 in 200 in ... in 500",
          all(a <= b for a, b in zip(sets, sets[1:])))


# =================================== A3. holdout split of the validation set; stopping rules
def test_holdout_and_rules():
    print("\nA3. validation-test holdout and stopping rules")
    # 50 rows, 30/20: a 20% holdout is 10 rows, exactly 6/4; the other 40 rows are 24/16
    y = np.repeat([0, 1], [30, 20])[np.random.default_rng(3).permutation(50)]
    fit, hold = G.split_holdout(y, 0.2, seed=0)
    check("split_holdout 50 rows, 20%: holdout 6/4, fit 24/16 (hand-computed)",
          np.bincount(y[hold]).tolist() == [6, 4] and np.bincount(y[fit]).tolist() == [24, 16],
          f"hold {np.bincount(y[hold]).tolist()} fit {np.bincount(y[fit]).tolist()}")
    check("split_holdout: disjoint, together all 50 rows, sorted",
          len(np.intersect1d(fit, hold)) == 0 and np.array_equal(np.union1d(fit, hold), np.arange(50))
          and bool(np.all(np.diff(fit) > 0)) and bool(np.all(np.diff(hold) > 0)))
    # 53 rows, 32/21: 0.2 * 53 = 10.6 -> 11 rows; shares 6.64 / 4.36 -> floors 6/4, the
    # leftover row to the larger remainder (class 0) -> 7/4
    y53 = np.repeat([0, 1], [32, 21])[np.random.default_rng(4).permutation(53)]
    _, h53 = G.split_holdout(y53, 0.2, seed=0)
    check("split_holdout 53 rows: 11 held out, 7/4 (hand-computed rounding)",
          np.bincount(y53[h53]).tolist() == [7, 4], f"{np.bincount(y53[h53]).tolist()}")
    check("split_holdout: same seed -> same rows, another seed -> other rows",
          np.array_equal(G.split_holdout(y, 0.2, seed=0)[1], hold)
          and not np.array_equal(G.split_holdout(y, 0.2, seed=1)[1], hold))

    # stopping rules on a hand-made log: test (and validation-test) scored at 0, 2, 4, 6
    def r(it, va, vc, hold=None, test=None):
        row = {"iter": it, "val_accuracy": va, "val_ce": vc}
        if hold:
            row |= {"valtest_accuracy": hold[0], "valtest_ce": hold[1],
                    "test_soft_accuracy": test[0], "test_soft_ce": test[1],
                    "test_hard_accuracy": test[0] - 0.1, "test_hard_ce": test[1] + 0.1}
        return row
    log = [r(0, 0.80, 0.50, (0.70, 0.20), (0.75, 0.55)),
           r(1, 0.90, 0.40),
           r(2, 0.95, 0.30, (0.90, 0.35), (0.89, 0.40)),
           r(3, 1.00, 0.20),                       # validation accuracy first at its max
           r(4, 1.00, 0.10, (0.90, 0.30), (0.88, 0.35)),
           r(5, 1.00, 0.05),
           r(6, 1.00, 0.04, (0.80, 0.50), (0.84, 0.45))]
    got = {k: v["iteration"] for k, v in G.selection_rules(log, holdout=True).items()}
    # valtest_acc: 0.90 at 2 and 4, lower CE wins -> 4.  valtest_ce: lowest 0.20 -> 0.
    # val_acc (test iterations only): 1.00 at 4 and 6, lower CE -> 6.  val_first_max:
    # first max at 3 -> next scored iteration 4.  last: 6.  test_acc_oracle: 0.89 -> 2.
    want = {"valtest_acc": 4, "valtest_ce": 0, "val_acc": 6, "val_first_max": 4, "last": 6,
            "test_acc_oracle": 2}
    check("selection_rules: each rule's iteration (hand-computed)", got == want, f"{got}")
    rules = G.selection_rules(log, holdout=True)
    check("selection_rules: test scores of the chosen iteration, soft and hard",
          rules["valtest_acc"]["test_soft"] == {"accuracy": 0.88, "ce": 0.35}
          and abs(rules["valtest_acc"]["test_hard"]["accuracy"] - 0.78) < 1e-12,
          f"{rules['valtest_acc']}")
    check("selection_rules without a holdout: no valtest rules",
          set(G.selection_rules(log, holdout=False)) == {"val_acc", "val_first_max", "last", "test_acc_oracle"})


# ============================================================ A4. top-k gates: the pieces
def test_topk_pieces():
    print("\nA4. top-k gates: mask, straight-through gradients, score step, detection")
    s = torch.tensor([0.3, -1.0, 2.0, -1.0, 0.5])
    check("topk_mask: the 2 lowest scores masked; the tie at -1.0 keeps both (hand-computed)",
          G.topk_mask(s, 2).tolist() == [1, 0, 1, 0, 1], f"{G.topk_mask(s, 2).tolist()}")
    check("topk_mask: 3 lowest -> rows 1, 3, 0; 0 -> nothing masked",
          G.topk_mask(s, 3).tolist() == [0, 0, 1, 0, 1] and G.topk_mask(s, 0).tolist() == [1] * 5)
    check("topk_mask: all scores equal -> the first rows go (stable ties)",
          G.topk_mask(torch.zeros(4), 2).tolist() == [0, 0, 1, 1])

    grad = torch.tensor([1.0, 1.0, 2.0])
    S = torch.tensor([0.0, 2.0, -2.0])
    check("ste_grad identity: the gate gradient itself",
          torch.equal(G.ste_grad(grad, S, "identity", k_remove=1), grad))
    # sigmoid'(0) = 0.25; sigmoid'(+-2) = 0.8807971 * 0.1192029 = 0.1049936
    got = G.ste_grad(grad, S, "sigmoid", k_remove=1)
    check("ste_grad sigmoid: grad * sigmoid'(S) (hand-computed)",
          torch.allclose(got, torch.tensor([0.25, 0.1049936, 0.2099872]), atol=1e-6), f"{got.tolist()}")
    # centred on the boundary between the lowest score (-1) and the next (0): tau = -0.5,
    # width = std([-1, 0, 1, 2]) = 1.2909944; rows -1 and 0 sit 0.5 either side of it:
    # sigmoid'(0.5 / 1.2909944) / 1.2909944 = 0.2408569 / 1.2909944 = 0.1865674
    Sc = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    got = G.ste_grad(torch.ones(4), Sc, "sigmoid-centered", k_remove=1)
    check("ste_grad sigmoid-centered: peak at the keep/remove boundary (hand-computed)",
          abs(float(got[0]) - 0.1865674) < 1e-5 and abs(float(got[1]) - 0.1865674) < 1e-5
          and float(got[1]) > float(got[2]) > float(got[3]), f"{got.tolist()}")

    sc = torch.zeros(3)
    G.score_step(sc, torch.tensor([1.0, -2.0, 0.5]), 0.2, "linf")
    check("score_step linf: largest change = lr, no clipping (hand-computed)",
          torch.allclose(sc, torch.tensor([-0.1, 0.2, -0.05])), f"{sc.tolist()}")
    sc = torch.zeros(3)
    G.score_step(sc, torch.tensor([1.0, -2.0, 0.5]), 0.2, "none")
    check("score_step none: plain step lr * grad", torch.allclose(sc, torch.tensor([-0.2, 0.4, -0.1])))
    sc = torch.ones(3)
    G.score_step(sc, torch.zeros(3), 0.2, "linf")
    check("score_step: zero gradient -> no change", torch.equal(sc, torch.ones(3)))

    # detection from the scores, not from the (binary) mask
    pois = np.array([True, True, False, False, False])
    scores = np.array([-3.0, 2.0, 1.0, -2.0, 3.0])
    mask = G.topk_mask(torch.tensor(scores), 2).numpy()
    st = G.gate_stats(mask, pois, np.flatnonzero(mask == 0), scores=scores)
    # -scores: poisoned 3, -2; clean -1, 2, -3 -> 4 of 6 pairs ordered right
    check("gate_stats with scores: AUROC of -scores = 4/6, precision of the masked rows = 1/2",
          abs(st["auroc_poisoned"] - 4 / 6) < 1e-12 and st["precision_at_R"] == 0.5,
          f"auroc {st['auroc_poisoned']:.4f} precision {st['precision_at_R']}")


# ============================================================================ B. the gate
def encoder_inputs(clf, Xc, yc_t, Xq, *, g=None, null=None, need_grad=False):
    """(feature-encoder input as the encoder sees it, target-encoder input, probs)."""
    from tabfm_experiments.runtime import _forward, _install_recompute
    feat, targ = [], []
    with torch.enable_grad() if need_grad else torch.no_grad():
        clf.fit_with_differentiable_input(Xc, yc_t)
        _install_recompute(clf)
        gate = G.feature_gate(clf, g, null) if g is not None else contextlib.nullcontext()
        with gate:  # capture hooks registered after the gate run after it
            arch = clf.models_[0]
            hs = [arch.feature_group_embedder.register_forward_pre_hook(
                      lambda m, a: feat.append(a[0].detach().clone())),
                  arch.target_embedder.register_forward_pre_hook(
                      lambda m, a: targ.append(a[0].detach().clone()))]
            try:
                p = _forward(clf, Xq, need_grad=need_grad)
            finally:
                for h in hs:
                    h.remove()
    assert len(feat) == 1 and len(targ) == 1, (len(feat), len(targ))
    return feat[0], targ[0], p.detach()


def setup(device, paths=None, dataset="gas-drift", row_percent=20):
    """Data, poison and a classifier, loaded exactly as gate_context.main loads them."""
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2
    ctx, val, test = paths or (G.find_split(dataset, p) for p in ("context_", "val_", "test_attack_"))
    Xc, yc = G.load(ctx, device)
    Xv, yv = G.load(val, device)
    Xt, yt = G.load(test, device)
    info = json.loads((G.DATASETS_DIR / dataset / "splits" / "split_info.json").read_text())
    K = int(info.get("n_classes") or len(np.unique(yc)))
    rows, y_pois = poison("hub-flip-local", Xc, yc, K, row_percent, run_id=0, row_seed=1)
    clf = build_tabpfn_v2(device, 0, recompute_layers=True, n_classes=K)
    return dict(clf=clf, Xc=Xc, yc=yc, Xv=Xv, yv=yv, Xt=Xt, yt=yt, K=K, rows=rows, y_pois=y_pois,
                yp_t=torch.tensor(y_pois, dtype=torch.float32, device=device), paths=(ctx, val, test))


def test_gate(S, device):
    print("\nB. the gate")
    clf, Xc, yp_t, Xt, n = S["clf"], S["Xc"], S["yp_t"], S["Xt"], len(S["yc"])
    null = G.capture_null_input(clf, Xc, yp_t)
    S["null"] = null
    half = null.shape[1] // 2
    check("u_null: 64 groups x [2 values | 2 indicators] = [0, 0, -2, -2]",
          tuple(null.shape) == (64, 4) and bool((null[:, :half] == 0).all())
          and bool((null[:, half:] == -2).all()), f"{tuple(null.shape)}")
    from nan_oracle_experiment import null_feature_input
    ref = null_feature_input(clf, Xc.cpu().numpy(), S["y_pois"].astype(np.float32),
                             Xt.cpu().numpy(), device)
    check("u_null == nan_oracle_experiment's independent capture",
          torch.equal(null.cpu(), ref.cpu()))

    Xq = Xt[:300]
    ones = torch.ones(n, device=device)
    for need_grad in (False, True):
        path = "gradient path" if need_grad else "no-grad path"
        f0, t0, p0 = encoder_inputs(clf, Xc, yp_t, Xq, need_grad=need_grad)
        f1, t1, p1 = encoder_inputs(clf, Xc, yp_t, Xq, g=ones, null=null, need_grad=need_grad)
        check(f"g = 1: encoder input and probs bit-identical to no hook ({path})",
              torch.equal(f0, f1) and torch.equal(p0, p1),
              f"max|dp| {float((p0 - p1).abs().max()):.1e}")

        rng = np.random.default_rng(1)
        gv = rng.choice([0.0, 0.3, 0.77, 1.0], size=n).astype(np.float32)
        g = torch.tensor(gv, device=device)
        f2, t2, _ = encoder_inputs(clf, Xc, yp_t, Xq, g=g, null=null, need_grad=need_grad)
        gi = g[:, None, None].to(f0.device)
        want = gi * f0[:n] + (1 - gi) * null.to(f0.device)
        check(f"gated rows == g*u + (1-g)*u_null exactly ({path})",
              torch.equal(f2[:n], want), f"max diff {float((f2[:n] - want).abs().max()):.1e}")
        check(f"query rows and label-encoder input untouched ({path})",
              torch.equal(f2[n:], f0[n:]) and torch.equal(t2, t0))

        a, b = 17, 3001
        Xs, ys = Xc.clone(), yp_t.clone()
        Xs[[a, b]], ys[[a, b]] = Xc[[b, a]], yp_t[[b, a]]
        f3, _, _ = encoder_inputs(clf, Xs, ys, Xq, need_grad=need_grad)
        perm = torch.arange(f0.shape[0])
        perm[[a, b]] = perm[[b, a]]
        diff = float((f3 - f0[perm]).abs().max())
        gap = float((f0[a] - f0[b]).abs().max())
        check(f"encoder row i = context row i: swapping rows {a},{b} swaps them ({path})",
              diff < 1e-4 < gap, f"swap residual {diff:.1e} vs row gap {gap:.2f}")

    yt_t = torch.as_tensor(S["yt"], device=device)
    p_none = G.evaluate(clf, Xc, yp_t, Xt, yt_t, None, null, need_grad=False, batch_size=None)[2]
    p_ones = G.evaluate(clf, Xc, yp_t, Xt, yt_t, ones, null, need_grad=False, batch_size=None)[2]
    check("evaluate with g = None == evaluate with g = 1 (bit-identical)", np.array_equal(p_none, p_ones))

    # g = 0 on exactly the poisoned rows vs the stored, independently gated oracle run
    z = np.load(ROOT / "results/nan_oracle/gas-drift/part1/probs.npz")
    oracle = torch.ones(n, device=device)
    oracle[torch.as_tensor(S["rows"])] = 0.0
    for name, gate, key in (("g = 0 on poisoned rows", oracle, "R20__feat-gate"),
                            ("g = 1", ones, "R20__poisoned")):
        p = G.evaluate(clf, Xc, yp_t, Xt, yt_t, gate, null, need_grad=False, batch_size=None)[2]
        d = float(np.abs(p - z[key]).max())
        check(f"{name} reproduces nan_oracle_experiment {key}", d < 1e-4, f"max|dp| {d:.1e}")

    # the gradient path and the no-grad path agree for an interior gate
    g = torch.tensor(np.random.default_rng(2).uniform(0.2, 0.8, n), dtype=torch.float32,
                     device=device, requires_grad=True)
    _, _, pa = G.evaluate(clf, Xc, yp_t, Xt[:200], yt_t[:200], g, null, need_grad=True, batch_size=None)
    _, _, pb = G.evaluate(clf, Xc, yp_t, Xt[:200], yt_t[:200], g, null, need_grad=False, batch_size=None)
    check("gradient path probs == no-grad path probs (interior g)",
          float(np.abs(pa - pb).max()) < 1e-4, f"max|dp| {float(np.abs(pa - pb).max()):.1e}")

    try:
        G.evaluate(clf, Xc, yp_t, Xt[:10], yt_t[:10], torch.ones(n + 50, device=device), null,
                   need_grad=False, batch_size=None)
        refused = False
    except RuntimeError:
        refused = True
    check("hook refuses a gate longer than the context", refused)


# ========================================================================== C. gradient
def test_gradient(S, device):
    print("\nC. gradient")
    clf, Xc, yp_t, null, n = S["clf"], S["Xc"], S["yp_t"], S["null"], len(S["yc"])
    Xq, yq = S["Xv"][:200], torch.as_tensor(S["yv"][:200], device=device)
    g0 = torch.tensor(np.random.default_rng(3).uniform(0.2, 0.8, n), dtype=torch.float32,
                      device=device, requires_grad=True)
    _, grad, _ = G.evaluate(clf, Xc, yp_t, Xq, yq, g0, null, need_grad=True, batch_size=None)

    def L(gv):
        return G.evaluate(clf, Xc, yp_t, Xq, yq, gv, null, need_grad=False, batch_size=None)[0]

    # The loss is strongly curved in g (one row: the central difference is -0.07 at
    # h = 0.1 but -0.21 at h <= 0.003, which is autograd's value), so the steps must be
    # small. Measured convergence: h from 0.1 down to 3e-4.
    gd = g0.detach()
    top = torch.argsort(grad.abs(), descending=True)
    rnd = torch.zeros(n, device=device)
    signs = torch.randint(0, 2, (200,), generator=torch.Generator().manual_seed(0)) * 2.0 - 1.0
    rnd[top[:200]] = signs.to(device)  # random signs: not aligned with the gradient
    for name, d, h in (("sign(grad)", torch.sign(grad), 1e-4),
                       ("random signs on the 200 largest rows", rnd, 1e-3)):
        fd = (L(gd + h * d) - L(gd - h * d)) / (2 * h)
        an = float((grad * d).sum())
        rel = abs(fd - an) / abs(an)
        check(f"directional derivative, {name}: finite diff == autograd",
              rel < 0.02, f"fd {fd:.5e}  autograd {an:.5e}  rel {rel:.1e}")
    for i in top[:5].tolist():
        e = torch.zeros(n, device=device)
        e[i] = 1e-3
        fd = (L(gd + e) - L(gd - e)) / 2e-3
        an = float(grad[i])
        rel = abs(fd - an) / abs(an)
        check(f"d CE / d g[{i}]: finite diff (h = 1e-3) == autograd", rel < 0.02,
              f"fd {fd:.4e}  autograd {an:.4e}  rel {rel:.1e}")

    _, grad_c, _ = G.evaluate(clf, Xc, yp_t, Xq, yq, g0, null, need_grad=True, batch_size=80)
    rel = float((grad_c - grad).abs().max() / grad.abs().max())
    check("chunked (80/80/40) gradient == unchunked gradient", rel < 1e-3, f"rel max diff {rel:.1e}")


# ======================================================================= D. end-to-end
def wandb_records(path: Path):
    """The protobuf records of a W&B run file (run-<id>.wandb), read independently of W&B.

    wandb 0.30 keeps history, summary and artifacts in this file (offline runs write no
    wandb-summary.json). Framing is LevelDB's log format: a 7-byte file header, then
    32 KiB blocks of [crc32c u32][length u16][type u8][payload]; type 1 = whole record,
    2/3/4 = first/middle/last fragment."""
    from wandb.proto import wandb_internal_pb2 as pb
    data, pos, buf = path.read_bytes(), 7, b""
    while pos < len(data):
        left = 32768 - pos % 32768
        if left < 7:  # block trailer
            pos += left
            continue
        _, n, t = struct.unpack("<IHB", data[pos:pos + 7])
        pos += 7
        payload, pos = data[pos:pos + n], pos + n
        if t == 0 and n == 0:  # zero padding to the end of the block
            pos += left - 7
            continue
        if t in (2, 3):
            buf = payload if t == 2 else buf + payload
            continue
        r = pb.Record()
        r.ParseFromString(buf + payload if t == 4 else payload)
        buf = b""
        yield r


def check_run(out: Path, S, device, *, optimizer: str, lr: float, steps: int, test_every: int,
              tag: str, row_percent: float = 20, holdout: bool = False) -> dict:
    """Checks on one finished gate_context run, against independent recomputation.

    S["Xv"], S["yv"] are the rows the gates were fitted on. With holdout, S["Xh"], S["yh"]
    are the held-out validation-test rows and S["Xv_all"], S["yv_all"] all validation rows."""
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import evaluate_context
    from tabfm_experiments.sampling import k_from_percent

    names = (["config.json", "run.log", "log.csv", "test_metrics.json", "test_probs.npz",
              "poison.npz", "gates/final.npz"] + [f"gates/top{k}.npz" for k in (1, 2, 3)]
             + [f"plots/{m}.png" for m in G.METRICS]
             + ["plots/gates.png", "plots/detection.png", "plots/gate_hist_best.png"])
    missing = [s for s in names if not (out / s).is_file() or (out / s).stat().st_size == 0]
    check(f"{tag} every output file exists and is non-empty", not missing, f"{missing}")

    rows = list(csv.DictReader(open(out / "log.csv")))
    check(f"{tag} log.csv has steps+1 rows; iteration 0 is ungated",
          len(rows) == steps + 1 and float(rows[0]["g_mean"]) == 1.0
          and float(rows[0]["auroc_poisoned"]) == 0.5)
    want_test = {it for it in range(steps + 1) if it % test_every == 0 or it == steps}
    got_test = {int(r["iter"]) for r in rows if r["test_soft_ce"] != ""}
    check(f"{tag} test scores logged on exactly the test iterations", got_test == want_test,
          f"{sorted(got_test)}")
    if holdout:
        got_h = {int(r["iter"]) for r in rows if r.get("valtest_ce", "") != ""}
        check(f"{tag} validation-test scores logged on exactly the test iterations", got_h == want_test,
              f"{sorted(got_h)}")
    dg = float(rows[0]["max_abs_dg"])
    check(f"{tag} first update moves the largest gate by lr",
          abs(dg - lr) < (1e-6 if optimizer == "pgd" else 1e-3), f"{dg:.6f}")

    sel = "valtest" if holdout else "val"  # the split the best iteration is chosen on
    cand = [r for r in rows if r.get(f"{sel}_ce", "") != ""]
    order = sorted(cand, key=lambda r: (-float(r[f"{sel}_accuracy"]), float(r[f"{sel}_ce"]), int(r["iter"])))
    ok = True
    for k in (1, 2, 3):
        zk = np.load(out / f"gates/top{k}.npz")
        ok &= int(zk["iteration"]) == int(order[k - 1]["iter"])
        ok &= abs(float(zk["g"].mean()) - float(order[k - 1]["g_mean"])) < 1e-5
    check(f"{tag} top-3 files = log rows sorted by ({sel} acc desc, {sel} ce asc, iter)", ok,
          f"{[int(r['iter']) for r in order[:3]]}")

    pz = np.load(out / "poison.npz")
    check(f"{tag} poisoned rows/labels match poison(); labels differ only there",
          np.array_equal(pz["poisoned_rows"], S["rows"]) and np.array_equal(pz["y_poisoned"], S["y_pois"])
          and set(np.flatnonzero(pz["y_poisoned"] != pz["y_clean"])) == set(S["rows"].tolist()))

    clf, Xc, yp_t, Xt = S["clf"], S["Xc"], S["yp_t"], S["Xt"]
    yv_t, yt_t = torch.as_tensor(S["yv"], device=device), torch.as_tensor(S["yt"], device=device)
    ce_ref, _, _ = evaluate_context(clf, Xc, yp_t, S["Xv"], yv_t, need_grad=False)
    d = abs(float(ce_ref) - float(rows[0]["val_ce"]))
    check(f"{tag} iteration-0 validation CE == runtime.evaluate_context", d < 1e-5, f"diff {d:.1e}")
    if holdout:
        ce_h, _, _ = evaluate_context(clf, Xc, yp_t, S["Xh"], torch.as_tensor(S["yh"], device=device),
                                      need_grad=False)
        d = abs(float(ce_h) - float(rows[0]["valtest_ce"]))
        check(f"{tag} iteration-0 validation-test CE == runtime.evaluate_context on the held-out rows",
              d < 1e-5, f"diff {d:.1e}")

    tm = json.loads((out / "test_metrics.json").read_text())
    tp = np.load(out / "test_probs.npz")
    ok = all(abs(multiclass_metrics(tp[k], S["yt"], S["K"])[m] - tm["metrics"][k][m]) < 1e-9
             for k in tp.files if k != "y_test" for m in G.METRICS)
    check(f"{tag} test_metrics.json == metrics recomputed from test_probs.npz", ok)

    _, _, p0 = evaluate_context(clf, Xc, yp_t, Xt, yt_t, need_grad=False)
    m0 = multiclass_metrics(p0.cpu().numpy(), S["yt"], S["K"])
    check(f"{tag} iteration-0 soft test scores == runtime.evaluate_context, ungated",
          all(abs(float(rows[0][f"test_soft_{m}"]) - m0[m]) < 1e-6 for m in G.METRICS))

    n, tie = len(S["yc"]), pz["tie_order"]
    k = k_from_percent(n, row_percent)
    rand_keep = np.flatnonzero(tie >= k)  # all gates equal: the k smallest tie keys go
    _, _, pr = evaluate_context(clf, Xc[rand_keep], yp_t[rand_keep], Xt, yt_t, need_grad=False)
    mr = multiclass_metrics(pr.cpu().numpy(), S["yt"], S["K"])
    check(f"{tag} iteration-0 hard test scores == a random R% removal (independent)",
          all(abs(float(rows[0][f"test_hard_{m}"]) - mr[m]) < 1e-6 for m in G.METRICS)
          and np.abs(tp["poisoned_random_removal"] - pr.cpu().numpy()).max() < 1e-6)

    z1 = np.load(out / "gates/top1.npz")
    g1 = z1["g"]
    drop = sorted(range(n), key=lambda i: (g1[i], tie[i]))[:k]  # independent of lowest_gated
    keep = np.setdiff1d(np.arange(n), drop)
    _, _, ph = evaluate_context(clf, Xc[keep], yp_t[keep], Xt, yt_t, need_grad=False)
    d = float(np.abs(tp["poisoned_best_gate_hard"] - ph.cpu().numpy()).max())
    check(f"{tag} best gates, hard: == evaluate_context without the k lowest-gated rows",
          d < 1e-6, f"max|dp| {d:.1e}")
    p = G.evaluate(clf, Xc, yp_t, Xt, yt_t, torch.tensor(g1, device=device), S["null"],
                   need_grad=False, batch_size=None)[2]
    d = float(np.abs(p - tp["poisoned_best_gate_soft"]).max())
    check(f"{tag} best gates, soft: re-scoring gates/top1.npz reproduces the saved probs",
          d < 1e-6, f"max|dp| {d:.1e}")
    check(f"{tag} test_metrics best iteration == top-1 file", tm["best_iteration"] == int(z1["iteration"]))

    # the two references that use the clean validation rows as context, recomputed
    Xv_all = S.get("Xv_all", S["Xv"])  # every validation row, fitted on or held out
    yv_f = torch.as_tensor(S.get("yv_all", S["yv"]), dtype=torch.float32, device=device)
    for name, Xr, yr in (("val_as_context", Xv_all, yv_f),
                         ("poisoned_plus_val", torch.cat([Xc, Xv_all]), torch.cat([yp_t, yv_f]))):
        _, _, pr = evaluate_context(clf, Xr, yr, Xt, yt_t, need_grad=False)
        d = float(np.abs(tp[name] - pr.cpu().numpy()).max()) if name in tp.files else float("inf")
        check(f"{tag} reference {name} == evaluate_context on {len(yr)} context rows",
              d < 1e-6, f"max|dp| {d:.1e}")
    vs = tm.get("vs_val_as_context", {})
    m = tm["metrics"]
    check(f"{tag} vs_val_as_context == best soft gates - validation set as context",
          set(vs) == set(G.METRICS) and all(
              abs(vs[k] - (m["poisoned_best_gate_soft"][k] - m["val_as_context"][k])) < 1e-12
              for k in G.METRICS), f"{vs}")
    sr = tm.get("selection_rules", {})
    want_rules = {"val_acc", "val_first_max", "last", "test_acc_oracle"} | (
        {"valtest_acc", "valtest_ce"} if holdout else set())
    ok = set(sr) == want_rules and all(
        abs(v["test_soft"]["accuracy"] - float(rows[v["iteration"]]["test_soft_accuracy"])) < 1e-12
        and abs(v["test_hard"]["accuracy"] - float(rows[v["iteration"]]["test_hard_accuracy"])) < 1e-12
        for v in sr.values())
    check(f"{tag} selection_rules: every rule, its test scores == log.csv at its iteration", ok,
          f"{ {k: v['iteration'] for k, v in sr.items()} }")
    if holdout:
        check(f"{tag} best iteration == the validation-test accuracy rule's",
              tm["best_iteration"] == sr.get("valtest_acc", {}).get("iteration"))

    # the newest local run (a --force rerun leaves the earlier run's folder in place);
    # folder names start with the start time, so name order is time order
    runs = sorted((out / "wandb").glob("*run-*/run-*.wandb"), key=lambda f: f.parent.name.split("run-", 1)[1])
    hist, summ, arts, name = [], {}, [], None
    for r in (wandb_records(runs[-1]) if runs else []):
        kind = r.WhichOneof("record_type")
        if kind == "run":
            name = r.run.display_name
        elif kind == "history":
            hist.append({i.key or "/".join(i.nested_key): i.value_json for i in r.history.item})
        elif kind == "summary":
            summ.update({i.key or "/".join(i.nested_key): i.value_json for i in r.summary.update})
        elif kind == "artifact" and r.artifact.type == "gate-run":  # (the overlay charts'
            arts.append({c.path for c in r.artifact.manifest.contents})  # tables are artifacts too)
    by_step = {}
    for h in hist:  # a step can arrive in more than one record (e.g. the final images)
        by_step.setdefault(int(h["_step"]), {}).update(h)

    def same(step, wkey, ckey):  # the W&B value equals the log.csv value
        return wkey in by_step.get(step, {}) and float(by_step[step][wkey]) == float(rows[step][ckey])

    every = [(f"{m}/val", f"val_{m}") for m in G.METRICS] + [
        (f"detection/{w}", c) for c, w in G.DETECTION.items()]
    ok = sorted(by_step) == list(range(steps + 1)) and all(
        same(it, w, c) for it in range(steps + 1) for w, c in every)
    check(f"{tag} W&B every iteration: all 5 val metrics + detection precision/recall/auroc/ap",
          ok, f"{len(by_step)} steps x {len(every)} keys, equal to log.csv")
    splits = ("test_soft", "test_hard") + (("valtest",) if holdout else ())
    tests = [(f"{m}/{s}", f"{s}_{m}") for s in splits for m in G.METRICS]
    got = {it for it, h in by_step.items() if any(w in h for w, _ in tests)}
    ok = got == want_test and all(same(it, w, c) for it in want_test for w, c in tests)
    check(f"{tag} W&B test iterations only: all 5 metrics on {', '.join(splits)}", ok, f"steps {sorted(got)}")
    # validation every iteration + the test-iteration curves + the flat validation-as-context line
    n_pts = (steps + 1) + len(splits) * len(want_test) + 2
    last = by_step.get(steps, {})
    ok = (len(runs) >= 1 and name == out.name
          and all(last.get(f"overlay/{m}_table/nrows") == str(n_pts) for m in G.METRICS)
          and summ.get("best_iteration") == str(tm["best_iteration"])
          and len(arts) == 1 and set(names) <= arts[0])
    check(f"{tag} W&B: run name, overlay charts ({n_pts} points each), summary, artifact files", ok,
          f"{len(runs)} run file(s)")
    return tm


def test_end_to_end(S, device, gpu):
    print("\nD. end-to-end")
    tmp = Path(tempfile.mkdtemp(prefix="gate_test_"))
    try:
        for optimizer in ("pgd", "adam"):
            steps, lr, test_every = 3, 0.05, 2
            out = G.main(["--gpu", gpu, "--row-percent", "20", "--steps", str(steps),
                          "--optimizer", optimizer, "--lr", str(lr), "--test-every", str(test_every),
                          "--out", str(tmp), "--plot-every", "2", "--wandb-mode", "offline"])
            tag = f"[{optimizer}]"
            tm = check_run(out, S, device, optimizer=optimizer, lr=lr, steps=steps,
                           test_every=test_every, tag=tag)

            # the references against stored results of earlier, independent runs
            grid = json.loads((ROOT / "results/multiclass/gas-drift/hub-flip-local_r020/summary.json")
                              .read_text())["runs"][0]["metrics"]
            pm = tm["metrics"]["poisoned_no_gate"]
            check(f"{tag} ungated test metrics == stored multiclass grid (R=20)",
                  abs(pm["ce"] - grid["ce"]) < 1e-4 and pm["accuracy"] == grid["accuracy"],
                  f"ce diff {abs(pm['ce'] - grid['ce']):.1e}")
            z = np.load(ROOT / "results/nan_oracle/gas-drift/part1/probs.npz")
            tp = np.load(out / "test_probs.npz")
            for name, key in (("poisoned_oracle_gate", "R20__feat-gate"),
                              ("poisoned_oracle_removal", "R20__remove")):
                d = float(np.abs(tp[name] - z[key]).max())
                check(f"{tag} {name} == stored nan_oracle_experiment {key}", d < 1e-4, f"max|dp| {d:.1e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ======================================================= E. validation drawn from the context
def wandb_run_info(out: Path) -> tuple[str, dict]:
    """(group, config) of the newest local W&B run under out, read from its run file."""
    runs = sorted((out / "wandb").glob("*run-*/run-*.wandb"), key=lambda f: f.parent.name.split("run-", 1)[1])
    group, config = None, {}
    for r in (wandb_records(runs[-1]) if runs else []):
        kind = r.WhichOneof("record_type")
        if kind == "run":
            group = r.run.run_group
            config.update({i.key: json.loads(i.value_json) for i in r.run.config.update})
        elif kind == "config":
            config.update({i.key: json.loads(i.value_json) for i in r.config.update})
    return group, config


def test_val_split(device, gpu):
    print("\nE. validation drawn from the context")
    from prepare_multiclass_splits import stratified_exact
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2

    tmp = Path(tempfile.mkdtemp(prefix="gate_valsplit_test_"))
    try:
        base = ["--gpu", gpu, "--row-percent", "20", "--out", str(tmp), "--wandb-mode", "disabled"]
        for extra, why in ((["--val-from-context", "100", "--val", str(G.find_split("gas-drift", "val_"))],
                            "together with --val"),
                           (["--val-from-context", "4"], "when the draw misses a class (6 classes, 4 rows)"),
                           (["--val-from-context", "5000"], "when it would take the whole context")):
            try:
                G.main(base + ["--steps", "0"] + extra)
                refused = None
            except SystemExit as e:  # argparse exits with code 2; the script's guards with a message
                refused = e.code
            check(f"--val-from-context refused {why}", isinstance(refused, str), f"{refused!r}")

        V, steps, lr, test_every = 100, 3, 0.05, 2
        out = G.main(base[:-2] + ["--val-from-context", str(V), "--val-seed", "0", "--steps", str(steps),
                                  "--test-every", str(test_every), "--plot-every", "2",
                                  "--wandb-mode", "offline", "--wandb-group", "valsplit-test"])
        tag = "[val from context]"
        sp = np.load(out / "split.npz")
        vr, cr = sp["val_rows"], sp["context_rows"]
        X_all, y_all = G.load(G.find_split("gas-drift", "context_"), device)
        n_all = len(y_all)
        check(f"{tag} split: {V} validation + {n_all - V} context rows, disjoint, covering the file",
              len(vr) == V and len(cr) == n_all - V and len(np.intersect1d(vr, cr)) == 0
              and np.array_equal(np.union1d(vr, cr), np.arange(n_all)))
        ref = stratified_exact(np.arange(n_all), y_all, V, np.random.default_rng(0))
        check(f"{tag} validation class counts == stratified_exact's quotas",
              np.array_equal(np.bincount(y_all[vr], minlength=6), np.bincount(y_all[ref], minlength=6)))

        # everything check_run needs, rebuilt from the file and the saved split
        Xc, yc = X_all[torch.as_tensor(cr, device=device)], y_all[cr]
        Xt, yt = G.load(G.find_split("gas-drift", "test_attack_"), device)
        rows, y_pois = poison("hub-flip-local", Xc, yc, 6, 20, run_id=0, row_seed=1)
        clf = build_tabpfn_v2(device, 0, recompute_layers=True, n_classes=6)
        yp_t = torch.tensor(y_pois, dtype=torch.float32, device=device)
        S = dict(clf=clf, Xc=Xc, yc=yc, Xv=X_all[torch.as_tensor(vr, device=device)], yv=y_all[vr],
                 Xt=Xt, yt=yt, K=6, rows=rows, y_pois=y_pois, yp_t=yp_t)
        S["null"] = G.capture_null_input(clf, Xc, yp_t)
        check_run(out, S, device, optimizer="pgd", lr=lr, steps=steps, test_every=test_every, tag=tag)

        cfg = json.loads((out / "config.json").read_text())
        check(f"{tag} config: {n_all - V} context rows and {V} validation rows",
              cfg["n_context"] == n_all - V and cfg["n_val"] == V, f"{cfg['n_context']} / {cfg['n_val']}")
        group, wcfg = wandb_run_info(out)
        args = wcfg.get("args", {})
        check(f"{tag} W&B run: group, and the draw's size and seed in its config",
              group == "valsplit-test" and args.get("val_from_context") == V and args.get("val_seed") == 0,
              f"group {group!r}, val_from_context {args.get('val_from_context')}")

        # ---- a 20% validation-test holdout: fit on 80%, choose the iteration on 20% ----------
        for extra, why in ((["--val-holdout", "0.2", "--test-every", "0"], "without test iterations"),
                           (["--val-holdout", "1"], "with nothing left to fit on")):
            try:
                G.main(base + ["--steps", "0", "--val-from-context", "50"] + extra)
                refused = None
            except SystemExit as e:
                refused = e.code
            check(f"--val-holdout refused {why}", isinstance(refused, str), f"{refused!r}")

        V, steps, test_every = 50, 3, 1
        out = G.main(base[:-2] + ["--val-from-context", str(V), "--val-seed", "0", "--val-holdout", "0.2",
                                  "--steps", str(steps), "--test-every", str(test_every), "--plot-every", "2",
                                  "--wandb-mode", "offline", "--wandb-group", "valsplit-test"])
        tag = "[holdout]"
        sp = np.load(out / "split.npz")
        vr, fr, hr, cr = sp["val_rows"], sp["val_fit_rows"], sp["val_holdout_rows"], sp["context_rows"]
        check(f"{tag} split: 50 validation rows = 40 fitted + 10 held out, disjoint",
              len(vr) == 50 and len(fr) == 40 and len(hr) == 10 and len(np.intersect1d(fr, hr)) == 0
              and np.array_equal(np.union1d(fr, hr), vr) and len(np.intersect1d(vr, cr)) == 0)
        check(f"{tag} held-out rows cover every class", len(np.unique(y_all[hr])) == 6,
              f"{np.bincount(y_all[hr], minlength=6).tolist()}")
        sel = lambda r: X_all[torch.as_tensor(r, device=device)]  # noqa: E731
        Xc, yc = sel(cr), y_all[cr]
        rows, y_pois = poison("hub-flip-local", Xc, yc, 6, 20, run_id=0, row_seed=1)
        yp_t = torch.tensor(y_pois, dtype=torch.float32, device=device)
        S = dict(clf=clf, Xc=Xc, yc=yc, Xv=sel(fr), yv=y_all[fr], Xh=sel(hr), yh=y_all[hr],
                 Xv_all=sel(vr), yv_all=y_all[vr], Xt=Xt, yt=yt, K=6, rows=rows, y_pois=y_pois, yp_t=yp_t)
        S["null"] = G.capture_null_input(clf, Xc, yp_t)
        check_run(out, S, device, optimizer="pgd", lr=lr, steps=steps, test_every=test_every, tag=tag,
                  holdout=True)
        cfg = json.loads((out / "config.json").read_text())
        check(f"{tag} config: 40 fitted and 10 held-out validation rows",
              cfg["validation"].get("n_fit") == 40 and cfg["validation"].get("n_holdout") == 10,
              f"{cfg['validation']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================== F. top-k gates
def test_topk(device, gpu):
    print("\nF. top-k gates (exactly R% of the rows masked, straight-through scores)")
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2, evaluate_context
    from tabfm_experiments.sampling import k_from_percent
    from sklearn.metrics import roc_auc_score

    tmp = Path(tempfile.mkdtemp(prefix="gate_topk_test_"))
    try:
        base = ["--gpu", gpu, "--row-percent", "20", "--out", str(tmp), "--val-from-context", "50",
                "--val-holdout", "0.2", "--gate", "topk"]
        for extra, why in ((["--optimizer", "adam"], "with Adam"),
                           (["--remove-percent", "0"], "with nothing to remove")):
            try:
                G.main(base + ["--steps", "0", "--wandb-mode", "disabled"] + extra)
                refused = None
            except SystemExit as e:
                refused = e.code
            check(f"--gate topk refused {why}", isinstance(refused, str), f"{refused!r}")

        steps, test_every, lr, std, seed = 4, 2, 0.2, 1.0, 3
        out = G.main(base + ["--ste", "sigmoid", "--score-init-std", str(std), "--init-seed", str(seed),
                             "--lr", str(lr), "--select-on", "ce", "--steps", str(steps),
                             "--test-every", str(test_every), "--plot-every", "2",
                             "--wandb-mode", "offline", "--wandb-group", "topk-test"])
        tag = "[topk]"
        rows = list(csv.DictReader(open(out / "log.csv")))
        sp = np.load(out / "split.npz")
        X_all, y_all = G.load(G.find_split("gas-drift", "context_"), device)
        cr = sp["context_rows"]
        n = len(cr)
        k = k_from_percent(n, 20)
        Xc, yc = X_all[torch.as_tensor(cr, device=device)], y_all[cr]
        prows, y_pois = poison("hub-flip-local", Xc, yc, 6, 20, run_id=0, row_seed=1)
        pois = np.zeros(n, bool)
        pois[prows] = True

        check(f"{tag} every iteration masks exactly k = {k} of {n} rows",
              len(rows) == steps + 1 and all(abs(float(r["g_mean"]) - (1 - k / n)) < 1e-6
                                             and abs(float(r["frac_g_below_0.5"]) - k / n) < 1e-6 for r in rows))
        # the documented initialisation: N(0, std^2) drawn on the CPU from Generator(seed)
        s0 = (torch.randn(n, generator=torch.Generator().manual_seed(seed)) * std).numpy()
        m0 = np.ones(n)
        m0[np.argsort(s0, kind="stable")[:k]] = 0
        check(f"{tag} iteration 0 = the k lowest initial scores (kept share of poisoned rows, precision)",
              abs(float(rows[0]["g_mean_poisoned"]) - m0[pois].mean()) < 1e-6
              and abs(float(rows[0]["precision_at_R"]) - pois[m0 == 0].mean()) < 1e-9,
              f"{rows[0]['g_mean_poisoned']} vs {m0[pois].mean():.6f}")
        check(f"{tag} each update moves the largest score by lr; mask flips come in pairs",
              all(abs(float(r["max_abs_dg"]) - lr) < 1e-5 for r in rows[:-1])
              and all(int(float(r["mask_flips"])) % 2 == 0 for r in rows[:-1]),
              f"{[r['mask_flips'] for r in rows[:-1]]}")

        fz = np.load(out / "gates/final.npz")
        mf = np.ones(n)
        mf[np.argsort(fz["scores"], kind="stable")[:k]] = 0
        check(f"{tag} final.npz: mask == the k lowest of its scores", np.array_equal(fz["g"], mf))

        tm = json.loads((out / "test_metrics.json").read_text())
        cand = [r for r in rows if r["valtest_ce"] != ""]
        best = min(cand, key=lambda r: (float(r["valtest_ce"]), -float(r["valtest_accuracy"]), int(r["iter"])))
        z1 = np.load(out / "gates/top1.npz")
        check(f"{tag} best iteration = lowest validation-test CE (ties: accuracy, earlier) = top-1 file",
              tm["best_iteration"] == int(best["iter"]) == int(z1["iteration"]),
              f"{tm['best_iteration']} / {best['iter']} / {int(z1['iteration'])}")
        m1 = np.ones(n)
        m1[np.argsort(z1["scores"], kind="stable")[:k]] = 0
        bs = tm["best_gate_stats"]
        check(f"{tag} best gates: mask from its scores; precision@R and score AUROC recomputed",
              np.array_equal(z1["g"], m1) and abs(bs["precision_at_R"] - pois[m1 == 0].mean()) < 1e-9
              and abs(bs["auroc_poisoned"] - roc_auc_score(pois, -z1["scores"])) < 1e-9,
              f"precision {bs['precision_at_R']:.3f} auroc {bs['auroc_poisoned']:.3f}")

        clf = build_tabpfn_v2(device, 0, recompute_layers=True, n_classes=6)
        Xt, yt = G.load(G.find_split("gas-drift", "test_attack_"), device)
        yt_t = torch.as_tensor(yt, device=device)
        yp_t = torch.tensor(y_pois, dtype=torch.float32, device=device)
        tp = np.load(out / "test_probs.npz")
        keep = np.flatnonzero(m1 == 1)
        _, _, ph = evaluate_context(clf, Xc[keep], yp_t[keep], Xt, yt_t, need_grad=False)
        d = float(np.abs(tp["poisoned_best_gate_hard"] - ph.cpu().numpy()).max())
        check(f"{tag} best gates, hard: == evaluate_context without the masked rows", d < 1e-6, f"max|dp| {d:.1e}")
        null = G.capture_null_input(clf, Xc, yp_t)
        p = G.evaluate(clf, Xc, yp_t, Xt, yt_t, torch.tensor(m1, dtype=torch.float32, device=device), null,
                       need_grad=False, batch_size=None)[2]
        d = float(np.abs(p - tp["poisoned_best_gate_soft"]).max())
        check(f"{tag} best gates, masked: re-scoring the top-1 mask reproduces the saved probs", d < 1e-6,
              f"max|dp| {d:.1e}")
        ok = all(abs(multiclass_metrics(tp[key], yt, 6)[m] - tm["metrics"][key][m]) < 1e-9
                 for key in tp.files if key != "y_test" for m in G.METRICS)
        check(f"{tag} test_metrics.json == metrics recomputed from test_probs.npz", ok)

        cfg = json.loads((out / "config.json").read_text())
        group, wcfg = wandb_run_info(out)
        args = wcfg.get("args", {})
        check(f"{tag} config and W&B: gate, surrogate, init std and seed, select-on, group",
              cfg["gate"] == {"type": "topk", "ste": "sigmoid", "score_init_std": std, "init_seed": seed,
                              "k_remove": k, "select_on": "ce"}
              and group == "topk-test" and args.get("gate") == "topk" and args.get("ste") == "sigmoid"
              and args.get("score_init_std") == std and args.get("init_seed") == seed,
              f"{cfg.get('gate')} group {group!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


SECTIONS = ("rules", "draw", "gate", "gradient", "e2e", "valsplit", "topk")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=SECTIONS)
    a = ap.parse_args(argv)
    device = f"cuda:{a.gpu}" if a.gpu != "cpu" else "cpu"
    if "rules" in a.sections:
        test_update_rules()
    if "draw" in a.sections:
        test_draw_validation()
        test_holdout_and_rules()
    if "topk" in a.sections:
        test_topk_pieces()
    if {"gate", "gradient", "e2e"} & set(a.sections):
        S = setup(device)
        test_gate(S, device)  # also captures S["null"], which the later sections use
        if "gradient" in a.sections:
            test_gradient(S, device)
        if "e2e" in a.sections:
            test_end_to_end(S, device, a.gpu)
    if "valsplit" in a.sections:
        test_val_split(device, a.gpu)
    if "topk" in a.sections:
        test_topk(device, a.gpu)
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

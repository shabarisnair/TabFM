#!/usr/bin/env python
"""Pick one restart, or let the restarts vote? (top-k gating, post hoc)

Each top-k run saves its hard mask at the chosen iteration (gates/top1.npz: g, scores). For
the restarts of one setting (same draw, holdout and poison) this compares:

  each restart       its own mask (reproduces the run's logged numbers)
  best restart       the restart with the lowest held-out CE (the current rule)
  vote               switch off the k rows masked by the most restarts; ties broken by the
                     mean within-restart score rank (low = switched off first)
  mean rank          switch off the k rows with the lowest mean score rank

--which final takes every restart's mask after its last update instead of at its chosen
iteration (then no step of the vote looks at the held-out rows). --curve N adds, for the
pooled restarts, the vote of r = 1, 2, ... restarts averaged over up to N random subsets
of each size: how many restarts the vote needs. Every mask switches off exactly k rows. Reported: CE on the fitted validation rows, held-out
CE and accuracy, test accuracy and precision@R. The held-out CE decides; test is reported.

    python scripts/consensus_topk_masks.py --gpu 0 --dataset gas-drift \\
        --runs '*_lr5_init?_probe0.05_it300' '*_lr5_init?_probe0.1_it300'
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402
from diagnose_topk_masks import CONTEXT, TEST  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", required=True, choices=list(CONTEXT))
    ap.add_argument("--runs", nargs="+", required=True,
                    help="globs of run directories under results/gate_topk/<dataset>/hub-flip-local_r020; "
                         "each glob is one setting, and all of them together are pooled as well")
    ap.add_argument("--curve", type=int, default=0, metavar="N",
                    help="also the vote of r = 1.. restarts (pooled), mean over up to N subsets per r")
    ap.add_argument("--which", default="top1", choices=["top1", "final"],
                    help="each restart's mask at its chosen iteration (top1) or after its last update (final)")
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = f"cuda:{a.gpu}"
    root = Path("results/gate_topk", a.dataset, "hub-flip-local_r020")
    groups = {pat: sorted(d for d in root.glob(pat) if (d / "test_metrics.json").exists()) for pat in a.runs}
    first = next(d for ds in groups.values() for d in ds)
    sp, pz = np.load(first / "split.npz"), np.load(first / "poison.npz")
    for d in (d for ds in groups.values() for d in ds):  # every restart must share the draw and the poison
        assert (np.load(d / "split.npz")["context_rows"] == sp["context_rows"]).all(), d
        assert (np.load(d / "poison.npz")["poisoned_rows"] == pz["poisoned_rows"]).all(), d
    X_all, y_all = G.load(Path(CONTEXT[a.dataset]), device)
    Xt, yt = G.load(Path(TEST[a.dataset]), device)
    sel = lambda r: X_all[torch.as_tensor(r, device=device)]  # noqa: E731
    Xc, Xf, Xh = sel(sp["context_rows"]), sel(sp["val_fit_rows"]), sel(sp["val_holdout_rows"])
    yf = torch.as_tensor(y_all[sp["val_fit_rows"]], device=device)
    yh = torch.as_tensor(y_all[sp["val_holdout_rows"]], device=device)
    yt_t = torch.as_tensor(yt, device=device)
    n, K = len(sp["context_rows"]), len(np.unique(y_all))
    yp = torch.tensor(pz["y_poisoned"], dtype=torch.float32, device=device)
    pois = np.zeros(n, bool)
    pois[pz["poisoned_rows"]] = True
    k = int(pois.sum())
    clf = build_tabpfn_v2(device, 0, recompute_layers=True, n_classes=K)
    null = G.capture_null_input(clf, Xc, yp)

    def score(m):
        gt = torch.as_tensor(m, dtype=torch.float32, device=device)
        ce_f = G.evaluate(clf, Xc, yp, Xf, yf, gt, null, need_grad=False, batch_size=None)[0]
        ce_h, _, ph = G.evaluate(clf, Xc, yp, Xh, yh, gt, null, need_grad=False, batch_size=None)
        pt = G.evaluate(clf, Xc, yp, Xt, yt_t, gt, null, need_grad=False, batch_size=None)[2]
        acc_h = float((ph.argmax(1) == y_all[sp["val_holdout_rows"]]).mean())
        return ce_f, ce_h, acc_h, multiclass_metrics(pt, yt, K)["accuracy"], pois[m == 0].mean()

    def off_lowest(key):  # mask with the k rows of the lowest key switched off
        m = np.ones(n)
        m[np.lexsort(key[::-1])[:k] if isinstance(key, tuple) else np.argsort(key, kind="stable")[:k]] = 0
        return m

    print(f"{a.dataset}: k = {k} of {n} rows switched off; each restart's mask: {a.which}")
    print(f"  {'mask':<52s} {'fitted CE':>9s} {'hold CE':>8s} {'hold acc':>8s} {'test acc':>8s} {'prec@R':>6s}")
    pooled = []
    for pat, dirs in list(groups.items()) + [("all settings pooled", None)]:
        if dirs is None:
            if len(groups) < 2:
                break
            restarts = pooled
        else:
            restarts = []
            for d in dirs:
                z = np.load(d / "gates" / f"{a.which}.npz")
                g, s = z["g"], z["scores"]
                assert ((g == 0) | (g == 1)).all() and (g == 0).sum() == k, d
                rank = np.argsort(np.argsort(s, kind="stable"), kind="stable") / (n - 1)  # 0 = lowest score
                restarts.append((d.name.split("_H20_")[-1], g, rank, score(g)))
            pooled += restarts
        print(f"\n  {pat}  ({len(restarts)} restarts)")
        if dirs is not None:
            for name, g, _, r in restarts:
                print(f"    {name:<50s} {r[0]:>9.4f} {r[1]:>8.4f} {r[2]:>8.3f} {r[3]:>8.3f} {r[4]:>6.3f}")
        best = min(restarts, key=lambda x: (x[3][1], -x[3][2]))
        votes = sum(1 - g for _, g, _, _ in restarts)
        mean_rank = np.mean([rk for _, _, rk, _ in restarts], axis=0)
        for label, r in [(f"best restart by held-out CE ({best[0]})", best[3]),
                         ("vote (ties: mean rank)", score(off_lowest((-votes, mean_rank)))),
                         ("mean rank", score(off_lowest(mean_rank)))]:
            print(f"    {label:<50s} {r[0]:>9.4f} {r[1]:>8.4f} {r[2]:>8.3f} {r[3]:>8.3f} {r[4]:>6.3f}")
        hist = np.bincount(votes.astype(int), minlength=len(restarts) + 1)
        print(f"    rows masked by 0..{len(restarts)} restarts: {hist.tolist()}; "
              f"poisoned share by vote count: "
              + " ".join(f"{v}:{pois[votes == v].mean():.2f}" for v in range(len(restarts) + 1) if hist[v]))

    if a.curve:  # vote of r restarts, r = 1 .. all, averaged over random subsets
        rng = np.random.default_rng(0)
        R = len(pooled)
        print(f"\n  vote of r of the {R} pooled restarts: mean (sd) over up to {a.curve} random subsets")
        print(f"    {'r':>2s} {'subsets':>7s} {'fitted CE':>15s} {'hold CE':>15s} {'test acc':>15s} {'prec@R':>15s}"
              f" | {'best of r by held-out CE: hold CE, test acc, prec@R':>52s}")
        for r in range(1, R + 1):
            if math.comb(R, r) <= a.curve:
                subsets = list(itertools.combinations(range(R), r))
            else:
                subsets = set()
                while len(subsets) < a.curve:
                    subsets.add(tuple(sorted(rng.choice(R, r, replace=False).tolist())))
                subsets = sorted(subsets)
            res, picked = [], []
            for sub in subsets:
                chosen = [pooled[i] for i in sub]
                v = sum(1 - g for _, g, _, _ in chosen)
                mr = np.mean([rk for _, _, rk, _ in chosen], axis=0)
                res.append(score(off_lowest((-v, mr))) if r > 1 else chosen[0][3])
                picked.append(min((c[3] for c in chosen), key=lambda x: (x[1], -x[2])))
            res, picked = np.array(res), np.array(picked)
            print(f"    {r:>2d} {len(res):>7d} " + " ".join(f"{res[:, j].mean():>8.4f} ({res[:, j].std():.3f})"
                                                       for j in (0, 1, 3, 4))
                  + " | " + " ".join(f"{picked[:, j].mean():>14.4f}" for j in (1, 3, 4)))


if __name__ == "__main__":
    main()

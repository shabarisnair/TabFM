#!/usr/bin/env python
"""Are gradient-ranked swaps good moves for a hard top-k mask? (top-k gating diagnostic)

At a hard mask x (k rows switched off), the straight-through gradient d CE/dx on the fitted
validation rows estimates, to first order, the effect of each single flip. A swap of b
pairs takes the b kept rows whose masking should lower the CE most (largest dCE/dx) and the
b masked rows whose return should lower it most (most negative dCE/dx). For b = 1, 2, 5,
10, 20, 50 this prints the exact change in fitted-row CE (and held-out CE), the first-order
prediction, and the same for random swaps of b pairs (mean of 3), at two starting masks:
the soft gates' lowest 20% and a random mask. --alphas ranks the swaps instead by the
gradient at a relaxed probe point (kept rows at 1 - alpha, masked rows at alpha), the
ReinMax-style idea of not differentiating at the discrete point itself.

    python scripts/diagnose_topk_swaps.py --gpu 1 --dataset url_unique
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402
from diagnose_topk_masks import CONTEXT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dataset", required=True, choices=list(CONTEXT))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0],
                    help="probe points: gradient with kept rows at 1 - alpha, masked rows at alpha")
    a = ap.parse_args(argv)
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = f"cuda:{a.gpu}"
    run = next(Path("results/gate_topk", a.dataset, "hub-flip-local_r020").glob("*_s-c_it300"))
    sp, pz = np.load(run / "split.npz"), np.load(run / "poison.npz")
    X_all, y_all = G.load(Path(CONTEXT[a.dataset]), device)
    sel = lambda r: X_all[torch.as_tensor(r, device=device)]  # noqa: E731
    Xc, Xf, Xh = sel(sp["context_rows"]), sel(sp["val_fit_rows"]), sel(sp["val_holdout_rows"])
    yf = torch.as_tensor(y_all[sp["val_fit_rows"]], device=device)
    yh = torch.as_tensor(y_all[sp["val_holdout_rows"]], device=device)
    n, K = len(sp["context_rows"]), len(np.unique(y_all))
    yp = torch.tensor(pz["y_poisoned"], dtype=torch.float32, device=device)
    pois = np.zeros(n, bool)
    pois[pz["poisoned_rows"]] = True
    k = int(pois.sum())
    clf = build_tabpfn_v2(device, 0, recompute_layers=True, n_classes=K)
    null = G.capture_null_input(clf, Xc, yp)

    def ce(x, Xq, yq):
        return G.evaluate(clf, Xc, yp, Xq, yq, torch.as_tensor(x, dtype=torch.float32, device=device),
                          null, need_grad=False, batch_size=None)[0]

    soft_dir = next(Path("results/gate_valsplit", a.dataset, "hub-flip-local_r020").glob("*_V500s0_H20_*"))
    g_soft = np.load(soft_dir / "gates" / "top1.npz")["g"]
    starts = {}
    m = np.ones(n)
    m[np.argsort(g_soft + 1e-6 * np.random.default_rng(1).random(n), kind="stable")[:k]] = 0
    starts["soft gates' lowest 20%"] = m
    m = np.ones(n)
    m[np.random.default_rng(2).permutation(n)[:k]] = 0
    starts["random mask"] = m
    rng = np.random.default_rng(3)
    for (name, x), alpha in [(st, al) for st in starts.items() for al in a.alphas]:
        # gradient at the probe point: kept rows at 1 - alpha, masked rows at alpha (0 = the hard mask)
        probe = np.where(x == 1, 1 - alpha, alpha)
        xt = torch.as_tensor(probe, dtype=torch.float32, device=device).requires_grad_(True)
        _, grad, _ = G.evaluate(clf, Xc, yp, Xf, yf, xt, null, need_grad=True, batch_size=None)
        grad = grad.cpu().numpy()
        c0 = ce(x, Xf, yf)
        h0 = ce(x, Xh, yh)
        kept, off = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
        out_rank = kept[np.argsort(-grad[kept])]   # masking these should help most
        in_rank = off[np.argsort(grad[off])]       # bringing these back should help most
        print(f"\n{a.dataset}, start = {name}, gradient at alpha = {alpha:g}: fitted CE {c0:.4f}, held-out CE {h0:.4f}, "
              f"precision {pois[x == 0].mean():.3f}")
        print(f"  {'pairs':>5s} | {'gradient-ranked: dCE fitted (predicted)':>40s} {'dCE held-out':>12s} "
              f"{'poisoned out/in':>15s} | {'random swaps: dCE fitted':>24s}")
        for b in (1, 2, 5, 10, 20, 50):
            xs = x.copy()
            xs[out_rank[:b]] = 0
            xs[in_rank[:b]] = 1
            pred = -grad[out_rank[:b]].sum() + grad[in_rank[:b]].sum()
            d_fit, d_hold = ce(xs, Xf, yf) - c0, ce(xs, Xh, yh) - h0
            rnd = []
            for _ in range(3):
                xr = x.copy()
                xr[rng.choice(kept, b, replace=False)] = 0
                xr[rng.choice(off, b, replace=False)] = 1
                rnd.append(ce(xr, Xf, yf) - c0)
            print(f"  {b:>5d} | {d_fit:>+20.4f} ({pred:>+16.4f}) {d_hold:>+12.4f} "
                  f"{int(pois[out_rank[:b]].sum()):>7d}/{int(pois[in_rank[:b]].sum()):<7d} | {np.mean(rnd):>+24.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Does the gating objective prefer the oracle mask? (top-k gating diagnostic)

For one dataset at R = 20%, V = 500 (seed 0) with 100 held out, everything rebuilt from a
finished top-k run (its split, poison and context), compare the training objective (CE on
the 400 fitted validation rows) and the held-out CE of several hard masks that each switch
off exactly k rows, plus two soft references:

  oracle           exactly the poisoned rows switched off
  random           k random rows (mean of 5)
  gradient start   the k rows with the largest dCE/dg at the full context (SNIP-like)
  soft gates, hard the k lowest gates of the soft-gate run on the same draw (gate_valsplit)
  best top-k       the best top-k mask found so far (lowest held-out CE among the runs)
  soft gates       the soft gates themselves (continuous)
  no gate          nothing switched off

If the oracle's fitted CE is far below the masks the optimiser finds, the search is the
bottleneck; if it is about the same, the fitted rows cannot single out the poison.

    python scripts/diagnose_topk_masks.py --gpu 1 --dataset url_unique
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402

CONTEXT = {"gas-drift": "datasets/gas-drift/splits/context_5000.csv",
           "url_unique": "datasets/url_unique/splits/selected/natural/context_5000.csv"}
TEST = {"gas-drift": "datasets/gas-drift/splits/test_attack_1000.csv",
        "url_unique": "datasets/url_unique/splits/test_attack_1000.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dataset", required=True, choices=list(CONTEXT))
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = f"cuda:{a.gpu}"
    root = Path("results/gate_topk") / a.dataset / "hub-flip-local_r020"
    runs = [d for d in root.glob("gate-topk_*") if (d / "test_metrics.json").exists()]
    best = min(runs, key=lambda d: json.load(open(d / "test_metrics.json"))["best_val_row"]["valtest_ce"])
    sp, pz = np.load(best / "split.npz"), np.load(best / "poison.npz")
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

    def scores(g):
        gt = torch.as_tensor(g, dtype=torch.float32, device=device)
        ce_f = G.evaluate(clf, Xc, yp, Xf, yf, gt, null, need_grad=False, batch_size=None)[0]
        ce_h = G.evaluate(clf, Xc, yp, Xh, yh, gt, null, need_grad=False, batch_size=None)[0]
        p = G.evaluate(clf, Xc, yp, Xt, yt_t, gt, null, need_grad=False, batch_size=None)[2]
        return ce_f, ce_h, multiclass_metrics(p, yt, K)["accuracy"]

    def mask_lowest(v):
        m = np.ones(n)
        m[np.argsort(v, kind="stable")[:k]] = 0
        return m

    masks = {"oracle": (~pois).astype(float), "no gate": np.ones(n)}
    ones = torch.ones(n, device=device, requires_grad=True)
    _, gfull, _ = G.evaluate(clf, Xc, yp, Xf, yf, ones, null, need_grad=True, batch_size=None)
    masks["gradient start"] = mask_lowest(-gfull.cpu().numpy())
    soft_dir = next(Path("results/gate_valsplit", a.dataset, "hub-flip-local_r020").glob("*_V500s0_H20_*"))
    g_soft = np.load(soft_dir / "gates" / "top1.npz")["g"]
    masks["soft gates, hard"] = mask_lowest(g_soft)
    masks["soft gates"] = g_soft
    masks["best top-k"] = np.load(best / "gates" / "top1.npz")["g"]
    rng = np.random.default_rng(0)
    rand = [scores(mask_lowest(rng.random(n))) for _ in range(5)]

    print(f"{a.dataset}: n = {n} context rows, k = {k} switched off; best top-k run {best.name}")
    print(f"  {'mask':<18s} {'fitted CE':>10s} {'held-out CE':>12s} {'test acc':>9s} {'precision':>10s}")
    for name, m in masks.items():
        ce_f, ce_h, acc = scores(m)
        prec = pois[m == 0].mean() if (m == 0).any() and name not in ("soft gates",) else float("nan")
        print(f"  {name:<18s} {ce_f:>10.4f} {ce_h:>12.4f} {acc:>9.3f} {prec:>10.3f}")
    r = np.array(rand)
    print(f"  {'random (5)':<18s} {r[:, 0].mean():>10.4f} {r[:, 1].mean():>12.4f} {r[:, 2].mean():>9.3f} {0.2:>10.3f}")


if __name__ == "__main__":
    main()

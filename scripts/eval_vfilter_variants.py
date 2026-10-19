#!/usr/bin/env python
"""Variants of the V-filter: using the clean rows to judge the poisoned context's labels.

For each soft-gate draw of results/gate_valsplit (dataset, R, V, validation seed; the same
validation rows and poison), with plain TabPFN and no training. The base score of context
row i is s_i = p(given label of i | the V clean rows as the context).

  frac <f>        remove the f share of the context with the lowest s (f = 0.1, 0.2, 0.3;
                  0.2 = R, the rate the poison was made at, assumed known by the others)
  threshold       remove every row with s < 0.5 (no assumed poison rate)
  iter<t>         iterative, cross-fitted: after the f = R removal, the context is split
                  in two random halves A and B; A's rows are re-scored with the V clean rows
                  plus B's kept rows as the context, and B's with the V rows plus A's kept
                  rows (no row is ever scored by a context that contains it), and the R share
                  with the lowest new scores is removed; repeated, t = 2, 3 rounds in total
For each: test accuracy of the kept rows as the context, and of the kept rows plus the V clean
rows; precision of the removed rows (and how many were removed).

Appends one CSV row per draw to --out.

    python scripts/eval_vfilter_variants.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --out results/tables/gate_soft/vfilter_variants_R20.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402

VARIANTS = ("frac0.1", "frac0.2", "frac0.3", "threshold", "iter2", "iter3")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 300, 400, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--soft-root", type=Path, default=Path("results/gate_valsplit"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    fields = ["dataset", "R", "V", "seed", "k", "n_context", "v_only"] + [
        f"{v}_{m}" for v in VARIANTS for m in ("acc", "acc_plus_v", "prec", "n_removed")]
    done = set()
    if a.out.exists():
        done = {(r["dataset"], int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    clfs = {}
    for ds in a.datasets:
        for V in a.sizes:
            for s in a.seeds:
                if (ds, V, s) in done:
                    continue
                hits = sorted(a.soft_root.glob(f"{ds}/*_r{a.R:03d}/gate-valsplit_{ds}_*_R{a.R}_V{V}s{s}_H20_pgd-linf_lr0.05_it300"))
                if not hits or not (hits[0] / "test_metrics.json").exists():
                    continue
                d = hits[0]
                cfg = json.loads((d / "config.json").read_text())
                sp, pz = np.load(d / "split.npz"), np.load(d / "poison.npz")
                K, k = int(cfg["n_classes"]), int(cfg["k_remove"])
                X_all, y_all = G.load(Path(cfg["context"]), device)
                Xt, yt = G.load(Path(cfg["test"]), device)
                yt_t = torch.as_tensor(yt, device=device)
                idx = lambda r: torch.as_tensor(np.asarray(r), device=device)  # noqa: E731
                Xc, Xv = X_all[idx(sp["context_rows"])], X_all[idx(sp["val_rows"])]
                yv = torch.tensor(y_all[sp["val_rows"]], dtype=torch.float32, device=device)
                n = len(sp["context_rows"])
                yg = pz["y_poisoned"].astype(int)
                yp = torch.tensor(yg, dtype=torch.float32, device=device)
                pois = np.zeros(n, dtype=bool)
                pois[pz["poisoned_rows"]] = True
                if K not in clfs:
                    clfs[K] = build_tabpfn_v2(device, cfg["args"]["model_seed"], recompute_layers=True, n_classes=K)
                clf = clfs[K]

                def plaus(X_ctx, y_ctx, rows):  # p(given label) of context rows `rows`
                    p = G.evaluate(clf, X_ctx, y_ctx, Xc[idx(rows)], idx(yg[rows]), None, None,
                                   need_grad=False, batch_size=None)[2]
                    return p[np.arange(len(rows)), yg[rows]]

                def acc(keep, with_v):
                    kk = idx(np.flatnonzero(keep))
                    X_ctx, y_ctx = (torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv])) if with_v else (Xc[kk], yp[kk])
                    p = G.evaluate(clf, X_ctx, y_ctx, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
                    return float(multiclass_metrics(p, yt, K)["accuracy"])

                row = {"dataset": ds, "R": a.R, "V": V, "seed": s, "k": k, "n_context": n}
                row["v_only"] = float(multiclass_metrics(G.evaluate(clf, Xv, yv, Xt, yt_t, None, None, need_grad=False,
                                                                    batch_size=None)[2], yt, K)["accuracy"])
                s0 = plaus(Xv, yv, np.arange(n))
                keeps = {}
                for f in (0.1, 0.2, 0.3):
                    keep = np.ones(n, bool)
                    keep[np.argsort(s0, kind="stable")[:int(round(f * n))]] = False
                    keeps[f"frac{f:g}"] = keep
                keeps["threshold"] = s0 >= 0.5
                keep = keeps["frac0.2"].copy()
                rng = np.random.default_rng(s)
                for t in (2, 3):
                    half = np.zeros(n, bool)
                    half[rng.permutation(n)[:n // 2]] = True
                    score = np.empty(n)
                    for side in (half, ~half):  # score one half with the V rows + the other half's kept rows
                        other = np.flatnonzero(~side & keep)
                        X_ctx = torch.cat([Xv, Xc[idx(other)]])
                        y_ctx = torch.cat([yv, yp[idx(other)]])
                        score[side] = plaus(X_ctx, y_ctx, np.flatnonzero(side))
                    keep = np.ones(n, bool)
                    keep[np.argsort(score, kind="stable")[:k]] = False
                    keeps[f"iter{t}"] = keep.copy()
                for name in VARIANTS:
                    keep = keeps[name]
                    row[f"{name}_acc"] = acc(keep, False)
                    row[f"{name}_acc_plus_v"] = acc(keep, True)
                    row[f"{name}_prec"] = float(pois[~keep].mean()) if (~keep).any() else float("nan")
                    row[f"{name}_n_removed"] = int((~keep).sum())
                w.writerow(row)
                fh.flush()
                print(f"{ds} V{V} s{s}: V only {100 * row['v_only']:.1f} | " + " | ".join(
                    f"{v} {100 * row[v + '_acc']:.1f}/{100 * row[v + '_acc_plus_v']:.1f} p{row[v + '_prec']:.2f}"
                    for v in VARIANTS), flush=True)
    fh.close()


if __name__ == "__main__":
    main()

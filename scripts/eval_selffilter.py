#!/usr/bin/env python
"""Can TabPFN clean the poisoned context without any trusted rows? The self-filter.

For each soft-gate draw of results/gate_valsplit (dataset, R, V, validation seed; its
context and poison), with plain TabPFN and no training: the poisoned context is split into
--folds random folds; every row's given label is scored by TabPFN with the OTHER folds as
the context, p(given label | the rest of the poisoned context); the R% least plausible rows
are removed. Reported: its precision and the test accuracy of the kept rows, alone and plus
the V clean rows; next to the poisoned context, the V clean rows alone and the V-filter (the
same removal with the V clean rows as the scoring context).

    python scripts/eval_selffilter.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --out results/tables/gate_soft/selffilter_R20.csv
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

FIELDS = ["dataset", "R", "V", "seed", "k", "poisoned", "v_only", "self", "self_plus_v", "prec_self",
          "vfilter", "vfilter_plus_v", "prec_vfilter"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    done = set()
    if a.out.exists():
        done = {(r["dataset"], int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    clfs = {}
    for ds in a.datasets:
        for V in a.sizes:
            for s in a.seeds:
                if (ds, V, s) in done:
                    continue
                hits = sorted(Path("results/gate_valsplit").glob(f"{ds}/*_r{a.R:03d}/gate-valsplit_{ds}_*_R{a.R}_V{V}s{s}_H20_pgd-linf_lr0.05_it300"))
                if not hits:
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

                def plaus(X_ctx, y_ctx, rows):
                    p = G.evaluate(clf, X_ctx, y_ctx, Xc[idx(rows)], idx(yg[rows]), None, None,
                                   need_grad=False, batch_size=None)[2]
                    return p[np.arange(len(rows)), yg[rows]]

                def acc(X_ctx, y_ctx):
                    p = G.evaluate(clf, X_ctx, y_ctx, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
                    return float(multiclass_metrics(p, yt, K)["accuracy"])

                fold = np.random.default_rng(s).permutation(n) % a.folds
                score_self = np.empty(n)
                for f in range(a.folds):
                    rest = np.flatnonzero(fold != f)
                    score_self[fold == f] = plaus(Xc[idx(rest)], yp[idx(rest)], np.flatnonzero(fold == f))
                score_v = plaus(Xv, yv, np.arange(n))
                row = {"dataset": ds, "R": a.R, "V": V, "seed": s, "k": k,
                       "poisoned": acc(Xc, yp), "v_only": acc(Xv, yv)}
                for name, score in (("self", score_self), ("vfilter", score_v)):
                    drop = np.argsort(score, kind="stable")[:k]
                    keep = np.ones(n, dtype=bool)
                    keep[drop] = False
                    kk = idx(np.flatnonzero(keep))
                    row[name] = acc(Xc[kk], yp[kk])
                    row[f"{name}_plus_v"] = acc(torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv]))
                    row[f"prec_{name}"] = float(pois[drop].mean())
                w.writerow(row)
                fh.flush()
                print(f"{ds} V{V} s{s}: poisoned {100 * row['poisoned']:.1f} | V only {100 * row['v_only']:.1f} | "
                      f"self-filter {100 * row['self']:.1f} +V {100 * row['self_plus_v']:.1f} (prec {row['prec_self']:.2f}) | "
                      f"V-filter {100 * row['vfilter']:.1f} +V {100 * row['vfilter_plus_v']:.1f} (prec {row['prec_vfilter']:.2f})",
                      flush=True)
    fh.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""A coverage-aware V-filter: judge only the labels the clean rows can see.

The V-filter scores every context row's given label by TabPFN with the V clean rows as its
context and removes the R% least plausible. When the clean rows cover only part of the feature
space (--val-draw local), it must also judge rows far from any clean row, and it removes many
genuine ones there. Here, for the same draws as eval_vfilter_vforms.py (gate_context.
draw_validation_form; hub-flip-local, R%, run 0):

  vfilter        the R% least plausible rows of the whole context removed
  near<q>        only the share q of the context rows nearest to a clean row (standardised
                 features, distance to the nearest clean row) can be removed: the R% least
                 plausible among them (or all of them if fewer); the far rows are kept
  near<q>-thr    among the same near rows, every row whose given label has p < 0.5

Test accuracy of the kept rows plus the V clean rows, and precision of the removed rows.

    python scripts/eval_vfilter_coverage.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --out results/tables/gate_soft/vfilter_coverage_R20.csv
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
from eval_vfilter_attacks import CONTEXT, TEST  # noqa: E402

VARIANTS = ("vfilter", "near0.5", "near0.75", "near0.5-thr")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--forms", nargs="+", default=["local", "stratified"])
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.multiclass_attacks import poison, standardise
    from tabfm_experiments.runtime import build_tabpfn_v2
    from tabfm_experiments.sampling import k_from_percent

    device = G.device_from_gpu(a.gpu)
    fields = ["dataset", "form", "V", "seed", "v_only"] + [f"{v}_{m}" for v in VARIANTS for m in ("acc", "prec", "n")]
    done = set()
    if a.out.exists():
        done = {(r["dataset"], r["form"], int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    for ds in a.datasets:
        X_file, y_file = G.load(Path(CONTEXT[ds]), device)
        Xt, yt = G.load(Path(TEST[ds]), device)
        yt_t = torch.as_tensor(yt, device=device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
        clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K)
        Z_file = standardise(X_file)
        for form in a.forms:
            for V in a.sizes:
                for s in a.seeds:
                    if (ds, form, V, s) in done:
                        continue
                    val_rows, _ = G.draw_validation_form(X_file, y_file, V, s, form)
                    ctx_rows = np.setdiff1d(np.arange(len(y_file)), val_rows)
                    Xv, yv = X_file[torch.as_tensor(val_rows, device=device)], y_file[val_rows]
                    Xc, yc = X_file[torch.as_tensor(ctx_rows, device=device)], y_file[ctx_rows]
                    yv_f = torch.tensor(yv, dtype=torch.float32, device=device)
                    n = len(ctx_rows)
                    k = k_from_percent(n, a.R)
                    rows, y_pois = poison("hub-flip-local", Xc, yc, K, a.R, run_id=0, row_seed=Seeds.row_seed)
                    pois = np.zeros(n, dtype=bool)
                    pois[rows] = True
                    yp = torch.tensor(y_pois, dtype=torch.float32, device=device)
                    pc = G.evaluate(clf, Xv, yv_f, Xc, torch.as_tensor(y_pois, device=device), None, None,
                                    need_grad=False, batch_size=None)[2]
                    score = pc[np.arange(n), y_pois]
                    dmin = torch.cdist(Z_file[torch.as_tensor(ctx_rows, device=device)],
                                       Z_file[torch.as_tensor(val_rows, device=device)]).min(1).values.cpu().numpy()
                    drops = {"vfilter": np.argsort(score, kind="stable")[:k]}
                    for q in (0.5, 0.75):
                        near = np.flatnonzero(dmin <= np.quantile(dmin, q))
                        drops[f"near{q:g}"] = near[np.argsort(score[near], kind="stable")[:k]]
                    near = np.flatnonzero(dmin <= np.quantile(dmin, 0.5))
                    drops["near0.5-thr"] = near[score[near] < 0.5]
                    row = {"dataset": ds, "form": form, "V": V, "seed": s}
                    p = G.evaluate(clf, Xv, yv_f, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
                    row["v_only"] = float(multiclass_metrics(p, yt, K)["accuracy"])
                    for name, drop in drops.items():
                        keep = np.ones(n, dtype=bool)
                        keep[drop] = False
                        kk = torch.as_tensor(np.flatnonzero(keep), device=device)
                        p = G.evaluate(clf, torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv_f]), Xt, yt_t, None, None,
                                       need_grad=False, batch_size=None)[2]
                        row[f"{name}_acc"] = float(multiclass_metrics(p, yt, K)["accuracy"])
                        row[f"{name}_prec"] = float(pois[drop].mean()) if len(drop) else float("nan")
                        row[f"{name}_n"] = int(len(drop))
                    w.writerow(row)
                    fh.flush()
                    print(f"{ds} {form} V{V} s{s}: V only {100 * row['v_only']:.1f} | " + " | ".join(
                        f"{v} {100 * row[v + '_acc']:.1f} (p{row[v + '_prec']:.2f}, n{row[v + '_n']})" for v in VARIANTS), flush=True)
    fh.close()


if __name__ == "__main__":
    main()

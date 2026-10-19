#!/usr/bin/env python
"""Does the form of the clean set change who wins? V only vs the V-filter, per form of V.

The clean set is drawn from the context file by gate_context.draw_validation_form (as
gate_context.py --val-draw does), in one of these forms (every class present, so that
TabPFN sees every class in any context):
  stratified   class-stratified random rows (gate_context.draw_validation)
  local        for each class, its rows nearest to one random anchor row (standardised
               features), as many as the class's stratified share: a clean set covering one
               region of the feature space, as when trusted labels come from one source
  imbalanced   class-stratified, except that one class (--minor-class, default the rarest)
               gets only --minor-n rows
The rest of the file is the context, poisoned with --attack at R%. No training. Test accuracy
of the poisoned context, the V clean rows alone, the V-filter (the R% least plausible labels
under TabPFN with the V rows as its context removed) with and without the V rows, the oracle
with and without the V rows; the V-filter's precision; and the same accuracies on the test rows
of the minor class (imbalanced) or far from the anchor (local: the half of the test rows
farthest from it).

    python scripts/eval_vfilter_vforms.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --out results/tables/gate_soft/vforms_R20.csv
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

FORMS = ("stratified", "local", "imbalanced")
METHODS = ("poisoned", "v_only", "vfilter", "vfilter_plus_v", "oracle", "oracle_plus_v")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--forms", nargs="+", default=list(FORMS), choices=FORMS)
    ap.add_argument("--attack", default="hub-flip-local")
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--minor-n", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.multiclass_attacks import poison, standardise
    from tabfm_experiments.runtime import build_tabpfn_v2
    from tabfm_experiments.sampling import k_from_percent

    device = G.device_from_gpu(a.gpu)
    fields = (["dataset", "form", "attack", "R", "V", "seed", "minor_class", "prec_vfilter"]
              + list(METHODS) + [f"sub_{m}" for m in METHODS])
    done = set()
    if a.out.exists():
        done = {(r["dataset"], r["form"], r["attack"], int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
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
        minor_class = int(np.argmin(np.bincount(y_file, minlength=K)))
        clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K)
        Zfile_t = standardise(torch.cat([X_file, Xt]))  # one scaling for the anchor distances
        for form in a.forms:
            for V in a.sizes:
                for s in a.seeds:
                    if (ds, form, a.attack, V, s) in done:
                        continue
                    val_rows, anchor = G.draw_validation_form(X_file, y_file, V, s, form, minor_class, a.minor_n)
                    ctx_rows = np.setdiff1d(np.arange(len(y_file)), val_rows)
                    Xv, yv = X_file[torch.as_tensor(val_rows, device=device)], y_file[val_rows]
                    Xc, yc = X_file[torch.as_tensor(ctx_rows, device=device)], y_file[ctx_rows]
                    if len(np.unique(yv)) < K or len(np.unique(yc)) < K:
                        print(f"skip {ds} {form} V{V} s{s}: a class is missing", flush=True)
                        continue
                    yv_f = torch.tensor(yv, dtype=torch.float32, device=device)
                    n = len(ctx_rows)
                    k = k_from_percent(n, a.R)
                    rows, y_pois = poison(a.attack, Xc, yc, K, a.R, run_id=0, row_seed=Seeds.row_seed)
                    pois = np.zeros(n, dtype=bool)
                    pois[rows] = True
                    yp = torch.tensor(y_pois, dtype=torch.float32, device=device)
                    pc = G.evaluate(clf, Xv, yv_f, Xc, torch.as_tensor(y_pois, device=device), None, None,
                                    need_grad=False, batch_size=None)[2]
                    drop = np.argsort(pc[np.arange(n), y_pois], kind="stable")[:k]
                    keep = np.ones(n, dtype=bool)
                    keep[drop] = False
                    if form == "local":  # the half of the test rows farthest from the anchor
                        dt = torch.cdist(Zfile_t[anchor:anchor + 1], Zfile_t[len(y_file):])[0].cpu().numpy()
                        sub = dt >= np.median(dt)
                    elif form == "imbalanced":
                        sub = yt == minor_class
                    else:
                        sub = np.ones(len(yt), dtype=bool)
                    ctxs = {"poisoned": (Xc, yp), "v_only": (Xv, yv_f)}
                    kk = torch.as_tensor(np.flatnonzero(keep), device=device)
                    ko = torch.as_tensor(np.flatnonzero(~pois), device=device)
                    ctxs["vfilter"] = (Xc[kk], yp[kk])
                    ctxs["vfilter_plus_v"] = (torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv_f]))
                    ctxs["oracle"] = (Xc[ko], yp[ko])
                    ctxs["oracle_plus_v"] = (torch.cat([Xc[ko], Xv]), torch.cat([yp[ko], yv_f]))
                    row = {"dataset": ds, "form": form, "attack": a.attack, "R": a.R, "V": V, "seed": s,
                           "minor_class": minor_class if form == "imbalanced" else "",
                           "prec_vfilter": float(pois[drop].mean())}
                    for m, (Xx, yy) in ctxs.items():
                        p = G.evaluate(clf, Xx, yy, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
                        row[m] = float(multiclass_metrics(p, yt, K)["accuracy"])
                        row[f"sub_{m}"] = float((p.argmax(1) == yt)[sub].mean())
                    w.writerow(row)
                    fh.flush()
                    print(f"{ds} {form} V{V} s{s}: " + " | ".join(f"{m} {100 * row[m]:.1f} ({100 * row['sub_' + m]:.1f})"
                                                                for m in METHODS)
                          + f" | prec {row['prec_vfilter']:.2f}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()

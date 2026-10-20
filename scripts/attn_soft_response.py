#!/usr/bin/env python
"""How TabPFN responds to soft context-row weights (the row-weight mechanism of tabfm_experiments/row_weights.py),
on the larger-V benchmark's V = 500, R = 30% draws (seed 0).

  sweep     one log w for (a) every row given the smallest class, (b) a random 30% of the rows (seed 0), the rest
            at 1: held-out rows (the poisoned rows only as the context) and test rows (poisoned rows + V) -> log loss,
            accuracy, ROC-AUC and the mean predicted probability of that class, per log w
  channels  the V-filter start of the soft learners (log w = s - max s, s = log p(given label | the 400 fitted V
            rows)) installed on all channels, or on the attention / the label mean / the statistics alone

    python scripts/attn_soft_response.py sweep --device cuda:2 [--datasets ...]     -> results/tables/attn_form/weight_sweep.csv
    python scripts/attn_soft_response.py channels --device cuda:2                   -> results/tables/attn_form/vfilter_channels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

LEVELS = (0.0, -0.25, -0.5, -1.0, -1.6, -2.3, -3.0, -4.6, -6.9, -9.2, -20.0, -100.0, -1000.0, -float("inf"))
OUT = Path("results/tables/attn_form")


def instance(ds: str, dev: str):
    import bench_largev as BL
    data = BL.Data(ds, dev)
    return data, BL.load_draw(data, BL.tag_of(ds, 30.0, 500, 0))


def sweep(dev: str, datasets: list[str]) -> list[dict]:
    import attn_learn as A
    from tabfm_experiments.row_weights import evaluate_weighted
    rows = []
    for ds in datasets:
        data, d = instance(ds, dev)
        n, K = d["n"], d["K"]
        zv = torch.zeros(len(d["yv"]), device=dev)
        Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([d["yp"], d["yv"]])
        cl = int(np.argmin(np.bincount(d["yt"], minlength=K)))
        rnd = np.zeros(n, bool)
        rnd[np.random.default_rng(0).choice(n, int(0.3 * n), replace=False)] = True
        for what, mask in (("class", d["yp"] == cl), ("random30", rnd)):
            m = torch.as_tensor(mask, device=dev)
            for lv in LEVELS:
                lw = torch.where(m, torch.full((n,), lv, device=dev), torch.zeros(n, device=dev))
                ph = evaluate_weighted(data.clf, d["Xc"], d["yp"], d["Xh"], d["yh"], lw, need_grad=False)[2]
                pt = evaluate_weighted(data.clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw, zv]), need_grad=False)[2]
                mh, mt = A.metrics(ph, d["yh"]), A.metrics(pt, d["yt"])
                rows.append(dict(dataset=ds, what=what, cls=cl, n_rows=int(mask.sum()), logw=lv, hold_ce=mh["ce"],
                                 hold_acc=mh["acc"], hold_p_cls=float(ph[:, cl].mean()), test_v_ce=mt["ce"],
                                 test_v_acc=mt["acc"], test_v_auc=mt["auc"], test_v_p_cls=float(pt[:, cl].mean()),
                                 true_share_cls=float((d["yt"] == cl).mean())))
        del data, d
        torch.cuda.empty_cache()
    return rows


def channels(dev: str, datasets: list[str]) -> list[dict]:
    import attn_learn as A
    from tabfm_experiments.row_weights import (evaluate_weighted, model_weights_installed,
                                               preprocessing_weights_installed)
    from tabfm_experiments.runtime import _forward, _install_recompute

    def probs(clf, Xc, yc, Xq, la, ll, ls):
        with torch.no_grad(), preprocessing_weights_installed(ls):
            clf.fit_with_differentiable_input(Xc, torch.as_tensor(yc, dtype=torch.float32, device=dev))
            _install_recompute(clf)
            with model_weights_installed(clf, la, label_log_w=ll, stats_log_w=ls):
                return _forward(clf, Xq, need_grad=False).float().cpu().numpy()

    rows = []
    for ds in datasets:
        data, d = instance(ds, dev)
        n, K = d["n"], d["K"]
        p = evaluate_weighted(data.clf, d["Xf"], d["yf"], d["Xc"], d["yp"], torch.zeros(len(d["yf"]), device=dev),
                              need_grad=False)[2]
        s = np.log(np.clip(p[np.arange(n), d["yp"]], 1e-12, None))
        b = torch.as_tensor(s - s.max(), dtype=torch.float32, device=dev)
        z, zv = torch.zeros(n, device=dev), torch.zeros(len(d["yv"]), device=dev)
        Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([d["yp"], d["yv"]])
        w = np.exp(s - s.max())
        mix = {f"w_share_class{c}": float(w[d["yp"] == c].sum() / w.sum()) for c in range(K)}
        for name, (la, ll, ls) in {"none": (z, z, z), "all": (b, b, b), "attention": (b, z, z), "label_mean": (z, b, z),
                                   "statistics": (z, z, b)}.items():
            ph = probs(data.clf, d["Xc"], d["yp"], d["Xh"], la, ll, ls)
            pt = probs(data.clf, Xcv, ycv, d["Xt"], torch.cat([la, zv]), torch.cat([ll, zv]), torch.cat([ls, zv]))
            mh, mt = A.metrics(ph, d["yh"]), A.metrics(pt, d["yt"])
            rows.append(dict(dataset=ds, channels=name, hold_ce=mh["ce"], hold_acc=mh["acc"], test_v_ce=mt["ce"],
                             test_v_acc=mt["acc"], test_v_auc=mt["auc"],
                             **{f"hold_pred_share_class{c}": float((ph.argmax(1) == c).mean()) for c in range(K)}, **mix))
        del data, d
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("sweep", "channels"))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args(argv)
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "sweep":
        ds = a.datasets or ["is_this_a_good_customer", "students_dropout", "heloc", "coil2000_insurance_policies",
                            "ecommerce_shipping"]
        path = OUT / "weight_sweep.csv"
        pd.DataFrame(sweep(a.device, ds)).to_csv(path, index=False)
    else:
        ds = a.datasets or ["is_this_a_good_customer", "students_dropout", "heloc"]
        path = OUT / "vfilter_channels.csv"
        pd.DataFrame(channels(a.device, ds)).to_csv(path, index=False)
    print("->", path)


if __name__ == "__main__":
    main()

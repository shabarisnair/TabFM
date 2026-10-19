#!/usr/bin/env python
"""Post-hoc ensembles of the learned attention-weight rankings (no training).

For every draw of results/attn_learn/confirm with the three main configurations finished, each
configuration's ranking at the iteration its held-out log loss chose (hard deployment; saved by
attn_learn.py next to its JSON) is turned into ranks; then the k = R% lowest rows are deleted under
  ens_M        the mean rank of M1, M2, M3
  ens_M_vf     the mean rank of M1, M2, M3 and the V-filter (V-filter scores with all V rows)
  ens_M1_vf    the mean rank of M1 and the V-filter
and the test set is scored with the remaining rows, and with the V clean rows added. Results go to
<draw dir>/posthoc.json, read by scripts/report_attn_confirm.py.

    python scripts/attn_posthoc.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MAIN = ("M1_topk_lr0.2", "M2_adam_prior0.1", "M3_pgd_lr1")


def main() -> None:
    import attn_learn as A
    from report_attn_confirm import config_of, select
    from scipy.stats import rankdata
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.row_weights import evaluate_weighted
    from tabfm_experiments.runtime import build_tabpfn_v2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--root", type=Path, default=Path("results/attn_learn/confirm"))
    a = ap.parse_args()
    device = f"cuda:{a.gpu}"
    clfs = {}
    for draw in sorted(p for p in a.root.iterdir() if p.is_dir()):
        out = draw / "posthoc.json"
        if out.exists():
            continue
        runs = {}
        for f in draw.glob("*.json"):
            if f.name in ("references.json", "posthoc.json"):
                continue
            r = json.loads(f.read_text())
            name = config_of(r["cfg"])
            if name in MAIN and f.with_suffix(".npz").exists():
                runs[name] = (r, np.load(f.with_suffix(".npz")))
        if len(runs) < len(MAIN):
            continue
        r0 = runs[MAIN[0]][0]["draw"]
        d = A.make_draw(r0["ds"], float(r0["R"]), int(r0["V"]), int(r0["seed"]), 0.2, device)
        if d["ds"] not in clfs:
            clfs.clear()
            clfs[d["ds"]] = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=d["K"])
        clf = clfs[d["ds"]]
        n, k = d["n"], d["k"]
        ranks = {}
        for name, (r, z) in runs.items():
            it = select(r["evals"], "hard", "ce", "")["it"]
            key = z["keys"][list(z["its"]).index(it)]
            assert np.array_equal(z["pois"], d["pois"])
            ranks[name] = rankdata(key)                  # low rank = most suspicious
        zv = torch.zeros(len(d["yv"]), device=device)
        p = evaluate_weighted(clf, d["Xv"], d["yv"], d["Xc"], d["yp"], zv, need_grad=False)[2]
        ranks["vfilter"] = rankdata(np.log(np.clip(p[np.arange(n), d["yp"]], 1e-12, None)))
        variants = {"ens_M": list(MAIN), "ens_M_vf": list(MAIN) + ["vfilter"], "ens_M1_vf": [MAIN[0], "vfilter"]}
        res = {}
        Xcv = torch.cat([d["Xc"], d["Xv"]])
        ycv = np.concatenate([d["yp"], d["yv"]])
        tie = np.random.default_rng(12345).random(n)
        for vname, members in variants.items():
            score = np.mean([ranks[m] for m in members], axis=0)
            dele = np.zeros(n, dtype=bool)
            dele[np.lexsort((tie, score))[:k]] = True
            lw = torch.where(torch.as_tensor(dele, device=device), torch.full((n,), -math.inf, device=device),
                             torch.zeros(n, device=device))
            t1 = A.metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xt"], d["yt"], lw, need_grad=False)[2], d["yt"])
            t2 = A.metrics(evaluate_weighted(clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw, zv]), need_grad=False)[2], d["yt"])
            res[vname] = {"hard": t1, "hard_v": t2, "prec": float(d["pois"][dele].mean())}
        out.write_text(json.dumps(res, indent=1))
        print(draw.name, {v: (round(x["hard_v"]["auc"], 4), round(x["hard_v"]["ce"], 4), round(x["prec"], 3)) for v, x in res.items()},
              flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Combine the V-filter's verdict with the top-k vote: is the pair more robust than either?

For each top-k vote folder (vote_topk.py; its restarts give the draw's data, split and poison):
  filter   the R% context rows with the lowest p(given label | all V clean rows as the context)
  vote     the vote mask (the k rows masked by the most restarts; ties: mean score rank)
  rankavg  the k rows with the lowest mean of the two ranks (filter: by p(given label);
           vote: by -votes, then mean score rank)
  both     the rows both remove, then the rest of the k by the mean rank
Test accuracy of the kept rows plus the V clean rows, and precision of the removed rows.

    python scripts/eval_combine_filter_vote.py --gpu 0 results/gate_soft/url_unique/hub-flip-local_r020/gate-topk-vote4_*
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


def ranks(order_key: np.ndarray) -> np.ndarray:
    r = np.empty(len(order_key))
    r[np.argsort(order_key, kind="stable")] = np.arange(len(order_key))
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("votes", type=Path, nargs="+")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    print(f"{'vote folder':78s} | {'V only':>6s} | filter / vote / rankavg / both  (test acc + V; precision)")
    for vd in a.votes:
        if not (vd / "vote.json").exists():
            continue
        v = json.loads((vd / "vote.json").read_text())
        run0 = Path(v["restart_runs"][0])
        cfg = json.loads((run0 / "config.json").read_text())
        sp, pz = np.load(run0 / "split.npz"), np.load(run0 / "poison.npz")
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
        clf = build_tabpfn_v2(device, cfg["args"]["model_seed"], recompute_layers=True, n_classes=K)
        pc = G.evaluate(clf, Xv, yv, Xc, idx(yg), None, None, need_grad=False, batch_size=None)[2]
        r_filter = ranks(pc[np.arange(n), yg])                          # 0 = least plausible
        vz = np.load(vd / "vote.npz")
        r_vote = np.lexsort((vz["mean_rank"], -vz["votes"].astype(float))).argsort().astype(float)  # 0 = most votes
        drops = {"filter": np.argsort(r_filter, kind="stable")[:k],
                 "vote": np.flatnonzero(vz["mask"] == 0),
                 "rankavg": np.argsort(r_filter + r_vote, kind="stable")[:k]}
        both = np.intersect1d(drops["filter"], drops["vote"])
        in_both = set(both.tolist())
        rest = [i for i in np.argsort(r_filter + r_vote, kind="stable") if i not in in_both][:k - len(both)]
        drops["both"] = np.concatenate([both, np.asarray(rest, dtype=int)])

        def acc_plus_v(drop):
            keep = np.ones(n, dtype=bool)
            keep[drop] = False
            kk = idx(np.flatnonzero(keep))
            p = G.evaluate(clf, torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv]), Xt, yt_t, None, None,
                           need_grad=False, batch_size=None)[2]
            return 100 * float(multiclass_metrics(p, yt, K)["accuracy"])

        p_v = G.evaluate(clf, Xv, yv, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
        txt = " / ".join(f"{acc_plus_v(dr):.1f} ({pois[dr].mean():.2f})" for dr in drops.values())
        print(f"{vd.name:78s} | {100 * float(multiclass_metrics(p_v, yt, K)['accuracy']):6.1f} | {txt}", flush=True)


if __name__ == "__main__":
    main()

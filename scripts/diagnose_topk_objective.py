#!/usr/bin/env python
"""Can the training objective single out the poison? (top-k gating diagnostic)

For each given vote folder of run_gate_topk_gen_sweep.sh (one draw: dataset, R, V, seed),
score four masks on the draw's own data: no gate, the oracle mask (exactly the poisoned rows
off), the top-k vote mask, and random masks of the same size (mean of 3). Printed: CE on
the fitted validation rows (the training objective), held-out CE, test accuracy, precision.
If the oracle lowers the fitted CE only a little, or the vote reaches a lower fitted CE than
the oracle with a low precision, the validation rows cannot tell the poison from the rest:
the objective, not the search, limits the method.

    python scripts/diagnose_topk_objective.py --gpu 1 results/gate_topk_gen/heloc/hub-flip-local_r020/gate-topk-vote4_*
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("votes", type=Path, nargs="+", help="vote folders (with vote.json and vote.npz)")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    print(f"{'draw':24s} | {'mask':8s} {'fitted CE':>9s} {'held-out CE':>11s} {'test acc':>8s} | precision")
    for vd in a.votes:
        v = json.loads((vd / "vote.json").read_text())
        run0 = Path(v["restart_runs"][0])
        cfg = json.loads((run0 / "config.json").read_text())
        sp, pz = np.load(run0 / "split.npz"), np.load(run0 / "poison.npz")
        K, k = int(cfg["n_classes"]), int(cfg["k_remove"])
        X_all, y_all = G.load(Path(cfg["context"]), device)
        Xt, yt = G.load(Path(cfg["test"]), device)
        sel = lambda r: X_all[torch.as_tensor(r, device=device)]  # noqa: E731
        Xc, Xf, Xh = sel(sp["context_rows"]), sel(sp["val_fit_rows"]), sel(sp["val_holdout_rows"])
        yf, yh = y_all[sp["val_fit_rows"]], y_all[sp["val_holdout_rows"]]
        n = len(sp["context_rows"])
        yp = torch.tensor(pz["y_poisoned"], dtype=torch.float32, device=device)
        pois = np.zeros(n, dtype=bool)
        pois[pz["poisoned_rows"]] = True
        clf = build_tabpfn_v2(device, cfg["args"]["model_seed"], recompute_layers=True, n_classes=K)
        null = G.capture_null_input(clf, Xc, yp)

        def score(m):
            gt = None if m is None else torch.as_tensor(m, dtype=torch.float32, device=device)
            ev = lambda Xq, yq: G.evaluate(clf, Xc, yp, Xq, torch.as_tensor(yq, device=device), gt, null,  # noqa: E731
                                           need_grad=False, batch_size=None)
            return ev(Xf, yf)[0], ev(Xh, yh)[0], multiclass_metrics(ev(Xt, yt)[2], yt, K)["accuracy"]

        rng = np.random.default_rng(0)
        rand = []
        for _ in range(3):
            m = np.ones(n)
            m[rng.permutation(n)[:k]] = 0
            rand.append(score(m))
        vote = np.load(vd / "vote.npz")["mask"]
        name = f"{v['dataset']} R{v['row_percent']:g} V{v['val_size']}"
        none, oracle = score(None), score((~pois).astype(float))
        for label, res, prec in (("no gate", none, "-"), ("oracle", oracle, "1.00"),
                                 ("vote", score(vote), f"{pois[vote == 0].mean():.2f}"),
                                 ("random", tuple(np.mean(rand, axis=0)), f"{pois.mean():.2f}")):
            print(f"{name:24s} | {label:8s} {res[0]:9.4f} {res[1]:11.4f} {100 * res[2]:8.1f} | {prec}")
        print(f"{'':24s}   the oracle lowers the fitted CE by {100 * (1 - oracle[0] / none[0]):.0f}% against no gate")


if __name__ == "__main__":
    main()

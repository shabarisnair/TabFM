#!/usr/bin/env python
"""Use the clean rows AND the poisoned context: test accuracy of combined contexts.

For each soft-gate run of results/gate_valsplit (one draw: dataset, R, V, validation seed,
20% held out), on the draw's own data, with plain TabPFN (no training):

  V only            the V clean rows alone as the context (= val_as_context, re-scored)
  poisoned + V      the poisoned context and the V clean rows
  soft              the soft gates at the run's chosen iteration (gates/top1.npz)
  soft + V          the soft-gated poisoned context and the V clean rows (gate 1)
  V-filter          TabPFN with the V clean rows as its context scores every context row's
                    given (possibly flipped) label; the k least plausible rows (lowest
                    p(given label)) are removed, k = R% of the context. Uses the clean rows
                    directly, as a filter; no optimisation, no iteration to choose
  V-filter + V      the filtered context and the V clean rows
  oracle + V        the poisoned rows removed exactly, plus the V clean rows (upper bound)
  top-k + V         with --topk-root: the top-k vote mask's kept rows plus the V clean rows
                    (when that draw has a vote)
and the precision@R of the soft gates' k lowest, of the V-filter and of the top-k vote.

Writes one CSV row per draw to --out (appending; a draw already in the file is skipped).

    python scripts/eval_context_combos.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --R 20 --out results/tables/gate_soft/combos_R20.csv --topk-root results/gate_topk_gen
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

FIELDS = ["dataset", "R", "V", "seed", "k", "n_context", "v_only", "poisoned", "poisoned_plus_v", "soft",
          "soft_plus_v", "vfilter", "vfilter_plus_v", "oracle", "oracle_plus_v", "topk", "topk_plus_v",
          "prec_soft", "prec_vfilter", "prec_topk", "soft_iter"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 300, 400, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--soft-root", type=Path, default=Path("results/gate_valsplit"))
    ap.add_argument("--soft-lr", default="0.05", help="the soft-gate runs' lr (run-name field)")
    ap.add_argument("--topk-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    done = set()
    if a.out.exists():
        done = {(r["dataset"], int(r["R"]), int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
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
                if (ds, a.R, V, s) in done:
                    continue
                hits = sorted(a.soft_root.glob(f"{ds}/*_r{a.R:03d}/gate-valsplit_{ds}_*_R{a.R}_V{V}s{s}_H20_pgd-linf_lr{a.soft_lr}_it300"))
                if not hits or not (hits[0] / "test_metrics.json").exists():
                    continue
                d = hits[0]
                cfg = json.loads((d / "config.json").read_text())
                tm = json.loads((d / "test_metrics.json").read_text())
                sp, pz = np.load(d / "split.npz"), np.load(d / "poison.npz")
                K, k = int(cfg["n_classes"]), int(cfg["k_remove"])
                X_all, y_all = G.load(Path(cfg["context"]), device)
                Xt, yt = G.load(Path(cfg["test"]), device)
                yt_t = torch.as_tensor(yt, device=device)
                idx = lambda r: torch.as_tensor(r, device=device)  # noqa: E731
                Xc, Xv = X_all[idx(sp["context_rows"])], X_all[idx(sp["val_rows"])]
                yv = y_all[sp["val_rows"]]
                n = len(sp["context_rows"])
                yp_np = pz["y_poisoned"].astype(int)
                yp = torch.tensor(yp_np, dtype=torch.float32, device=device)
                yv_f = torch.tensor(yv, dtype=torch.float32, device=device)
                pois = np.zeros(n, dtype=bool)
                pois[pz["poisoned_rows"]] = True
                if K not in clfs:
                    clfs[K] = build_tabpfn_v2(device, cfg["args"]["model_seed"], recompute_layers=True, n_classes=K)
                clf = clfs[K]

                def test_acc(X_ctx, y_ctx, g=None):
                    null = G.capture_null_input(clf, X_ctx, y_ctx) if g is not None else None
                    gt = None if g is None else torch.as_tensor(g, dtype=torch.float32, device=device)
                    p = G.evaluate(clf, X_ctx, y_ctx, Xt, yt_t, gt, null, need_grad=False, batch_size=None)[2]
                    return float(multiclass_metrics(p, yt, K)["accuracy"])

                def plus_v(keep):  # the kept context rows, then the V clean rows
                    kk = idx(np.flatnonzero(keep))
                    return torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv_f])

                row = {"dataset": ds, "R": a.R, "V": V, "seed": s, "k": k, "n_context": n}
                row["v_only"] = test_acc(Xv, yv_f)
                ref = tm["metrics"]["val_as_context"]["accuracy"]
                assert abs(row["v_only"] - ref) <= 0.003, (d, row["v_only"], ref)  # the same draw
                row["poisoned"] = tm["metrics"]["poisoned_no_gate"]["accuracy"]
                row["poisoned_plus_v"] = test_acc(*plus_v(np.ones(n, bool)))
                g = np.load(d / "gates" / "top1.npz")["g"]
                row["soft_iter"] = tm["best_iteration"]
                row["soft"] = tm["metrics"]["poisoned_best_gate_soft"]["accuracy"]
                again = test_acc(Xc, yp, g=g)  # the gating path here reproduces the run's
                assert abs(again - row["soft"]) <= 0.003, (d, again, row["soft"])
                X_aug, y_aug = plus_v(np.ones(n, bool))
                row["soft_plus_v"] = test_acc(X_aug, y_aug, g=np.concatenate([g, np.ones(len(yv))]))
                row["prec_soft"] = float(pois[G.lowest_gated(g, k, pz["tie_order"])].mean())
                # the clean rows as a filter: p(given label | the V clean rows as the context)
                pc = G.evaluate(clf, Xv, yv_f, Xc, torch.as_tensor(yp_np, device=device), None, None,
                                need_grad=False, batch_size=None)[2]
                score = pc[np.arange(n), yp_np]
                drop = np.argsort(score, kind="stable")[:k]
                keep = np.ones(n, bool)
                keep[drop] = False
                row["prec_vfilter"] = float(pois[drop].mean())
                kk = idx(np.flatnonzero(keep))
                row["vfilter"] = test_acc(Xc[kk], yp[kk])
                row["vfilter_plus_v"] = test_acc(*plus_v(keep))
                row["oracle"] = tm["metrics"]["poisoned_oracle_gate"]["accuracy"]
                row["oracle_plus_v"] = test_acc(*plus_v(~pois))
                if a.topk_root is not None:
                    vote = sorted(a.topk_root.glob(f"{ds}/*_r{a.R:03d}/gate-topk-vote*_{ds}_*_R{a.R}_V{V}s{s}_*/vote.npz"))
                    if vote:
                        m = np.load(vote[0])["mask"]
                        vj = json.loads((vote[0].parent / "vote.json").read_text())
                        assert np.array_equal(np.load(Path(vj["restart_runs"][0]) / "split.npz")["context_rows"],
                                              sp["context_rows"]), vote[0]
                        row["topk"] = vj["vote"]["test"]["accuracy"]
                        row["topk_plus_v"] = test_acc(*plus_v(m == 1))
                        row["prec_topk"] = float(pois[m == 0].mean())
                w.writerow(row)
                fh.flush()
                print(f"{ds} V{V} s{s}: V only {100 * row['v_only']:.1f} | poisoned+V {100 * row['poisoned_plus_v']:.1f}"
                      f" | soft {100 * row['soft']:.1f} soft+V {100 * row['soft_plus_v']:.1f} | V-filter "
                      f"{100 * row['vfilter']:.1f} +V {100 * row['vfilter_plus_v']:.1f} (prec {row['prec_vfilter']:.2f})"
                      f" | oracle+V {100 * row['oracle_plus_v']:.1f}"
                      + (f" | top-k+V {100 * row['topk_plus_v']:.1f}" if "topk_plus_v" in row else ""), flush=True)
    fh.close()


if __name__ == "__main__":
    main()

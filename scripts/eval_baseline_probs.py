#!/usr/bin/env python
"""Per-row test predictions of every way of using the clean rows (the baseline table).

For each soft-gate run of results/gate_valsplit at R%, validation seed 0 (one draw: dataset,
V, 20% of V held out for the gates), with plain TabPFN and the weights learned before (no
training):

  copied from the runs   poisoned; soft gate (the gates at the run's held-out-accuracy
                         iteration); oracle (the poisoned rows removed): test_probs.npz.
                         top-k (the vote mask, its rows masked): vote.npz of --topk-root
  computed here          V only (all V clean rows as the context); V + poisoned; V + filtered
                         (the R% least plausible given labels under TabPFN with the V rows as
                         its context removed); soft gate + V (the learned gates on the poisoned
                         rows, the V rows at gate 1); top-k + V (the vote's kept rows);
                         oracle + V

Every computed accuracy is checked against results/tables/gate_soft/combos_R20.csv
(scripts/eval_context_combos.py, the same computation). Writes <out>/<dataset>_V<V>.npz:
y_test, the test probabilities of each method, and the precision@R of the filter, of the
soft gates' k lowest and of the top-k vote (soft gate + V and top-k + V remove or gate the
same rows, so share them).

    python scripts/eval_baseline_probs.py --gpu 1
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

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
METHODS = ("v_only", "poisoned", "poisoned_plus_v", "vfilter_plus_v", "soft", "soft_plus_v", "topk", "topk_plus_v",
           "oracle_train", "oracle_plus_v")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soft-root", type=Path, default=Path("results/gate_valsplit"))
    ap.add_argument("--topk-root", type=Path, default=Path("results/gate_topk_gen"))
    ap.add_argument("--combos", type=Path, default=Path("results/tables/gate_soft/combos_R20.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/gate_soft/baseline_probs_R20_s0"))
    a = ap.parse_args(argv)
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    ref = {(r["dataset"], int(r["V"])): r for r in csv.DictReader(open(a.combos))
           if int(r["R"]) == a.R and int(r["seed"]) == a.seed}
    a.out.mkdir(parents=True, exist_ok=True)
    clfs, mismatches = {}, []
    for ds in DATASETS:
        for d in sorted(a.soft_root.glob(f"{ds}/*_r{a.R:03d}/gate-valsplit_{ds}_*_R{a.R}_V*s{a.seed}_H20_pgd-linf_lr0.05_it300"),
                        key=lambda p: int(p.name.split(f"_R{a.R}_V")[1].split("s")[0])):
            if not (d / "test_metrics.json").exists():
                continue
            cfg = json.loads((d / "config.json").read_text())
            V = int(cfg["args"]["val_from_context"])
            dest = a.out / f"{ds}_V{V}.npz"
            if dest.exists():
                continue
            tp = np.load(d / "test_probs.npz")
            sp, pz = np.load(d / "split.npz"), np.load(d / "poison.npz")
            K, k = int(cfg["n_classes"]), int(cfg["k_remove"])
            X_all, y_all = G.load(Path(cfg["context"]), device)
            Xt, yt = G.load(Path(cfg["test"]), device)
            assert np.array_equal(yt, tp["y_test"]), d
            yt_t = torch.as_tensor(yt, device=device)
            idx = lambda r: torch.as_tensor(r, device=device)  # noqa: E731
            Xc, Xv = X_all[idx(sp["context_rows"])], X_all[idx(sp["val_rows"])]
            n = len(sp["context_rows"])
            yp_np = pz["y_poisoned"].astype(int)
            yp = torch.tensor(yp_np, dtype=torch.float32, device=device)
            yv_f = torch.tensor(y_all[sp["val_rows"]], dtype=torch.float32, device=device)
            pois = np.zeros(n, dtype=bool)
            pois[pz["poisoned_rows"]] = True
            if K not in clfs:
                clfs[K] = build_tabpfn_v2(device, cfg["args"]["model_seed"], recompute_layers=True, n_classes=K)
            clf = clfs[K]

            def probs(X_ctx, y_ctx, g=None):
                null = G.capture_null_input(clf, X_ctx, y_ctx) if g is not None else None
                gt = None if g is None else torch.as_tensor(g, dtype=torch.float32, device=device)
                return G.evaluate(clf, X_ctx, y_ctx, Xt, yt_t, gt, null, need_grad=False, batch_size=None)[2]

            def plus_v(keep):  # the kept context rows, then the V clean rows
                kk = idx(np.flatnonzero(keep))
                return torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv_f])

            P = {"poisoned": tp["poisoned_no_gate"], "soft": tp["poisoned_best_gate_soft"],
                 "oracle_train": tp["poisoned_oracle_removal"]}
            P["v_only"] = probs(Xv, yv_f)
            P["poisoned_plus_v"] = probs(*plus_v(np.ones(n, bool)))
            g = np.load(d / "gates" / "top1.npz")["g"]
            X_aug, y_aug = plus_v(np.ones(n, bool))
            P["soft_plus_v"] = probs(X_aug, y_aug, g=np.concatenate([g, np.ones(len(yv_f))]))
            prec = {"prec_soft": float(pois[G.lowest_gated(g, k, pz["tie_order"])].mean())}
            # the clean rows as a filter: p(given label | the V clean rows as the context)
            pc = G.evaluate(clf, Xv, yv_f, Xc, torch.as_tensor(yp_np, device=device), None, None,
                            need_grad=False, batch_size=None)[2]
            drop = np.argsort(pc[np.arange(n), yp_np], kind="stable")[:k]
            keep = np.ones(n, bool)
            keep[drop] = False
            prec["prec_vfilter"] = float(pois[drop].mean())
            P["vfilter_plus_v"] = probs(*plus_v(keep))
            P["oracle_plus_v"] = probs(*plus_v(~pois))
            votes = sorted(a.topk_root.glob(f"{ds}/*_r{a.R:03d}/gate-topk-vote*_{ds}_*_R{a.R}_V{V}s{a.seed}_*/vote.npz"))
            assert len(votes) == 1, (ds, V, votes)
            vz = np.load(votes[0])
            vj = json.loads((votes[0].parent / "vote.json").read_text())
            assert np.array_equal(np.load(Path(vj["restart_runs"][0]) / "split.npz")["context_rows"], sp["context_rows"])
            m = vz["mask"]
            P["topk"] = vz["test_probs"]
            P["topk_plus_v"] = probs(*plus_v(m == 1))
            prec["prec_topk"] = float(pois[m == 0].mean())

            acc = {key: float(multiclass_metrics(P[key], yt, K)["accuracy"]) for key in METHODS}
            r = ref[(ds, V)]
            checks = {key: float(r[key]) for key in METHODS if key in r} | {
                "oracle_train": json.loads((d / "test_metrics.json").read_text())["metrics"]["poisoned_oracle_removal"]["accuracy"],
                "prec_soft": float(r["prec_soft"]), "prec_vfilter": float(r["prec_vfilter"]), "prec_topk": float(r["prec_topk"])}
            got = acc | prec
            for key, want in checks.items():
                if abs(got[key] - want) > 1e-6:
                    mismatches.append((ds, V, key, got[key], want))
            np.savez_compressed(dest, y_test=yt, **{key: P[key].astype(np.float32) for key in METHODS},
                                **{key: np.float64(v) for key, v in prec.items()}, k=k, n_context=n, n_classes=K)
            print(f"{ds} V{V}: " + " ".join(f"{key} {100 * acc[key]:.1f}" for key in METHODS), flush=True)
    for mm in mismatches:
        print("differs from the earlier numbers:", mm)
    print(f"{len(mismatches)} differences from {a.combos} / the runs")


if __name__ == "__main__":
    main()

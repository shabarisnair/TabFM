#!/usr/bin/env python
"""Is the V-filter's edge specific to one attack? V only vs V-filter vs oracle, per attack.

For each dataset, attack, V and validation seed, the draw is built exactly as
gate_context.py builds it (V class-stratified clean rows from the context file, the rest
poisoned at R% with --attack's defaults, run 0). No training. The attacks are those of
tabfm_experiments.multiclass_attacks; boundary-flip-knn is the one aimed at evading a filter
like the V-filter (its new labels look plausible to a model of the clean data). Test
accuracy of:

  poisoned        the poisoned context
  V only          the V clean rows alone
  V-filter (+ V)  the R% of context rows whose given label is least plausible to TabPFN
                  with the V clean rows as its context removed (plus the V clean rows)
  oracle (+ V)    exactly the poisoned rows removed (plus the V clean rows)
and the V-filter's precision. For hub-flip-local the split and poison are checked against
the saved soft-gate run of the same draw.

    python scripts/eval_vfilter_attacks.py --gpu 0 --datasets gas-drift url_unique students_dropout \\
        --attacks hub-flip-local hub-flip-random label-flip-bernoulli --out results/tables/gate_soft/vfilter_attacks_R20.csv
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

CONTEXT = {"gas-drift": "datasets/gas-drift/splits/context_5000.csv",
           "url_unique": "datasets/url_unique/splits/selected/natural/context_5000.csv",
           "heloc": "datasets/heloc/splits/selected/natural/context_5000.csv",
           "credit-approval": "datasets/credit-approval/splits/train.csv",
           "students_dropout": "datasets/students_dropout/splits/context_3000.csv"}
TEST = {"gas-drift": "datasets/gas-drift/splits/test_attack_1000.csv",
        "url_unique": "datasets/url_unique/splits/test_attack_1000.csv",
        "heloc": "datasets/heloc/splits/test_attack_1000.csv",
        "credit-approval": "datasets/credit-approval/splits/test.csv",
        "students_dropout": "datasets/students_dropout/splits/test_attack_500.csv"}
FIELDS = ["dataset", "attack", "R", "V", "seed", "k", "poisoned", "v_only", "vfilter", "vfilter_plus_v",
          "oracle", "oracle_plus_v", "prec_vfilter"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--attacks", nargs="+", required=True)
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2
    from tabfm_experiments.sampling import k_from_percent

    device = G.device_from_gpu(a.gpu)
    done = set()
    if a.out.exists():
        done = {(r["dataset"], r["attack"], int(r["V"]), int(r["seed"])) for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for ds in a.datasets:
        X_file, y_file = G.load(Path(CONTEXT[ds]), device)
        Xt, yt = G.load(Path(TEST[ds]), device)
        yt_t = torch.as_tensor(yt, device=device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
        clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K)

        def acc(X_ctx, y_ctx):
            p = G.evaluate(clf, X_ctx, y_ctx, Xt, yt_t, None, None, need_grad=False, batch_size=None)[2]
            return float(multiclass_metrics(p, yt, K)["accuracy"])

        for V in a.sizes:
            for s in a.seeds:
                val_rows = G.draw_validation(y_file, V, s)
                ctx_rows = np.setdiff1d(np.arange(len(y_file)), val_rows)
                Xv, yv = X_file[torch.as_tensor(val_rows, device=device)], y_file[val_rows]
                Xc, yc = X_file[torch.as_tensor(ctx_rows, device=device)], y_file[ctx_rows]
                yv_f = torch.tensor(yv, dtype=torch.float32, device=device)
                n = len(ctx_rows)
                k = k_from_percent(n, a.R)
                for attack in a.attacks:
                    if (ds, attack, V, s) in done:
                        continue
                    rows, y_pois = poison(attack, Xc, yc, K, a.R, run_id=0, row_seed=Seeds.row_seed)
                    if attack == "hub-flip-local":  # the same draw as the saved soft-gate run?
                        hit = sorted(Path("results/gate_valsplit").glob(
                            f"{ds}/*_r{int(a.R):03d}/gate-valsplit_{ds}_*_R{int(a.R)}_V{V}s{s}_H20_pgd-linf_lr0.05_it300"))
                        if hit:
                            assert np.array_equal(np.load(hit[0] / "split.npz")["context_rows"], ctx_rows), hit[0]
                            assert np.array_equal(np.load(hit[0] / "poison.npz")["y_poisoned"], y_pois), hit[0]
                    pois = np.zeros(n, dtype=bool)
                    pois[rows] = True
                    yp = torch.tensor(y_pois, dtype=torch.float32, device=device)
                    pc = G.evaluate(clf, Xv, yv_f, Xc, torch.as_tensor(y_pois, device=device), None, None,
                                    need_grad=False, batch_size=None)[2]
                    drop = np.argsort(pc[np.arange(n), y_pois], kind="stable")[:k]
                    keep = np.ones(n, dtype=bool)
                    keep[drop] = False
                    kk, ko = torch.as_tensor(np.flatnonzero(keep), device=device), torch.as_tensor(np.flatnonzero(~pois), device=device)
                    row = {"dataset": ds, "attack": attack, "R": a.R, "V": V, "seed": s, "k": k,
                           "poisoned": acc(Xc, yp), "v_only": acc(Xv, yv_f),
                           "vfilter": acc(Xc[kk], yp[kk]), "vfilter_plus_v": acc(torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv_f])),
                           "oracle": acc(Xc[ko], yp[ko]), "oracle_plus_v": acc(torch.cat([Xc[ko], Xv]), torch.cat([yp[ko], yv_f])),
                           "prec_vfilter": float(pois[drop].mean())}
                    w.writerow(row)
                    fh.flush()
                    print(f"{ds} {attack} V{V} s{s}: poisoned {100 * row['poisoned']:.1f} | V only {100 * row['v_only']:.1f}"
                          f" | V-filter {100 * row['vfilter']:.1f} +V {100 * row['vfilter_plus_v']:.1f} (prec {row['prec_vfilter']:.2f})"
                          f" | oracle {100 * row['oracle']:.1f} +V {100 * row['oracle_plus_v']:.1f}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()

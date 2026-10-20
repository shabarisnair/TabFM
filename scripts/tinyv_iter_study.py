#!/usr/bin/env python
"""How many updates should the tiny-V learners run? With one to five clean rows per class there is no held-out part of
V, so the number of updates must be fixed in advance (bench_tinyv used 20). This runs both learners past that point
and records the TEST metrics along the way, to find where they stop improving and start to overfit V.

  m1    attention weights, hard top-k (tinyv_learned.ATTN_CONFIGS["topk"]: PGD-L-inf lr 0.2, probe 0.05, V-filter
        start); deployed hard + V (the k lowest-ranked rows removed, all of V added)
  soft  the best soft form (results/tables/attn_form/findings.md): w in [0, 1], V-filter start, PGD-L-inf lr 0.8;
        deployed soft + V (the poisoned rows at their weights, V added at weight 1; w = 0 deletes a row); also soft
        alone
Both learn on all of V (as in bench_tinyv). Instances: R = 30%, c = 1, 3, 5 rows per class, the 6 datasets, built as
bench_tinyv.build_draw does but with --seed (default 1: fresh draws, so the count is not tuned on the benchmark's own
seed-0 instances). Test metrics after every update in EVALS. Workers on one machine share the queue (lock files).

    python scripts/tinyv_iter_study.py work --device cuda:3 [--skip-datasets gas-drift]
    python scripts/tinyv_iter_study.py work --device cuda:3 --steps 300 --methods m1 --runs-dir runs_long \
        --datasets is_this_a_good_customer ecommerce_shipping students_dropout      # longer curves
    python scripts/tinyv_iter_study.py report
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("TINYV_ITERS_ROOT", "results/tinyv_iters"))
DATASETS = ("gas-drift", "coil2000_insurance_policies", "heloc", "students_dropout", "ecommerce_shipping",
            "is_this_a_good_customer")
CS, R = (1, 3, 5), 30.0
METHODS = ("m1", "soft", "soft_lr0.2", "soft_lr0.05")  # soft: lr 0.8 (the V = 500 choice); soft_lr<x>: the same at lr x
STEPS = 80
EVALS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300, 400, 500)
SOFT = dict(param="w", opt="pgd-linf", lr=0.8)  # attn_form_study's best: w / vfilter / pgd, lr 0.8


def tag_of(ds: str, c: int, seed: int) -> str:
    return f"{ds}_R{R:g}_c{c}_s{seed}"


def get_draw(bd, ds: str, c: int, seed: int) -> dict:
    """bench_tinyv's construction with this seed, saved once under ROOT/draws; in load_draw's form."""
    import bench_tinyv as B
    path = ROOT / "draws" / f"{tag_of(ds, c, seed)}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **B.build_draw(bd, R, c, seed))
    z = np.load(path)
    d = {key: z[key] for key in ("v_rows", "ctx_rows", "y_given", "pois_rows")}
    d |= {"k": int(z["k"]), "n": len(d["ctx_rows"]), "tag": tag_of(ds, c, seed), "seed": seed}
    d["pois"] = np.zeros(d["n"], dtype=bool)
    d["pois"][d["pois_rows"]] = True
    d["Xc"], d["Xv"] = bd.X[bd.idx(d["ctx_rows"])], bd.X[bd.idx(d["v_rows"])]
    d["yv"], d["y_true"] = bd.y[d["v_rows"]], bd.y[d["ctx_rows"]]
    return d


def run(bd, d: dict, method: str, steps: int = STEPS, log=print) -> dict:
    import attn_form_study as S
    import attn_learn as A
    import bench_tinyv as B
    import tinyv_learned as L
    from tabfm_experiments.row_weights import evaluate_weighted
    dev, clf = bd.device, bd.clf
    dd = {"n": d["n"], "Xc": d["Xc"], "yp": d["y_given"], "Xf": d["Xv"], "yf": d["yv"]}  # fitted rows = all of V
    if method == "m1":
        c = dict(A.DEFAULTS) | L.ATTN_CONFIGS["topk"] | {"seed": d["seed"]}
        raw0 = A.init_raw(c, dd, clf, dev)
    else:
        lr = float(method.split("_lr")[1]) if "_lr" in method else SOFT["lr"]
        c = dict(A.DEFAULTS) | SOFT | {"lr": lr, "floor": 0.0, "t_max": S.T_MAX, "t0": S.T0, "seed": d["seed"]}
        raw0 = S.init_raw("w", "vfilter", dd, clf, dev)
    torch.manual_seed(c["seed"])
    p = A.Param(c["param"], raw0, c, d["k"])
    p.project()
    st = A.Stepper(p, c)
    zv = torch.zeros(len(d["yv"]), device=dev)
    Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([d["y_given"], d["yv"]])
    curve, t0, train_ce = [], time.time(), None
    evals = [e for e in EVALS if e <= steps]
    for it in range(steps + 1):
        if it in evals:
            e = {"it": it, "train_ce": train_ce}
            if method == "m1":
                removed = p.deleted().cpu().numpy()
                e |= {"test": B.test_scores(bd, d, ~removed), "prec": float(d["pois"][removed].mean())}
            else:
                lw = p.log_w().detach()
                zero = torch.isinf(lw).cpu().numpy()
                e |= {"test": B.metrics(evaluate_weighted(clf, Xcv, ycv, bd.Xt, bd.yt, torch.cat([lw, zv]), need_grad=False)[2], bd.yt),
                      "test_nov": B.metrics(evaluate_weighted(clf, d["Xc"], d["y_given"], bd.Xt, bd.yt, lw, need_grad=False)[2], bd.yt),
                      "n_zero": int(zero.sum()), "prec_zero": float(d["pois"][zero].mean()) if zero.any() else None}
            e["sec"] = round(time.time() - t0, 1)
            curve.append(e)
            if it % 20 == 0:
                log(f"    {method} it {it:3d}: test auc {e['test']['auc']:.4f} acc {e['test']['acc']:.4f} ce {e['test']['ce']:.4f}"
                    f"  ({e['sec']:.0f} s)")
        if it == steps:
            break
        lw = p.probe_log_w() if method == "m1" else p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        loss, g, _ = evaluate_weighted(clf, d["Xc"], d["y_given"], d["Xv"], d["yv"], lw_d, need_grad=True)
        train_ce = float(loss)
        if method == "m1":
            g_raw = g
        else:
            p.raw.grad = None
            lw.backward(g)
            g_raw = torch.nan_to_num(p.raw.grad.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        st.step(g_raw)
    return {"method": method, "instance": d["tag"], "n": d["n"], "k": d["k"], "n_v": int(len(d["yv"])), "config": c,
            "curve": curve, "seconds": round(time.time() - t0, 1), "host": socket.gethostname()}


def work(device: str, seed: int, skip: list[str], steps: int = STEPS, methods=METHODS, datasets=DATASETS,
         runs_dir: str = "runs") -> None:
    import bench_largev as BL
    import bench_tinyv as B
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} {device}] {m}", flush=True)  # noqa: E731
    sets = {}
    for ds in [x for x in DATASETS if x in datasets]:  # the expensive datasets first
        if ds in skip:
            continue
        for c in CS:
            for method in methods:
                out = ROOT / runs_dir / tag_of(ds, c, seed) / f"{method}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists() or not BL.claim(out.with_suffix(".lock")):
                    continue
                try:
                    if ds not in sets:
                        sets.clear()
                        torch.cuda.empty_cache()
                        sets[ds] = B.BenchData(ds, device)
                    bd = sets[ds]
                    d = get_draw(bd, ds, c, seed)
                    log(f"{d['tag']} {method}: start (n {d['n']}, k {d['k']}, V {len(d['yv'])})")
                    res = run(bd, d, method, steps, log)
                    BL.write_json(out, res)
                    log(f"{d['tag']} {method}: done in {res['seconds']:.0f} s")
                finally:
                    out.with_suffix(".lock").unlink(missing_ok=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("work",))
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-datasets", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["m1", "soft"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--runs-dir", default="runs", help="under ROOT; a longer study goes to its own folder")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    work(a.device, a.seed, a.skip_datasets, a.steps, a.methods, a.datasets, a.runs_dir)


if __name__ == "__main__":
    main()

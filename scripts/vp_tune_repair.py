#!/usr/bin/env python
"""Tuning the two repair methods of the V + P benchmark (bench_vp.py's methods 12 and 13) on the development draws of
scripts/vp_tune.py (seeds 100 and 101; the benchmark uses 0-4), before the benchmark runs.

Both ignore P's given labels: P is unlabelled, and TabPFN labels it from V.
  one-shot  TabPFN with V alone as the context predicts every row of P; a subset of P joins V with the predicted labels.
            Candidates, by TabPFN's confidence (max probability): the top x% of P (top), the top x% of each predicted
            class (cls), or the rows above a confidence threshold (thr). A subset with no row is V alone.
  greedy    the trusted set starts as V; every step TabPFN with V + the rows labelled so far predicts the rest, and the
            ceil(1% of |P|) rows it is most sure of join with their predicted labels: over all rows (global), or split
            over the predicted classes in proportion to the pool's predictions (cls). Run until P is used up, with the
            test metrics every 5 steps, so that any stop at a multiple of 5% of P can be scored afterwards.
The poisonings' draws share V and P's rows (checked), so with the labels unused each (dataset, c, seed) is run once:
72 units (36 draws x the two selections; the global unit also scores the one-shot candidates). Several workers can
share a list (locks). The report ranks every candidate by its mean rank among its family per draw and metric (ROC-AUC,
balanced accuracy, log loss), as vp_tune_report.py.

    python scripts/vp_tune_repair.py plan --scripts 7 [--out scripts/vp_tune_repair_jobs]   # GPU scripts, other machines
    python scripts/vp_tune_repair.py run --device cuda:0 [--units FILE]   # or --device cpu --threads 8
    python scripts/vp_tune_repair.py status [--units FILE]
    python scripts/vp_tune_repair.py report                # -> results/tables/vp_tune/report_repair.md
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_tinyv as B  # noqa: E402
import bench_vp as V  # noqa: E402
import vp_tune as T  # noqa: E402

OUT = T.ROOT / "repair"
ONE_SHOT = [("top", x) for x in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)] + [("cls", x) for x in (0.1, 0.2, 0.3, 0.5, 0.7)] \
    + [("thr", x) for x in (0.6, 0.7, 0.8, 0.9)]
SELECTS = ("global", "cls")
TEST_EVERY = 5
# heaviest first, so that the workers finish together
ORDER = ("gas-drift", "coil2000_insurance_policies", "heloc", "students_dropout", "ecommerce_shipping",
         "is_this_a_good_customer")


def units() -> list[tuple[str, int, int, str]]:
    return [(ds, c, s, sel) for ds in ORDER for c in T.CS for s in T.SEEDS for sel in SELECTS]


def unit_name(ds: str, c: int, seed: int, sel: str) -> str:
    return f"{ds}_c{c}_s{seed}_{sel}"


def cand_name(rule: str, x: float) -> str:
    return f"{rule}{x:g}"


def run_unit(bd: B.BenchData, ds: str, c: int, seed: int, sel: str) -> dict:
    d = T.load_draw(bd, ds, "rand45", c, seed)
    z = np.load(T.ROOT / "draws" / f"{T.tag_of(ds, 'hub30', c, seed)}.npz")
    assert np.array_equal(z["v_rows"], d["v_rows"]) and np.array_equal(z["ctx_rows"], d["ctx_rows"]), "V or P differs"
    d["K"] = bd.K
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {"ds": ds, "c": c, "seed": seed, "select": sel, "n": d["n"], "K": bd.K, "host": socket.gethostname(),
           "threads": torch.get_num_threads()}
    if sel == "global":  # the one-shot candidates, once per draw
        Q = V.predict_pool(bd, d, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.arange(d["n"]))
        lab = Q.argmax(1).astype(np.int64)
        conf = Q.max(1)
        v_only = B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)
        cache, cands = {}, {}
        for rule, x in ONE_SHOT:
            keep = V.one_shot_select(Q, rule, x)
            key = np.packbits(keep).tobytes()
            if key not in cache:
                cache[key] = B.test_scores(bd, d, keep, lab) if keep.any() else v_only
            cands[cand_name(rule, x)] = {"test": cache[key]} | V.label_acc_stats(d, keep, lab)
        out["one_shot"] = cands
        out["v_only"] = v_only
        out["conf"] = {"mean": float(conf.mean()), "acc": float((lab == d["y_true"]).mean())}
        out["seconds_one_shot"] = round(time.time() - t0, 1)
        np.savez_compressed(OUT / f"{unit_name(ds, c, seed, sel)}_Q.npz", Q=Q.astype(np.float32))
    path = V.repair_path(bd, d, sel, 0.01, 1.0, TEST_EVERY)
    out |= {"m": path["m"], "steps": path["steps"], "states": path["states"], "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(OUT / f"{unit_name(ds, c, seed, sel)}.npz", pos=path["pos"], labels=path["lab"].astype(np.int16))
    return out


def read_units(path: Path | None) -> list[tuple[str, int, int, str]]:
    if path is None:
        return units()
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            ds, c, seed, sel = line.split()
            out.append((ds, int(c), int(seed), sel))
    return out


def run(a) -> None:
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    bd = None
    for ds, c, seed, sel in read_units(a.units):
        if a.datasets and ds not in a.datasets:
            continue
        name = unit_name(ds, c, seed, sel)
        out = OUT / f"{name}.json"
        if out.exists():
            continue
        lock = out.with_suffix(".lock")
        if not B.claim(lock):
            continue
        try:
            if bd is None or bd.ds != ds:
                bd = None
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bd = B.BenchData(ds, a.device)
            t0 = time.time()
            res = run_unit(bd, ds, c, seed, sel)
            T.write_json(out, res)
            last = res["states"][-1]["test"]
            print(f"{name}: {time.time() - t0:.0f} s; all of P labelled: auc {last['auc']:.3f} bacc {last['bacc']:.3f} "
                  f"ce {last['ce']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001 - one failing unit must not stop the worker; a second pass retries it
            import traceback
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
        finally:
            lock.unlink(missing_ok=True)


def status(a) -> None:
    todo = read_units(a.units)
    done = [u for u in todo if (OUT / f"{unit_name(*u)}.json").exists()]
    print(f"finished {len(done)} / {len(todo)} units" + (f" of {a.units}" if a.units else ""))
    for u in todo:
        if u not in done:
            print("  not done:", unit_name(*u))


# ---------------------------------------------------------------- GPU scripts for machines that do not share a disk

# seconds on a free H100 per unit, from bench_vp's measured greedy run (plan_vp.COST: to about 75% of P, records
# every 2 steps): the path to all of P costs about (1 / 0.75)^2 = 1.8 of it; the global unit adds the one-shot candidates
LAUNCHER = """#!/bin/bash
# The repair methods' development study (scripts/vp_tune_repair.py; planned by its plan stage), script {i:02d} of {N}:
# {n_units} units, about {minutes:.0f} min on a free H100. Run from a clone of this repository with results/vp_tune/draws,
# datasets/ and the preflight inputs (results/attn_form/preflight_reference, results/bench_largev/draws) copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Results: results/vp_tune/repair/ (bring them back with
# {out}/merge.sh). No W&B. Rerunning resumes; logs in {logs}/. TabPFN inference only (no gradients): far less GPU
# memory than the benchmark's attention jobs.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN (as the benchmark runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p {logs}
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > {logs}/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" {logs}/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/vp_tune_repair.py run --device "cuda:$GPU" --units {out}/units_{i:02d}.txt >> {logs}/gpu_{i:02d}.log 2>&1
done
$PY scripts/vp_tune_repair.py status --units {out}/units_{i:02d}.txt
grep -h "FAILED" {logs}/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""
MERGE = """#!/bin/bash
# Bring one machine's finished units of the repair development study ({out}/gpu_XX.sh) into {res}; files already here
# are kept as they are, and the locks and temporary files of units still running there are left out.
#   bash {out}/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/{up}"
count() {{ find {res} -name '*.json' | wc -l; }}
mkdir -p {res}
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${{1%/}}/{res}/" {res}/
echo "merged $(( $(count) - before )) new unit results ($(count) of 72 here)"
"""
CODE = ["scripts/vp_tune_repair.py", "scripts/vp_tune.py", "scripts/bench_vp.py", "scripts/bench_tinyv.py",
        "scripts/bench_small_v.py", "scripts/schedule_study.py", "scripts/gate_context.py",
        "scripts/tabfm_experiments/runtime.py", "scripts/tabfm_experiments/config.py"]


def unit_cost(ds: str, sel: str) -> float:
    import plan_vp
    cc = plan_vp.COST[ds]
    return 1.8 * cc["greedy"] + (cc["oneshot"] if sel == "global" else 0.0) + 10.0


def plan(a) -> None:
    from plan_soft_m1 import sha256
    out = a.out
    todo = [u for u in units() if not (OUT / f"{unit_name(*u)}.json").exists()]
    cost = {u: unit_cost(u[0], u[3]) for u in todo}
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in sorted(todo, key=lambda u: -cost[u]):  # longest first onto the least-loaded script
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += cost[u]
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("gpu_*.sh")) + list(out.glob("units_*.txt")):
        old.unlink()
    up = "/".join([".."] * len(out.parts))
    logs = f"logs/{out.name}"
    lines = ["| script | units | minutes (est., free H100) | gas-drift / coil2000 units |", "|---|---|---|---|"]
    for i, b in enumerate(bins):
        b.sort(key=lambda u: (ORDER.index(u[0]), u[1], u[2], u[3]))  # one dataset after another: fewer model loads
        (out / f"units_{i:02d}.txt").write_text("".join(f"{ds} {c} {s} {sel}\n" for ds, c, s, sel in b))
        sh = out / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_units=len(b), minutes=load[i] / 60, out=out.as_posix(), up=up,
                                      logs=logs))
        os.chmod(sh, 0o755)
        heavy = sum(u[0] in ("gas-drift", "coil2000_insurance_policies") for u in b)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 60:.0f} | {heavy} |")
    merge = out / "merge.sh"
    merge.write_text(MERGE.format(out=out.as_posix(), up=up, res=OUT.as_posix()))
    os.chmod(merge, 0o755)
    files = list(CODE)
    for ds in B.DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
    files += [f"{T.ROOT}/draws/{T.tag_of(ds, att, c, s)}.npz" for ds, att, c, s in T.instances()]
    (out / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{len(todo)} units, about {load.sum() / 60:.0f} H100-minutes in all, {load.max() / 60:.0f} on the busiest "
              "script (a GPU shared with another busy job: two to four times that). Inputs to copy to each machine: "
              f"{T.ROOT}/draws ({len(T.instances())} draws), datasets/, results/attn_form/preflight_reference, "
              "results/bench_largev/draws."]
    print("\n".join(lines))
    (out / "split.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- report

METRICS = (("auc", False), ("bacc", False), ("ce", True))


def mean_ranks(table: dict) -> dict:
    """table: draw -> {candidate: test metrics}. Per draw and metric, the candidates' ranks (1 = best, ties averaged),
    averaged over the metrics and the draws."""
    import pandas as pd
    per = []
    for draw, cands in table.items():
        df = pd.DataFrame({k: v for k, v in cands.items()}).T
        r = sum(df[m].rank(ascending=asc) for m, asc in METRICS) / len(METRICS)
        per.append(r.rename(draw))
    return pd.concat(per, axis=1).mean(axis=1).to_dict()


def report(a) -> None:
    import pandas as pd
    recs = {}
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        recs[(r["ds"], r["c"], r["seed"], r["select"])] = r
    draws = sorted({(ds, c, s) for ds, c, s, _ in recs})
    full = [k for k in draws if all((*k, sel) in recs for sel in SELECTS)]
    lines = ["# Repair methods on the development draws", "",
             f"{len(full)} of 36 (dataset, c, seed) draws complete; P's labels are never used, so rand45 and hub30 "
             "share every number here. Mean rank: per draw and metric among the family's candidates (1 = best), "
             "averaged over ROC-AUC, balanced accuracy and log loss and over the draws. Values: means over the draws. "
             "label acc: the accuracy of the labels given to P's rows (against the true labels). V + P: the "
             "development runs' references (GPU), for orientation only.", ""]
    refs = {}
    for ds, c, s in full:
        for att in T.ATTACKS:
            vf = json.loads((T.ROOT / "vfilter" / f"{T.tag_of(ds, att, c, s)}.json").read_text())
            refs[(ds, c, s, att)] = vf["refs"]
    # the one-shot candidates
    one = {k: {n: x["test"] for n, x in recs[(*k, "global")]["one_shot"].items()} for k in full}
    rk = mean_ranks(one)
    rows = []
    for name in rk:
        vals = pd.DataFrame([one[k][name] for k in full]).mean()
        la = np.nanmean([np.nan if (x := recs[(*k, "global")]["one_shot"][name]["label_acc"]) is None else x for k in full])
        sk = np.mean([recs[(*k, "global")]["one_shot"][name]["share_kept"] for k in full])
        rows.append((name, rk[name], vals["auc"], 100 * vals["bacc"], vals["ce"], sk, la))
    lines += ["## One-shot repair", "", "| candidate | mean rank | ROC-AUC | bal. acc. | log loss | share of P | label acc |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r:.2f} | {u:.3f} | {b:.1f} | {e:.3f} | {s:.2f} | {la:.3f} |" for n, r, u, b, e, s, la in
              sorted(rows, key=lambda t: t[1])]
    # the greedy paths: every recorded state of both selections
    path = {}
    for k in full:
        n = recs[(*k, "global")]["n"]
        cands = {}
        for sel in SELECTS:
            for st in recs[(*k, sel)]["states"][1:]:
                f = round(st["step"] / 100, 2)
                cands[f"{sel}@{f:g}"] = st["test"] | {"share_kept": st["accepted"] / n, "label_acc": st["label_acc"]}
        path[k] = cands
    rk = mean_ranks({k: {n: {m: v[m] for m, _ in METRICS} for n, v in c.items()} for k, c in path.items()})
    rows = []
    for name in rk:
        vals = pd.DataFrame([path[k][name] for k in full if name in path[k]]).mean()
        rows.append((name, rk[name], vals["auc"], 100 * vals["bacc"], vals["ce"], vals["share_kept"], vals["label_acc"]))
    lines += ["", "## Greedy repair: stop (share of P labelled) x selection", "",
              "| candidate | mean rank | ROC-AUC | bal. acc. | log loss | share of P | label acc |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r:.2f} | {u:.3f} | {b:.1f} | {e:.3f} | {s:.2f} | {la:.3f} |" for n, r, u, b, e, s, la in
              sorted(rows, key=lambda t: t[1])]
    # references
    vo = pd.DataFrame([recs[(*k, "global")]["v_only"] for k in full]).mean()
    lines += ["", "## References (means over the same draws)", "", "| entry | ROC-AUC | bal. acc. | log loss |", "|---|---|---|---|",
              f"| V only (CPU, these runs) | {vo['auc']:.3f} | {100 * vo['bacc']:.1f} | {vo['ce']:.3f} |"]
    for att in T.ATTACKS:
        for e in ("v_plus_p", "oracle"):
            m = pd.DataFrame([refs[(*k, att)][e] for k in full]).mean()
            lines.append(f"| {e} {att} (GPU) | {m['auc']:.3f} | {100 * m['bacc']:.1f} | {m['ce']:.3f} |")
    # per dataset: the all-of-P states and V only
    lines += ["", "## Per dataset (means over c and seeds): V only, one-shot top100 / top30, greedy global@1 / @0.3", "",
              "| dataset | V only AUC / CE | one-shot top1 AUC / CE | one-shot top0.3 AUC / CE | greedy global@1 AUC / CE | greedy global@0.3 AUC / CE | V + P rand45 AUC / CE | V + P hub30 AUC / CE |",
              "|---|---|---|---|---|---|---|---|"]
    for ds in ORDER:
        ks = [k for k in full if k[0] == ds]
        if not ks:
            continue
        def mm(get):  # noqa: E306
            v = pd.DataFrame([get(k) for k in ks]).mean()
            return f"{v['auc']:.3f} / {v['ce']:.3f}"
        lines.append(f"| {ds} | {mm(lambda k: recs[(*k, 'global')]['v_only'])} | {mm(lambda k: one[k]['top1'])} | "
                     f"{mm(lambda k: one[k]['top0.3'])} | {mm(lambda k: path[k].get('global@1'))} | "
                     f"{mm(lambda k: path[k].get('global@0.3'))} | {mm(lambda k: refs[(*k, 'rand45')]['v_plus_p'])} | "
                     f"{mm(lambda k: refs[(*k, 'hub30')]['v_plus_p'])} |")
    out = Path("results/tables/vp_tune/report_repair.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "run", "status", "report"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--units", type=Path, default=None, help="lines '<dataset> <c> <seed> <select>', run in this order")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--scripts", type=int, default=7, help="plan: the number of GPU scripts")
    ap.add_argument("--out", type=Path, default=Path("scripts/vp_tune_repair_jobs"), help="plan: their folder")
    a = ap.parse_args(argv)
    {"plan": plan, "run": run, "status": status, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plan the V + P benchmark (scripts/bench_vp.py) as N bash scripts for GPUs on machines that do not share a disk.

Jobs: the 180 instances (6 datasets x rand45 / hub30 x c = 1, 3, 5 x seeds 0-4) x bench_vp.JOBS (refs, vfilter,
greedy, baselines, sigmoid, m1, greedy_thr, vrepair, grepair), only those without a result. Cost model (seconds on an
H100, measured: COST below) per dataset: sigmoid 70 and m1 400 iterations at the dataset's seconds per iteration
(records included), greedy its measured run (greedy_thr and grepair scaled from it), and the one-shot jobs a few
seconds. Jobs are packed longest first onto the least-loaded script; each script runs its cheapest jobs first.

Each gpu_XX.sh takes the GPU index as its argument, checks the machine first (scripts/attn_form_jobs/check_remote.py
with inputs.sha256: the datasets, the 180 draws and the benchmark code byte-identical to this machine's, the model
checkpoint, the TabPFN copy, a numeric preflight), loads .env for W&B and stops unless W&B accepts the key, then runs
its list twice (the second pass retries failures); finished jobs are skipped, so rerunning resumes. merge.sh brings a
machine's results back (locks and temporary files left out).

    python scripts/plan_vp.py --scripts 10 [--out scripts/vp_jobs] [--wandb-group vp_bench]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_tinyv as B  # noqa: E402
import bench_vp as V  # noqa: E402
from plan_soft_m1 import sha256  # noqa: E402

# seconds on an H100 (smoke runs and the earlier studies): per iteration of sigmoid / m1, records included; greedy's
# whole run; refs + vfilter + baselines together
COST = {"gas-drift": {"iter": 6.5, "greedy": 60.0, "oneshot": 25.0},
        "coil2000_insurance_policies": {"iter": 4.3, "greedy": 50.0, "oneshot": 22.0},
        "heloc": {"iter": 1.4, "greedy": 20.0, "oneshot": 15.0},
        "students_dropout": {"iter": 0.8, "greedy": 13.0, "oneshot": 12.0},
        "ecommerce_shipping": {"iter": 0.7, "greedy": 10.0, "oneshot": 12.0},
        "is_this_a_good_customer": {"iter": 0.15, "greedy": 3.0, "oneshot": 8.0}}
COPY = 15.0  # a repair job that copies its twin's result: loading, the train-side V score and W&B

LAUNCHER = """#!/bin/bash
# The V + P benchmark (scripts/bench_vp.py; planned by scripts/plan_vp.py), script {i:02d} of {N}: {n_jobs} jobs, about
# {hours:.1f} GPU-hours (H100). Run from a clone of this repository with results/bench_vp/draws, datasets/ and the
# preflight inputs (results/attn_form/preflight_reference, results/bench_largev/draws) copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py), and unless W&B accepts the API key in .env. Each job is
# one W&B run, named <job>_<instance>, in project TFM_experiments, group {group}, with its result files as an artifact.
# Rerunning resumes; logs in {logs}/. One script per GPU, with {memory}.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p {logs}
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > {logs}/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" {logs}/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
. scripts/load_env.sh "$PWD/.env"  # WANDB_API_KEY from this clone's .env (a variable already set wins)
$PY -c "import wandb; api = wandb.Api(timeout=60); api.viewer; print('  wandb: logged in, entity', api.default_entity)" \\
  || {{ echo "  wandb: NOT logged in. Put WANDB_API_KEY in .env here (or export it, or run wandb login)"; exit 1; }}
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_vp.py run --device "cuda:$GPU" --job-list {out}/jobs_{i:02d}.txt --wandb-mode online \\
    --wandb-group {group} >> {logs}/gpu_{i:02d}.log 2>&1
done
$PY scripts/bench_vp.py status --job-list {out}/jobs_{i:02d}.txt
grep -h "FAILED" {logs}/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""
MERGE = """#!/bin/bash
# Bring one machine's finished V + P benchmark runs ({out}/gpu_XX.sh) into {runs}; files already here are kept as they
# are, and the locks and temporary files of jobs still running there are left out.
#   bash {out}/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/{up}"
count() {{ find {runs} -name '*.json' | wc -l; }}
mkdir -p {runs}
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${{1%/}}/{runs}/" {runs}/
echo "merged $(( $(count) - before )) new result files"
"""
CODE = ["scripts/bench_vp.py", "scripts/vp_tune.py", "scripts/bench_tinyv.py", "scripts/attn_learn.py",
        "scripts/attn_form_study.py", "scripts/tinyv_learned.py", "scripts/small_v_methods.py", "scripts/bench_small_v.py",
        "scripts/schedule_study.py", "scripts/gate_context.py", "scripts/tabfm_experiments/row_weights.py"]


def cost(tag: str, job: str) -> float:
    ds, attack, c, seed = V.parse_tag(tag)
    cc = COST[ds]
    if job == "sigmoid":
        return V.SIGMOID["steps"] * cc["iter"] + 30.0
    if job == "m1":
        return V.M1["steps"] * cc["iter"] + 30.0
    if job == "greedy":
        return cc["greedy"] * (0.8 if attack == "rand45" else 1.0)  # rand45 stops at 65 steps, hub30 at 80
    if job == "greedy_thr":  # stops near the collapse (about 50-57% of P); a step costs in proportion to the context
        return cc["greedy"] * 0.5
    if job == "grepair":  # to V.REPAIR's stop, from about 80% for greedy's whole run: cost grows with its square
        return cc["greedy"] * (V.REPAIR["greedy"]["until"] / 0.8) ** 2 * 1.2
    return cc["oneshot"] / 3 + 10.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("scripts/vp_jobs"))
    ap.add_argument("--wandb-group", default="vp_bench")
    a = ap.parse_args(argv)
    OUT = a.out
    jobs = [(t, j) for t in V.instances() for j in V.JOBS if not (V.ROOT / "runs" / t / f"{j}.json").exists()]
    # a repair job and its other poisoning's twin go to the same script, rand45 first: the second copies the first
    # (bench_vp.repair_twin), which needs both on one disk
    groups: dict = {}
    for t, j in jobs:
        ds, attack, c, seed = V.parse_tag(t)
        groups.setdefault((ds, c, seed, j) if j in V.REPAIR_JOBS else (t, j), []).append((t, j))
    units = [(cost(*g[0]) + COPY * (len(g) - 1), g) for g in groups.values()]
    units.sort(key=lambda u: -u[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in units:  # longest first onto the least-loaded script
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += u[0]
    # each script runs its cheapest jobs first
    bins = [[(cost(t, j), t, j) for _, g in sorted(b, key=lambda u: u[0]) for t, j in g] for b in bins]
    attention = any(j in ("sigmoid", "m1") for _, j in jobs)
    memory = ("about 47 GiB free: a gas-drift attention job peaks at 46 GiB" if attention else
              "about 4 GiB free: these jobs are TabPFN inference only (gas-drift peaks at 2.6 GiB)")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("gpu_*.sh")) + list(OUT.glob("jobs_*.txt")):
        old.unlink()
    up = "/".join([".."] * len(OUT.parts))
    logs = f"logs/{OUT.name}"
    lines = ["| script | jobs | GPU-hours (est., H100) | sigmoid / m1 jobs (gas-drift, coil2000) |", "|---|---|---|---|"]
    for i, b in enumerate(bins):
        (OUT / f"jobs_{i:02d}.txt").write_text("".join(f"{t} {j}\n" for _, t, j in b))
        sh = OUT / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=OUT.as_posix(), up=up,
                                      logs=logs, group=a.wandb_group, memory=memory))
        os.chmod(sh, 0o755)
        heavy = [(t, j) for _, t, j in b if j in ("sigmoid", "m1") and V.parse_tag(t)[0] in ("gas-drift", "coil2000_insurance_policies")]
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.2f} | {len(heavy)} |")
    runs = (V.ROOT / "runs").as_posix()
    merge = OUT / "merge.sh"
    merge.write_text(MERGE.format(out=OUT.as_posix(), up=up, runs=runs))
    os.chmod(merge, 0o755)
    files = list(CODE)
    for ds in V.DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
    files += [f"{V.ROOT}/draws/{t}.npz" for t in V.instances()]
    (OUT / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{len(jobs)} jobs, about {load.sum() / 3600:.1f} H100-hours in all, {load.max() / 3600:.2f} on the busiest "
              f"script (on a GPU that another busy job shares, two to four times that). Each script needs {memory}."]
    print("\n".join(lines))
    (OUT / "split.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

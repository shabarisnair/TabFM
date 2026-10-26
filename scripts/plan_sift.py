#!/usr/bin/env python
"""Plan the tiny-V runs of the sifting methods (bench_tinyv.py metasift / tabsift / tabsift_m1; tinyv_sift.py) at R = 30%,
as N bash scripts for GPUs on machines that do not share a disk: 6 datasets x V = 1, 3, 5 rows per class x the methods.

Cost model (H100-seconds, from the benchmark's per-update cost bench_largev.U): tabsift and tabsift_m1 take 100 full
steps per configuration, a step being one refit of the context with every poisoned row as a query (about 2 U; measured
on heloc and coil2000), plus their recording (every 2 steps, 3 test evaluations of about 0.3 U; tabsift_m1 also the
hard split's objective, one forward with the held-out rows as the queries, about 0.6 U); metasift about 3 minutes for
its 4 configurations plus the default's recording (about 100 records of 3 test evaluations on 5,000 rows). Jobs are
packed longest first onto the least-loaded list; an ensemble (tinyv_sift.ENSEMBLES) goes with its members, after them,
into one list, and each list runs its cheapest instances first. Each gpu_XX.sh takes the GPU index as its argument, checks the machine
first (scripts/attn_form_jobs/check_remote.py with inputs.sha256: the draws, data, the sift code and, for metasift, the
Meta-Sift code byte-identical to this machine's, the model checkpoint, the TabPFN copy, a numeric preflight) and then
runs its list; finished jobs are skipped, so rerunning resumes. With --wandb the scripts also load .env
(scripts/load_env.sh: WANDB_API_KEY), stop unless W&B accepts the key, and run with --wandb-mode online (one W&B run
per job, <method>_<instance>, in project TFM_experiments, group tinyv_<method>). merge.sh brings a machine's results
back.

    python scripts/plan_sift.py [--scripts 9]                                                   # -> scripts/sift_jobs/
    python scripts/plan_sift.py --methods tabsift_m1 --scripts 5 --out scripts/sift_m1_jobs --wandb
    python scripts/plan_sift.py --methods tabsift_m1_rand_s0 tabsift_m1_rand_s1 tabsift_m1_rand_s2 tabsift_m1_rand_s3 \
        tabsift_m1_rand_s4 tabsift_m1_rand_ens --scripts 4 --out scripts/sift_m1_rand_jobs --wandb \
        --wandb-group tinyv_tabsift_m1_rand_v2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_largev as BL  # noqa: E402
import bench_tinyv as B  # noqa: E402
import tinyv_sift as T  # noqa: E402
from plan_soft_m1 import sha256  # noqa: E402

LAUNCHER = """#!/bin/bash
# Sifting methods on the tiny-V benchmark at R = 30% (scripts/plan_sift.py), script {i:02d} of {N}: {n_jobs} jobs, about
# {hours:.1f} GPU-hours (H100). Run from a clone of this repository with results/, datasets/ and Meta-Sift/ copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in {logs}/.{wandb_note}
# One script per GPU: two gas-drift or coil2000 jobs on one GPU run out of memory.
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
{wandb_check}for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list {out}/tinyv_{i:02d}.txt{wandb_flag} >> {logs}/gpu_{i:02d}.log 2>&1
done
$PY scripts/bench_tinyv.py status --job-list {out}/tinyv_{i:02d}.txt
grep -h "FAILED\\|SKIPPING" {logs}/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""
WANDB_CHECK = """. scripts/load_env.sh "$PWD/.env"  # WANDB_API_KEY from this clone's .env (a variable already set wins)
$PY -c "import wandb; api = wandb.Api(timeout=60); api.viewer; print('  wandb: logged in, entity', api.default_entity)" \\
  || { echo "  wandb: NOT logged in. Put WANDB_API_KEY in .env here (or export it, or run wandb login)"; exit 1; }
"""
MERGE = """#!/bin/bash
# Bring one machine's finished sifting runs ({out}/gpu_XX.sh: {methods}) into {runs};
# files already here are kept as they are. The locks and temporary files of jobs still running there are left out (a
# copied lock would make every run here skip that job). Then: python scripts/plot_bench_tinyv.py
#   bash {out}/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/{up}"
count() {{ find {runs} {names} | wc -l; }}
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' --include '*/' {includes} --exclude '*' \\
  "${{1%/}}/{runs}/" {runs}/
echo "merged $(( $(count) - before )) new result files"
"""
CODE = ["scripts/tinyv_sift.py", "scripts/bench_tinyv.py", "scripts/tabfm_experiments/row_weights.py"]


def cost(ds: str, method: str) -> float:
    U = BL.U[ds]
    if method == "metasift":
        return 180.0 + 90.0 * U
    if method in T.ENSEMBLES:  # the members' votes deployed (6 + 3 reference test evaluations), one hard objective
        return 9 * 0.3 * U + 0.6 * U + 30.0
    cfgs = [T.TABSIFT | cfg for cfg in T.GRIDS[method].values()]
    steps = sum(c["steps"] + 1 for c in cfgs)
    records = sum(c["steps"] // c["record_every"] + 1 for c in cfgs)
    hard = 0.6 if cfgs[0]["param"] == "topk" else 0.0
    return steps * 2.0 * U + records * (3 * 0.3 + hard) * U + 60.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", type=int, default=9)
    ap.add_argument("--methods", nargs="+", choices=B.SIFT, default=["metasift", "tabsift"])
    ap.add_argument("--out", type=Path, default=Path("scripts/sift_jobs"))
    ap.add_argument("--wandb", action="store_true", help="log every job to W&B (online), after checking the login")
    ap.add_argument("--wandb-group", default=None, help="the W&B group of every job (default: tinyv_<method>)")
    a = ap.parse_args(argv)
    OUT = a.out
    # a unit is the jobs of one instance that must run on one machine, in order: an ensemble's members, then it
    ens = [m for m in a.methods if m in T.ENSEMBLES]
    missing = {mm for m in ens for mm, _ in T.ENSEMBLES[m]} - set(a.methods)
    if missing:
        raise SystemExit(f"an ensemble needs its members planned too: {sorted(missing)}")
    groups = [[mm for mm, _ in T.ENSEMBLES[m]] + [m] for m in ens]
    groups += [[m] for m in a.methods if not any(m in g for g in groups)]
    units = []
    for ds in B.DATASETS:
        for c in B.CS:
            tag = B.tag_of(ds, 30.0, c, 0)
            for g in groups:
                jobs = [m for m in g if not (B.ROOT / "runs" / tag / f"{m}.json").exists()]
                if jobs:
                    units.append((sum(cost(ds, m) for m in jobs), [(tag, m) for m in jobs]))
    units.sort(key=lambda u: -u[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in units:  # longest first onto the least-loaded script
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += u[0]
    bins = [sorted(b, key=lambda u: u[0]) for b in bins]  # each script runs its cheapest units first
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("gpu_*.sh")) + list(OUT.glob("tinyv_*.txt")):
        old.unlink()
    up = "/".join([".."] * len(OUT.parts))
    logs = f"logs/{OUT.name}"
    group = a.wandb_group or "tinyv_<method>"
    wandb = dict(wandb_note="\n# W&B: it also stops unless W&B accepts the API key in .env. Each job is one W&B run, named "
                            f"<method>_<instance>,\n# in project TFM_experiments, group {group}, with its result files as an "
                            "artifact." if a.wandb else "",
                 wandb_check=WANDB_CHECK if a.wandb else "",
                 wandb_flag=(" --wandb-mode online" + (f" --wandb-group {a.wandb_group}" if a.wandb_group else "")) if a.wandb else "")
    lines = ["| script | jobs | GPU-hours (est., H100) | jobs |", "|---|---|---|---|"]
    for i, b in enumerate(bins):
        jobs = [j for _, u in b for j in u]
        (OUT / f"tinyv_{i:02d}.txt").write_text("".join(f"{tag} {job}\n" for tag, job in jobs))
        sh = OUT / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(jobs), hours=load[i] / 3600, out=OUT.as_posix(), up=up,
                                      logs=logs, **wandb))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(jobs)} | {load[i] / 3600:.2f} | "
                     + ", ".join(f"{u[0][0]} ({', '.join(m for _, m in u)})" for _, u in b) + " |")
    runs = (B.ROOT / "runs").as_posix()
    merge = OUT / "merge.sh"
    merge.write_text(MERGE.format(out=OUT.as_posix(), up=up, runs=runs, methods=", ".join(a.methods),
                                  names=" -o ".join(f"-name '{m}.json'" for m in a.methods),
                                  includes=" ".join(f"--include '{m}.*'" for m in a.methods)))
    os.chmod(merge, 0o755)
    files = ["Meta-Sift/meta_sift/meta_sift_func.py", "Meta-Sift/meta_sift/util.py",
             "Meta-Sift/meta_sift/poi_util.py"] if "metasift" in a.methods else []
    files += CODE
    for ds in B.DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
        files += [f"{B.ROOT}/draws/{B.tag_of(ds, 30.0, c, 0)}.npz" for c in B.CS]
    (OUT / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{sum(len(u) for _, u in units)} jobs, about {load.sum() / 3600:.1f} H100-hours in all, {load.max() / 3600:.2f} on the busiest "
              "script (on a GPU that another busy job shares, two to four times that)."]
    print("\n".join(lines))
    (OUT / "split.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

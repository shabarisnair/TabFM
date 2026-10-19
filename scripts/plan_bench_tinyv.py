#!/usr/bin/env python
"""Split the tiny-V benchmark's remaining jobs over N GPU scripts (bench_tinyv.py).

Every (instance, method) job without a result in <root>/runs gets an estimated cost: its share of one H100
in minutes. The estimates come from timings measured with about 3 jobs per H100: the schedule study's greedy
steps, the attention study's updates, the NaN-gate night and the earlier benchmark's grids. A job's share is
its measured time / 2.8, the effective jobs per GPU measured in the schedule study. The jobs are packed into
N lists, longest job first onto the least-loaded list. Each list gets a launcher that runs WORKERS (3)
workers on one GPU. The workers claim the list's jobs through lock files, longest first; a baseline computes
its configuration grid inside its job.

    python scripts/plan_bench_tinyv.py --scripts 10      # -> scripts/bench_tinyv_jobs/gpu_XX.{txt,sh} + README.md
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import bench_tinyv as B  # noqa: E402

CONC = 2.8
# seconds per greedy scoring step (one TabPFN call over the pool), measured with about 3 jobs per H100
STEP = {"gas-drift": 212.4 / 80, "coil2000_insurance_policies": 139.2 / 80, "heloc": 51.0 / 80, "students_dropout": 33.0 / 80}
STEP["ecommerce_shipping"] = 5000 ** 2 * (0.232 + 0.01774 * 10) / 25e6        # fitted n^2 (a + b * features)
STEP["is_this_a_good_customer"] = 1398 ** 2 * (0.232 + 0.01774 * 13) / 25e6 + 0.05
# seconds per attention-weight update (the attention study, its evaluation share removed), scaled by STEP
_ATTN = {"coil2000_insurance_policies": 8.4, "heloc": 2.5, "students_dropout": 1.35}
_ATTN["gas-drift"] = _ATTN["coil2000_insurance_policies"] * STEP["gas-drift"] / STEP["coil2000_insurance_policies"]
for _d in ("ecommerce_shipping", "is_this_a_good_customer"):
    _ATTN[_d] = _ATTN["heloc"] * STEP[_d] / STEP["heloc"]
# the earlier benchmark's whole-grid seconds per instance (credit-approval stands in for good_customer x 3,
# heloc for ecommerce, heloc x 1.5 for coil2000)
_GRID = {  # method: (credit-approval, gas-drift, heloc, students_dropout)
    "knn_sanitize": (2, 2, 2, 2), "label_spread": (5, 13, 12, 8), "knn_shapley": (0, 0, 0, 0), "data_oob": (2, 48, 6, 4),
    "cleanlab": (32, 33, 26, 26), "simifeat": (3, 8, 8, 7), "glc": (3, 67, 37, 29), "mwnet": (5, 72, 57, 38),
    "emlc": (9, 125, 93, 66), "ieg": (7, 106, 97, 54)}
MODES = {"knn_sanitize": 3, "label_spread": 3, "knn_shapley": 1, "data_oob": 1, "cleanlab": 2.3, "simifeat": 1,
         "glc": 3, "mwnet": 1, "emlc": 3, "ieg": 3}


def grid_seconds(method: str, ds: str) -> float:
    cr, gas, heloc, stu = _GRID[method]
    return {"gas-drift": gas, "heloc": heloc, "students_dropout": stu, "ecommerce_shipping": heloc,
            "coil2000_insurance_policies": 1.5 * heloc, "is_this_a_good_customer": 3 * cr}[ds]


def cost_minutes(ds: str, n: int, k: int, method: str) -> float:
    """Estimated share of one H100 (minutes)."""
    st = STEP[ds]
    if method == "greedy_m1":
        sec = (n - k) * st
    elif method == "greedy_pct1":
        sec = math.ceil((n - k) / math.ceil(0.01 * n)) * st
    elif method == "greedy_pct10":
        sec = math.ceil((n - k) / math.ceil(0.10 * n)) * st
    elif method == "vfilter":
        sec = 2 * st
    elif method == "refs":
        sec = 3 * st
    elif method in ("attn_topk", "attn_soft"):
        sec = 20 * _ATTN[ds] + 3 * st
    elif method == "nan_topk":
        sec = 4 * 20 * 1.6 * st + 3 * st
    else:  # a baseline: its grid, the V-scores of its distinct cleaned sets, three test scores
        n_cfg = len(B.baseline_configs(method))
        sec = grid_seconds(method, ds) * CONC + 0.75 * n_cfg * MODES[method] * 0.7 * st + 3 * st
    return sec / CONC / 60


LAUNCHER = """#!/usr/bin/env bash
# Tiny-V benchmark: GPU script {i:02d} of {N} -- {n_jobs} jobs, about {hours:.1f} H100-hours (estimate).
# Its longest job: {longest} (about {longest_min:.0f} H100-minutes). See scripts/bench_tinyv_jobs/README.md.
#
#   GPU=<device index> bash {out}/gpu_{i:02d}.sh
#
#   GPU      the CUDA device (default 0: the first one CUDA_VISIBLE_DEVICES exposes)
#   WORKERS  parallel workers on that GPU (default 3)
#   PY       the python of the tabfm environment
# Needs results/bench_tinyv/draws/ (the shared draws). Finished jobs are skipped, so it can simply be
# started again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/{up}"
PY=${{PY:-{python}}}
GPU=${{GPU:-0}}
WORKERS=${{WORKERS:-3}}
LIST={out}/gpu_{i:02d}.txt
mkdir -p {logs}
export PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}
for w in $(seq 1 "$WORKERS"); do
  "$PY" -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list "$LIST" --grid-if-missing --threads 4 \\
      > "{logs}/gpu_{i:02d}_w$w.log" 2>&1 &
  sleep 20
done
wait
# one worker alone retries whatever failed above (e.g. out of GPU memory beside the other workers)
"$PY" -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list "$LIST" --grid-if-missing --threads 4 \\
    > "{logs}/gpu_{i:02d}_retry.log" 2>&1 || true
"$PY" scripts/bench_tinyv.py status --job-list "$LIST"
grep -h "FAILED\\|SKIPPING" {logs}/gpu_{i:02d}_*.log || true
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("scripts/bench_tinyv_jobs"), help="relative to the repository root")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    todo = []
    for ds in B.DATASETS:
        for c in B.CS:
            for R in B.RS:
                for s in B.SEEDS:
                    tag = B.tag_of(ds, R, c, s)
                    z = np.load(B.ROOT / "draws" / f"{tag}.npz")
                    n, k = len(z["ctx_rows"]), int(z["k"])
                    for m in B.METHODS:
                        if not (B.ROOT / "runs" / tag / f"{m}.json").exists():
                            todo.append((cost_minutes(ds, n, k, m), tag, m))
    todo.sort(key=lambda t: (-t[0], t[1], t[2]))
    bins = [[] for _ in range(a.scripts)]
    load = np.zeros(a.scripts)
    for job in todo:  # longest processing time first, onto the least-loaded script
        i = int(np.argmin(load))
        bins[i].append(job)
        load[i] += job[0]
    a.out.mkdir(parents=True, exist_ok=True)
    for old in list(a.out.glob("gpu_*.txt")) + list(a.out.glob("gpu_*.sh")):
        old.unlink()
    rows = []
    for i, jobs in enumerate(bins):
        jobs.sort(key=lambda t: (-t[0], t[1], t[2]))  # longest first within the script
        (a.out / f"gpu_{i:02d}.txt").write_text(
            f"# GPU script {i:02d}: {len(jobs)} jobs, about {load[i] / 60:.1f} H100-hours; '<instance> <method>', "
            "longest first\n" + "".join(f"{tag} {m}\n" for _, tag, m in jobs))
        sh = a.out / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(jobs), hours=load[i] / 60, longest=f"{jobs[0][1]} {jobs[0][2]}",
                                      longest_min=jobs[0][0], python=a.python, out=a.out.as_posix(),
                                      up="/".join([".."] * len(a.out.parts)), logs=f"logs/bench_tinyv/{a.out.name}"))
        os.chmod(sh, 0o755)
        by_m = {}
        for _, _, m in jobs:
            by_m[m] = by_m.get(m, 0) + 1
        rows.append((i, len(jobs), load[i] / 60, jobs[0], by_m))
    lines = ["| script | jobs | H100-hours (est.) | longest job | jobs per method |", "|---|---|---|---|---|"]
    for i, nj, h, (lm, lt, lmeth), by_m in rows:
        lines.append(f"| gpu_{i:02d}.sh | {nj} | {h:.1f} | {lt} {lmeth} ({lm:.0f} min) | "
                     + ", ".join(f"{m} {c}" for m, c in sorted(by_m.items(), key=lambda t: B.METHODS.index(t[0]))) + " |")
    print("\n".join(lines))
    print(f"\n{len(todo)} jobs, about {load.sum() / 60:.1f} H100-hours in all, {load.max() / 60:.1f} on the busiest script")
    (a.out / "split.md").write_text("\n".join(lines) + f"\n\n{len(todo)} jobs, about {load.sum() / 60:.1f} H100-hours in all.\n")


if __name__ == "__main__":
    main()

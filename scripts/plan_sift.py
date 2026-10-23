#!/usr/bin/env python
"""Plan the tiny-V runs of the two sifting methods (bench_tinyv.py metasift / tabsift; tinyv_sift.py) at R = 30%, as N
bash scripts for GPUs on machines that do not share a disk: 6 datasets x V = 1, 3, 5 rows per class x 2 methods.

Cost model (H100-seconds, from the benchmark's per-update cost bench_largev.U): tabsift takes 2 starts x 100 full
steps, a step being one refit of the context with every poisoned row as a query (about 2 U; measured on heloc and
coil2000), plus its recording (every 2 steps, 3 test evaluations of about 0.3 U); metasift about 3 minutes for its 4
configurations plus the default's recording (about 100 records of 3 test evaluations on 5,000 rows). Jobs are packed
longest first onto the least-loaded list. Each
gpu_XX.sh takes the GPU index as its argument, checks the machine first (scripts/attn_form_jobs/check_remote.py with
inputs.sha256: the draws, data and Meta-Sift code byte-identical to this machine's, the model checkpoint, the code, a
numeric preflight) and then runs its list; finished jobs are skipped, so rerunning resumes.

    python scripts/plan_sift.py [--scripts 9]      # -> scripts/sift_jobs/
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

OUT = Path("scripts/sift_jobs")
LAUNCHER = """#!/bin/bash
# Sifting methods on the tiny-V benchmark at R = 30% (scripts/plan_sift.py), script {i:02d} of {N}: {n_jobs} jobs, about
# {hours:.1f} GPU-hours (H100). Run from a clone of this repository with results/, datasets/ and Meta-Sift/ copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/sift_jobs/.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/sift_jobs
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > logs/sift_jobs/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/sift_jobs/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list {out}/tinyv_{i:02d}.txt >> logs/sift_jobs/gpu_{i:02d}.log 2>&1
done
$PY scripts/bench_tinyv.py status --job-list {out}/tinyv_{i:02d}.txt
grep -h "FAILED\\|SKIPPING" logs/sift_jobs/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""


def cost(ds: str, method: str) -> float:
    U = BL.U[ds]
    if method == "metasift":
        return 180.0 + 90.0 * U
    steps = sum(cfg.get("steps", T.TABSIFT["steps"]) + 1 for cfg in T.GRIDS["tabsift"].values())
    records = sum(cfg.get("steps", T.TABSIFT["steps"]) // T.TABSIFT["record_every"] + 1 for cfg in T.GRIDS["tabsift"].values())
    return steps * 2.0 * U + records * 3 * 0.3 * U + 60.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts", type=int, default=9)
    a = ap.parse_args(argv)
    units = [(cost(ds, m), B.tag_of(ds, 30.0, c, 0), m) for ds in B.DATASETS for c in B.CS for m in B.SIFT
             if not (B.ROOT / "runs" / B.tag_of(ds, 30.0, c, 0) / f"{m}.json").exists()]
    units.sort(key=lambda u: -u[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in units:
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += u[0]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("gpu_*.sh")) + list(OUT.glob("tinyv_*.txt")):
        old.unlink()
    lines = ["| script | jobs | GPU-hours (est., H100) | jobs |", "|---|---|---|---|"]
    for i, b in enumerate(bins):
        (OUT / f"tinyv_{i:02d}.txt").write_text("".join(f"{tag} {job}\n" for _, tag, job in b))
        sh = OUT / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=OUT.as_posix(),
                                      up="/".join([".."] * len(OUT.parts))))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.2f} | " + ", ".join(f"{t} {j}" for _, t, j in b) + " |")
    files = ["Meta-Sift/meta_sift/meta_sift_func.py", "Meta-Sift/meta_sift/util.py", "Meta-Sift/meta_sift/poi_util.py"]
    for ds in B.DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
        files += [f"{B.ROOT}/draws/{B.tag_of(ds, 30.0, c, 0)}.npz" for c in B.CS]
    (OUT / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{len(units)} jobs, about {load.sum() / 3600:.1f} H100-hours in all, {load.max() / 3600:.2f} on the busiest "
              "script (on a GPU that another busy job shares, two to four times that)."]
    print("\n".join(lines))
    (OUT / "split.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

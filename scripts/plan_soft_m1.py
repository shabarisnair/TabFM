#!/usr/bin/env python
"""Plan the R = 30% runs that put the best soft form (and, at a new update count, M1) into both benchmarks, as N
bash scripts for GPUs on machines that do not share a disk.

  larger-V  attn_wsoft (bench_largev.py: w / V-filter / PGD lr 0.8, attn_form_study's held-out early stopping) at
            V = 50-400 (V = 500 comes from the attention-form study: bench_largev.py import-form)
  tiny-V    attn_wsoft_it<S> (bench_tinyv.py; S updates on all of V, scripts/tinyv_iter_study.py's count) and, when
            M1's count M is not tinyv_learned's 20, attn_topk_it<M>
Jobs are packed longest first onto the least-loaded list (bench_largev's cost model). Each gpu_XX.sh takes the GPU
index as its argument, checks the machine first (scripts/attn_form_jobs/check_remote.py with inputs.sha256: the draws
and data byte-identical to this machine's, the model checkpoint, the code, a numeric preflight) and then runs its
lists; finished jobs are skipped, so rerunning resumes.

    python scripts/plan_soft_m1.py --soft-iters S [--soft-lr 0.8] --m1-iters M [--scripts 8]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_largev as BL  # noqa: E402
import bench_tinyv as B  # noqa: E402

OUT = Path("scripts/soft_m1_jobs")
LARGE_VS = (50, 100, 200, 300, 400)
LAUNCHER = """#!/bin/bash
# Soft-form / M1 runs at R = 30% (scripts/plan_soft_m1.py), script {i:02d} of {N}: {n_jobs} jobs, about {hours:.1f} GPU-hours
# (H100). Run from a clone of this repository with results/ copied over, on any free GPU:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/soft_m1_jobs/.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/soft_m1_jobs
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > logs/soft_m1_jobs/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/soft_m1_jobs/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  [ -s {out}/largev_{i:02d}.txt ] && $PY -u scripts/bench_largev.py run --device "cuda:$GPU" \\
    --job-list {out}/largev_{i:02d}.txt >> logs/soft_m1_jobs/gpu_{i:02d}.log 2>&1
  [ -s {out}/tinyv_{i:02d}.txt ] && $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" \\
    --job-list {out}/tinyv_{i:02d}.txt >> logs/soft_m1_jobs/gpu_{i:02d}.log 2>&1
done
[ -s {out}/largev_{i:02d}.txt ] && $PY scripts/bench_largev.py status --job-list {out}/largev_{i:02d}.txt
grep -h "FAILED\\|SKIPPING" logs/soft_m1_jobs/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soft-iters", type=int, required=True)
    ap.add_argument("--soft-lr", type=float, default=0.8, help="tiny-V only (larger-V keeps the form study's 0.8)")
    ap.add_argument("--m1-iters", type=int, required=True)
    ap.add_argument("--scripts", type=int, default=8)
    a = ap.parse_args(argv)
    units = []  # (estimated H100-seconds, benchmark, tag, job)
    for ds in BL.DATASETS:
        for V in LARGE_VS:
            tag = BL.tag_of(ds, 30.0, V, 0)
            if not (BL.ROOT / "runs" / tag / "attn_wsoft.json").exists():
                units.append((BL.UPDATES["attn_wsoft"] * BL.U[ds] * ((BL.N_CTX[ds] - V) / BL.N_CTX[ds]) ** 2, "largev", tag,
                              "attn_wsoft"))
        soft = f"attn_wsoft_it{a.soft_iters}" if a.soft_lr == 0.8 else f"attn_wsoft_lr{a.soft_lr:g}_it{a.soft_iters}"
        tiny = [soft] + ([f"attn_topk_it{a.m1_iters}"] if a.m1_iters != 20 else [])
        for c in B.CS:
            tag = B.tag_of(ds, 30.0, c, 0)
            for job in tiny:
                if not (B.ROOT / "runs" / tag / f"{job}.json").exists():
                    steps = int(job.rsplit("it", 1)[1])
                    units.append(((steps + 5) * BL.U[ds], "tinyv", tag, job))
    units.sort(key=lambda u: -u[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in units:
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += u[0]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("gpu_*.sh")) + list(OUT.glob("largev_*.txt")) + list(OUT.glob("tinyv_*.txt")):
        old.unlink()
    lines = ["| script | jobs | GPU-hours (est.) | larger-V jobs | tiny-V jobs |", "|---|---|---|---|---|"]
    for i, b in enumerate(bins):
        for bench in ("largev", "tinyv"):
            js = [f"{tag} {job}" for _, bb, tag, job in b if bb == bench]
            (OUT / f"{bench}_{i:02d}.txt").write_text("".join(f"{j}\n" for j in js))
        sh = OUT / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=OUT.as_posix(),
                                      up="/".join([".."] * len(OUT.parts))))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.2f} | {sum(u[1] == 'largev' for u in b)} | "
                     f"{sum(u[1] == 'tinyv' for u in b)} |")
    # the inputs these jobs read, for check_remote.py
    files = []
    for ds in BL.DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
        files += [f"{BL.ROOT}/draws/{BL.tag_of(ds, 30.0, V, 0)}.npz" for V in LARGE_VS]
        files += [f"{B.ROOT}/draws/{B.tag_of(ds, 30.0, c, 0)}.npz" for c in B.CS]
    (OUT / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{len(units)} jobs, about {load.sum() / 3600:.1f} H100-hours in all, {load.max() / 3600:.2f} on the busiest "
              f"script. Tiny-V: soft form {a.soft_iters} updates at lr {a.soft_lr:g}, M1 {a.m1_iters} updates"
              + (" (unchanged: no M1 reruns)." if a.m1_iters == 20 else " (M1 rerun as attn_topk_it" + str(a.m1_iters) + ").")]
    print("\n".join(lines))
    (OUT / "split.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

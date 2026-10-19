#!/usr/bin/env bash
# Larger-V benchmark: GPU script 03 of 8 -- 102 jobs, about 9.0 H100-hours (estimate).
# Its longest job: gas-drift_R10_V100_s0 nan_topk_r0 (about 17 H100-minutes). See scripts/bench_largev_jobs/README.md.
#
#   GPU=<device index> bash scripts/bench_largev_jobs/gpu_03.sh
#
#   GPU      the CUDA device (default 0: the first one CUDA_VISIBLE_DEVICES exposes)
#   WORKERS  parallel workers on that GPU (default 2; gas-drift and coil2000 jobs take about 45 GB each)
#   PY       the python of the tabfm environment
# Needs results/bench_largev/draws/ (the shared draws). Finished jobs are skipped, so it can simply be started
# again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/ssn899/miniforge3/envs/tabfm/bin/python}
GPU=${GPU:-0}
WORKERS=${WORKERS:-2}
LIST=scripts/bench_largev_jobs/gpu_03.txt
mkdir -p logs/bench_largev/bench_largev_jobs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for w in $(seq 1 "$WORKERS"); do
  "$PY" -u scripts/bench_largev.py run --device "cuda:$GPU" --job-list "$LIST" --threads 4 \
      > "logs/bench_largev/bench_largev_jobs/gpu_03_w$w.log" 2>&1 &
  sleep 20
done
wait
# one worker alone retries whatever failed above (e.g. out of GPU memory), then any top-k vote left
"$PY" -u scripts/bench_largev.py run --device "cuda:$GPU" --job-list "$LIST" --threads 4 \
    > "logs/bench_largev/bench_largev_jobs/gpu_03_retry.log" 2>&1 || true
"$PY" -u scripts/bench_largev.py votes --device "cuda:$GPU" --job-list "$LIST" >> "logs/bench_largev/bench_largev_jobs/gpu_03_retry.log" 2>&1 || true
"$PY" scripts/bench_largev.py status --job-list "$LIST"
grep -h "FAILED\|SKIPPING" logs/bench_largev/bench_largev_jobs/gpu_03_*.log || true

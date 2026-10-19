#!/usr/bin/env bash
# Tiny-V benchmark: GPU script 06 of 11 -- 51 jobs, about 2.4 H100-hours (estimate).
# Its longest job: gas-drift_R30_c1_s0 greedy_m1 (about 55 H100-minutes). See README.md in this folder.
#
#   GPU=<device index> bash scripts/bench_tinyv_jobs/gpu_06.sh
#
#   GPU      the CUDA device (default 0: the first one CUDA_VISIBLE_DEVICES exposes)
#   WORKERS  parallel workers on that GPU (default 3)
#   PY       the python of the tabfm environment
# Needs results/bench_tinyv/draws/ (the shared draws). Finished jobs are skipped, so it can simply be
# started again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/ssn899/miniforge3/envs/tabfm/bin/python}
GPU=${GPU:-0}
WORKERS=${WORKERS:-3}
LIST=scripts/bench_tinyv_jobs/gpu_06.txt
mkdir -p logs/bench_tinyv
for w in $(seq 1 "$WORKERS"); do
  "$PY" -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list "$LIST" --grid-if-missing --threads 4 \
      > "logs/bench_tinyv/gpu_06_w$w.log" 2>&1 &
  sleep 20
done
wait
"$PY" scripts/bench_tinyv.py status --job-list "$LIST"

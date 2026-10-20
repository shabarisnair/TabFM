#!/bin/bash
# Bring one machine's finished soft-form / M1 runs (scripts/soft_m1_jobs/gpu_XX.sh) into this repository's results:
# the new run files of both benchmarks; files already here are kept as they are.
#   bash scripts/soft_m1_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
src=${1%/}
for bench in bench_largev bench_tinyv; do
  before=$(find "results/$bench/runs" -name '*.json' | wc -l)
  rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "$src/results/$bench/runs/" "results/$bench/runs/"
  echo "$bench: merged $(( $(find "results/$bench/runs" -name '*.json' | wc -l) - before )) new result files"
done

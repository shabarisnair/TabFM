#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_zero_jobs/gpu_XX.sh: tabsift_zero) into results/bench_tinyv/runs;
# files already here are kept as they are. The locks and temporary files of jobs still running there are left out (a
# copied lock would make every run here skip that job). Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_zero_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_tinyv/runs -name 'tabsift_zero.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' --include '*/' --include 'tabsift_zero.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(count) - before )) new result files"

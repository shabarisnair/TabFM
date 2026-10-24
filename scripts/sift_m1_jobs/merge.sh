#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_m1_jobs/gpu_XX.sh: tabsift_m1) into results/bench_tinyv/runs;
# files already here are kept as they are. Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_m1_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_tinyv/runs -name 'tabsift_m1.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --include '*/' --include 'tabsift_m1.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(count) - before )) new result files"

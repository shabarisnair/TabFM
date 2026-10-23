#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_jobs/gpu_XX.sh) into results/bench_tinyv/runs; files already
# here are kept as they are. Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
before=$(find results/bench_tinyv/runs -name 'metasift.json' -o -name 'tabsift.json' | wc -l)
rsync -a --ignore-existing --include '*/' --include 'metasift.*' --include 'tabsift.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(find results/bench_tinyv/runs -name 'metasift.json' -o -name 'tabsift.json' | wc -l) - before )) new result files"

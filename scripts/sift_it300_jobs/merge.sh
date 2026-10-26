#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_it300_jobs/gpu_XX.sh: tabsift_zero_lr0.01_it300, tabsift_zero_lr0.1_it300, tabsift_zero_lr5_it300, tabsift_adam_lr0.01_it300, tabsift_adam_lr0.1_it300) into results/bench_tinyv/runs;
# files already here are kept as they are. The locks and temporary files of jobs still running there are left out (a
# copied lock would make every run here skip that job). Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_it300_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_tinyv/runs -name 'tabsift_zero_lr0.01_it300.json' -o -name 'tabsift_zero_lr0.1_it300.json' -o -name 'tabsift_zero_lr5_it300.json' -o -name 'tabsift_adam_lr0.01_it300.json' -o -name 'tabsift_adam_lr0.1_it300.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' --include '*/' --include 'tabsift_zero_lr0.01_it300.*' --include 'tabsift_zero_lr0.1_it300.*' --include 'tabsift_zero_lr5_it300.*' --include 'tabsift_adam_lr0.01_it300.*' --include 'tabsift_adam_lr0.1_it300.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(count) - before )) new result files"

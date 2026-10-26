#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_lr_jobs/gpu_XX.sh: tabsift_adam_lr0.01, tabsift_adam_lr0.1, tabsift_adam_lr0.5, tabsift_adam_lr1, tabsift_adam_lr5, tabsift_sgd_lr1e3, tabsift_sgd_lr1e4, tabsift_sgd_lr5e4, tabsift_sgd_lr1e5, tabsift_sgd_lr5e5) into results/bench_tinyv/runs;
# files already here are kept as they are. The locks and temporary files of jobs still running there are left out (a
# copied lock would make every run here skip that job). Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_lr_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_tinyv/runs -name 'tabsift_adam_lr0.01.json' -o -name 'tabsift_adam_lr0.1.json' -o -name 'tabsift_adam_lr0.5.json' -o -name 'tabsift_adam_lr1.json' -o -name 'tabsift_adam_lr5.json' -o -name 'tabsift_sgd_lr1e3.json' -o -name 'tabsift_sgd_lr1e4.json' -o -name 'tabsift_sgd_lr5e4.json' -o -name 'tabsift_sgd_lr1e5.json' -o -name 'tabsift_sgd_lr5e5.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' --include '*/' --include 'tabsift_adam_lr0.01.*' --include 'tabsift_adam_lr0.1.*' --include 'tabsift_adam_lr0.5.*' --include 'tabsift_adam_lr1.*' --include 'tabsift_adam_lr5.*' --include 'tabsift_sgd_lr1e3.*' --include 'tabsift_sgd_lr1e4.*' --include 'tabsift_sgd_lr5e4.*' --include 'tabsift_sgd_lr1e5.*' --include 'tabsift_sgd_lr5e5.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(count) - before )) new result files"

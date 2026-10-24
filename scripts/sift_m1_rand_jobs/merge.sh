#!/bin/bash
# Bring one machine's finished sifting runs (scripts/sift_m1_rand_jobs/gpu_XX.sh: tabsift_m1_rand_s0, tabsift_m1_rand_s1, tabsift_m1_rand_s2, tabsift_m1_rand_s3, tabsift_m1_rand_s4, tabsift_m1_rand_ens) into results/bench_tinyv/runs;
# files already here are kept as they are. Then: python scripts/plot_bench_tinyv.py
#   bash scripts/sift_m1_rand_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_tinyv/runs -name 'tabsift_m1_rand_s0.json' -o -name 'tabsift_m1_rand_s1.json' -o -name 'tabsift_m1_rand_s2.json' -o -name 'tabsift_m1_rand_s3.json' -o -name 'tabsift_m1_rand_s4.json' -o -name 'tabsift_m1_rand_ens.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --include '*/' --include 'tabsift_m1_rand_s0.*' --include 'tabsift_m1_rand_s1.*' --include 'tabsift_m1_rand_s2.*' --include 'tabsift_m1_rand_s3.*' --include 'tabsift_m1_rand_s4.*' --include 'tabsift_m1_rand_ens.*' --exclude '*' \
  "${1%/}/results/bench_tinyv/runs/" results/bench_tinyv/runs/
echo "merged $(( $(count) - before )) new result files"

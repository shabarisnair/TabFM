#!/bin/bash
# Bring one machine's finished V + P benchmark runs (scripts/vp_jobs/gpu_XX.sh) into results/bench_vp/runs; files already here are kept as they
# are, and the locks and temporary files of jobs still running there are left out.
#   bash scripts/vp_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/bench_vp/runs -name '*.json' | wc -l; }
mkdir -p results/bench_vp/runs
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/results/bench_vp/runs/" results/bench_vp/runs/
echo "merged $(( $(count) - before )) new result files"

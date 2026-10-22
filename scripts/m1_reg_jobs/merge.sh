#!/bin/bash
# Bring one machine's finished M1 regularisation runs (scripts/m1_reg_jobs/gpu_XX.sh) into results/m1_reg/runs; files
# already here are kept as they are. Then: python scripts/report_m1_reg.py
#   bash scripts/m1_reg_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
before=$(find results/m1_reg/runs -name '*.json' 2>/dev/null | wc -l)
mkdir -p results/m1_reg/runs
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/results/m1_reg/runs/" results/m1_reg/runs/
echo "merged $(( $(find results/m1_reg/runs -name '*.json' | wc -l) - before )) new result files"

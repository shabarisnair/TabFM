#!/bin/bash
# Bring one machine's finished soft-form regularisation runs (scripts/soft_reg_jobs/gpu_XX.sh) into results/soft_reg/runs;
# files already here are kept as they are. Then: python scripts/report_soft_reg.py
#   bash scripts/soft_reg_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
before=$(find results/soft_reg/runs -name '*.json' 2>/dev/null | wc -l)
mkdir -p results/soft_reg/runs
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/results/soft_reg/runs/" results/soft_reg/runs/
echo "merged $(( $(find results/soft_reg/runs -name '*.json' | wc -l) - before )) new result files"

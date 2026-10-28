#!/bin/bash
# Bring one machine's finished units of the repair development study (scripts/vp_tune_repair_jobs/gpu_XX.sh) into results/vp_tune/repair; files already here
# are kept as they are, and the locks and temporary files of units still running there are left out.
#   bash scripts/vp_tune_repair_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/vp_tune/repair -name '*.json' | wc -l; }
mkdir -p results/vp_tune/repair
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/results/vp_tune/repair/" results/vp_tune/repair/
echo "merged $(( $(count) - before )) new unit results ($(count) of 72 here)"

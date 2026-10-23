#!/bin/bash
# Bring one machine's finished sigmoid-study runs (scripts/sigmoid_jobs/gpu_XX.sh) into results/sigmoid_study/runs; files
# already here are kept as they are. Then: python scripts/sigmoid_study.py report
#   bash scripts/sigmoid_jobs/merge.sh <host>:<its repo path>       (or a local path to a copy of that repository)
set -euo pipefail
cd "$(dirname "$0")/../.."
before=$(find results/sigmoid_study/runs -name '*.json' 2>/dev/null | wc -l)
mkdir -p results/sigmoid_study/runs
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/results/sigmoid_study/runs/" results/sigmoid_study/runs/
echo "merged $(( $(find results/sigmoid_study/runs -name '*.json' | wc -l) - before )) new result files"

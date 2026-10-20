#!/bin/bash
# Bring the second machine's finished runs into results/attn_form/runs (runs already here are kept as they are).
#   bash scripts/attn_form_jobs/merge_remote.sh <host>:<its repo>/results/attn_form/runs     (or a local copy of that folder)
set -euo pipefail
cd "$(dirname "$0")/../.."
count() { find results/attn_form/runs -name '*.json' | wc -l; }
before=$(count)
rsync -a --ignore-existing --exclude '*.lock' --exclude '*.tmp' "${1%/}/" results/attn_form/runs/
echo "merged $(( $(count) - before )) new runs"
PYTHONWARNINGS=ignore conda run -n tabfm --no-capture-output python scripts/attn_form_study.py status

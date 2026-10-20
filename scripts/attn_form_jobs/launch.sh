#!/bin/bash
# The soft attention-weight form study (scripts/attn_form_study.py): 3 workers per GPU sharing one queue (lock files);
# at most 1 gas-drift / coil2000 run per GPU at a time. Logs: logs/attn_form/gpu<g>_w<w>.log.
#   [ONLY='w/*/*,sigmoid/*/adam,sigmoid/*/pgd'] bash scripts/attn_form_jobs/launch.sh [GPU ...]   (default: 2 3)
# ONLY: this machine's share of the configurations (see README.md; empty: all of them).
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/attn_form
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
for g in "${@:-2 3}"; do
  for w in 0 1 2; do
    nohup conda run -n tabfm --no-capture-output python -u scripts/attn_form_study.py work --device "cuda:$g" \
      --worker "$((g * 10 + w))" --only "${ONLY:-}" >> "logs/attn_form/gpu${g}_w${w}.log" 2>&1 &
  done
done
echo "started: $(jobs -p | wc -l) workers (configurations: ${ONLY:-all})"

#!/bin/bash
# Soft-form / M1 runs at R = 30% (scripts/plan_soft_m1.py), script 04 of 5: 13 jobs, about 0.7 GPU-hours
# (H100). Run from a clone of this repository with results/ copied over, on any free GPU:
#   bash scripts/soft_m1_jobs/gpu_04.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/soft_m1_jobs/.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/soft_m1_jobs
echo "script 04 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/soft_m1_jobs/inputs.sha256 \
  > logs/soft_m1_jobs/gpu_04_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/soft_m1_jobs/gpu_04_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  [ -s scripts/soft_m1_jobs/largev_04.txt ] && $PY -u scripts/bench_largev.py run --device "cuda:$GPU" \
    --job-list scripts/soft_m1_jobs/largev_04.txt >> logs/soft_m1_jobs/gpu_04.log 2>&1
  [ -s scripts/soft_m1_jobs/tinyv_04.txt ] && $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" \
    --job-list scripts/soft_m1_jobs/tinyv_04.txt >> logs/soft_m1_jobs/gpu_04.log 2>&1
done
[ -s scripts/soft_m1_jobs/largev_04.txt ] && $PY scripts/bench_largev.py status --job-list scripts/soft_m1_jobs/largev_04.txt
grep -h "FAILED\|SKIPPING" logs/soft_m1_jobs/gpu_04.log || echo "script 04: no failures"

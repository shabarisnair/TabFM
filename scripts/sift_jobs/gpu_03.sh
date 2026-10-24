#!/bin/bash
# Sifting methods on the tiny-V benchmark at R = 30% (scripts/plan_sift.py), script 03 of 11: 3 jobs, about
# 0.4 GPU-hours (H100). Run from a clone of this repository with results/, datasets/ and Meta-Sift/ copied over:
#   bash scripts/sift_jobs/gpu_03.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/sift_jobs/.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/sift_jobs
echo "script 03 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/sift_jobs/inputs.sha256 \
  > logs/sift_jobs/gpu_03_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/sift_jobs/gpu_03_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list scripts/sift_jobs/tinyv_03.txt >> logs/sift_jobs/gpu_03.log 2>&1
done
$PY scripts/bench_tinyv.py status --job-list scripts/sift_jobs/tinyv_03.txt
grep -h "FAILED\|SKIPPING" logs/sift_jobs/gpu_03.log || echo "script 03: no failures"

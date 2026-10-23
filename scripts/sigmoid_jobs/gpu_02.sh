#!/bin/bash
# Sigmoid soft attention weights (scripts/sigmoid_study.py), script 02 of 9: 41 jobs, about 6.3 GPU-hours
# (H100). Run from a clone of this repository with results/ and datasets/ copied over:
#   bash scripts/sigmoid_jobs/gpu_02.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/sigmoid_jobs/.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/sigmoid_jobs
echo "script 02 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/sigmoid_jobs/inputs.sha256 \
  > logs/sigmoid_jobs/gpu_02_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/sigmoid_jobs/gpu_02_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/sigmoid_study.py work --device "cuda:$GPU" --job-list scripts/sigmoid_jobs/gpu_02.txt >> logs/sigmoid_jobs/gpu_02.log 2>&1
done
$PY scripts/sigmoid_study.py status --job-list scripts/sigmoid_jobs/gpu_02.txt
grep -h "FAILED" logs/sigmoid_jobs/gpu_02.log || echo "script 02: no failures"

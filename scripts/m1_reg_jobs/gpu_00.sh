#!/bin/bash
# M1 regularisation study (scripts/m1_reg_study.py), script 00 of 9: 12 jobs, about 2.2 GPU-hours
# (H100), in priority order (the core arms first). Run from a clone of this repository with results/ copied over:
#   bash scripts/m1_reg_jobs/gpu_00.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/m1_reg/.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/m1_reg
echo "script 00 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/m1_reg_jobs/inputs.sha256 \
  > logs/m1_reg/gpu_00_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/m1_reg/gpu_00_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/m1_reg_study.py work --device "cuda:$GPU" --job-list scripts/m1_reg_jobs/gpu_00.txt >> logs/m1_reg/gpu_00.log 2>&1
done
$PY scripts/m1_reg_study.py status --job-list scripts/m1_reg_jobs/gpu_00.txt
grep -h "FAILED" logs/m1_reg/gpu_00.log || echo "script 00: no failures"

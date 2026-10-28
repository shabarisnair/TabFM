#!/bin/bash
# The repair methods' development study (scripts/vp_tune_repair.py; planned by its plan stage), script 03 of 7:
# 11 units, about 11 min on a free H100. Run from a clone of this repository with results/vp_tune/draws,
# datasets/ and the preflight inputs (results/attn_form/preflight_reference, results/bench_largev/draws) copied over:
#   bash scripts/vp_tune_repair_jobs/gpu_03.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Results: results/vp_tune/repair/ (bring them back with
# scripts/vp_tune_repair_jobs/merge.sh). No W&B. Rerunning resumes; logs in logs/vp_tune_repair_jobs/. TabPFN inference only (no gradients): far less GPU
# memory than the benchmark's attention jobs.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN (as the benchmark runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/vp_tune_repair_jobs
echo "script 03 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/vp_tune_repair_jobs/inputs.sha256 \
  > logs/vp_tune_repair_jobs/gpu_03_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/vp_tune_repair_jobs/gpu_03_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/vp_tune_repair.py run --device "cuda:$GPU" --units scripts/vp_tune_repair_jobs/units_03.txt >> logs/vp_tune_repair_jobs/gpu_03.log 2>&1
done
$PY scripts/vp_tune_repair.py status --units scripts/vp_tune_repair_jobs/units_03.txt
grep -h "FAILED" logs/vp_tune_repair_jobs/gpu_03.log || echo "script 03: no failures"

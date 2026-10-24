#!/bin/bash
# Sifting methods on the tiny-V benchmark at R = 30% (scripts/plan_sift.py), script 01 of 4: 30 jobs, about
# 8.5 GPU-hours (H100). Run from a clone of this repository with results/, datasets/ and Meta-Sift/ copied over:
#   bash scripts/sift_m1_rand_jobs/gpu_01.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/sift_m1_rand_jobs/.
# W&B: it also stops unless W&B accepts the API key in .env. Each job is one W&B run, named <method>_<instance>,
# in project TFM_experiments, group tinyv_tabsift_m1_rand_v2, with its result files as an artifact.
# One script per GPU: two gas-drift or coil2000 jobs on one GPU run out of memory.
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/sift_m1_rand_jobs
echo "script 01 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/sift_m1_rand_jobs/inputs.sha256 \
  > logs/sift_m1_rand_jobs/gpu_01_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/sift_m1_rand_jobs/gpu_01_check.log
[ $status -eq 0 ] || exit 1
. scripts/load_env.sh "$PWD/.env"  # WANDB_API_KEY from this clone's .env (a variable already set wins)
$PY -c "import wandb; api = wandb.Api(timeout=60); api.viewer; print('  wandb: logged in, entity', api.default_entity)" \
  || { echo "  wandb: NOT logged in. Put WANDB_API_KEY in .env here (or export it, or run wandb login)"; exit 1; }
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_tinyv.py run --device "cuda:$GPU" --job-list scripts/sift_m1_rand_jobs/tinyv_01.txt --wandb-mode online --wandb-group tinyv_tabsift_m1_rand_v2 >> logs/sift_m1_rand_jobs/gpu_01.log 2>&1
done
$PY scripts/bench_tinyv.py status --job-list scripts/sift_m1_rand_jobs/tinyv_01.txt
grep -h "FAILED\|SKIPPING" logs/sift_m1_rand_jobs/gpu_01.log || echo "script 01: no failures"

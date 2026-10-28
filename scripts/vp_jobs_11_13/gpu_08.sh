#!/bin/bash
# The V + P benchmark (scripts/bench_vp.py; planned by scripts/plan_vp.py), script 08 of 12: 43 jobs, about
# 0.3 GPU-hours (H100). Run from a clone of this repository with results/bench_vp/draws, datasets/ and the
# preflight inputs (results/attn_form/preflight_reference, results/bench_largev/draws) copied over:
#   bash scripts/vp_jobs_11_13/gpu_08.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py), and unless W&B accepts the API key in .env. Each job is
# one W&B run, named <job>_<instance>, in project TFM_experiments, group vp_bench, with its result files as an artifact.
# Rerunning resumes; logs in logs/vp_jobs_11_13/. One script per GPU, with about 4 GiB free: these jobs are TabPFN inference only (gas-drift peaks at 2.6 GiB).
GPU=${1:-0}
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/vp_jobs_11_13
echo "script 08 on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest scripts/vp_jobs_11_13/inputs.sha256 \
  > logs/vp_jobs_11_13/gpu_08_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/vp_jobs_11_13/gpu_08_check.log
[ $status -eq 0 ] || exit 1
. scripts/load_env.sh "$PWD/.env"  # WANDB_API_KEY from this clone's .env (a variable already set wins)
$PY -c "import wandb; api = wandb.Api(timeout=60); api.viewer; print('  wandb: logged in, entity', api.default_entity)" \
  || { echo "  wandb: NOT logged in. Put WANDB_API_KEY in .env here (or export it, or run wandb login)"; exit 1; }
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/bench_vp.py run --device "cuda:$GPU" --job-list scripts/vp_jobs_11_13/jobs_08.txt --wandb-mode online \
    --wandb-group vp_bench >> logs/vp_jobs_11_13/gpu_08.log 2>&1
done
$PY scripts/bench_vp.py status --job-list scripts/vp_jobs_11_13/jobs_08.txt
grep -h "FAILED" logs/vp_jobs_11_13/gpu_08.log || echo "script 08: no failures"

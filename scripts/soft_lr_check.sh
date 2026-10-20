#!/bin/bash
# Does the best soft form reach lr 0.8's level at lr 0.2 with a 4x larger budget? (scripts/check_soft_lr.py)
# V = 500, R = 30%, heloc / students / ecommerce / good_customer, both rates in strict deterministic mode with the test
# set tracked: 8 runs, about 40-60 min on a free H100 (strict attention needs up to ~40 GB on heloc).
#   bash scripts/soft_lr_check.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# Needs results/bench_largev/draws/*_R30_V500_s0.npz and results/attn_form/preflight_reference (in results/).
# When it ends, copy results/soft_lr_check/ to the main machine's results/soft_lr_check/.
GPU=${1:-0}
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/soft_lr_check
echo "GPU $GPU: checking the machine (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" > logs/soft_lr_check/check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/soft_lr_check/check.log
[ $status -eq 0 ] || exit 1
$PY -u scripts/check_soft_lr.py run --device "cuda:$GPU" >> logs/soft_lr_check/run.log 2>&1
grep -E "_R30_V500_s0: lr .*chosen|FAILED|Error" logs/soft_lr_check/run.log
echo "done: copy results/soft_lr_check/ to the main machine, e.g. rsync -a results/soft_lr_check/ <main>:<repo>/results/soft_lr_check/"

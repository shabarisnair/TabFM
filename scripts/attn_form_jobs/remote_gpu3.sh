#!/bin/bash
# Attention-form study (scripts/attn_form_study.py) on the second machine, GPU 3. This machine's share is the
# configurations log*/*/* and sigmoid/*/sign; the main machine runs w/*/* and sigmoid/*/adam|pgd, so no run is done on
# both. The workers of remote_gpu0-3.sh share one queue (lock files under results/attn_form): they choose the learning
# rates, run gas-drift / coil2000 at the chosen ones, and exit when the share is done. Rerunning resumes.
#   bash scripts/attn_form_jobs/remote_gpu3.sh        (PY=/path/to/python to use another python than conda env tabfm)
# First, check_remote.py makes sure this machine computes exactly what the main one does (inputs, model checkpoint,
# code, numbers); nothing starts if it fails.
GPU=3
ONLY='log*/*/*,sigmoid/*/sign'
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/attn_form
echo "GPU $GPU: checking (about a minute; log: logs/attn_form/remote_gpu${GPU}_check.log)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" > "logs/attn_form/remote_gpu${GPU}_check.log" 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" "logs/attn_form/remote_gpu${GPU}_check.log"
[ $status -eq 0 ] || exit 1
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU")
n=$(( mem >= 80000 ? 3 : (mem >= 48000 ? 2 : 1) ))  # a gas-drift run takes ~38 GiB; at most one per GPU at a time
for w in $(seq 0 $((n - 1))); do
  $PY -u scripts/attn_form_study.py work --device "cuda:$GPU" --worker "$((GPU * 10 + w))" --only "$ONLY" \
    >> "logs/attn_form/remote_gpu${GPU}_w${w}.log" 2>&1 &
done
echo "GPU $GPU: $n workers (logs/attn_form/remote_gpu${GPU}_w*.log)"
wait
$PY scripts/attn_form_study.py status --only "$ONLY"

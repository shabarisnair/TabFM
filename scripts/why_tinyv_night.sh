#!/bin/bash
# The tiny-V "why" study (scripts/why_tinyv.py) end to end on two GPUs of this machine; run it detached:
#   setsid nohup bash scripts/why_tinyv_night.sh <GPU A> <GPU B> > logs/why_tinyv/night.out 2>&1 < /dev/null &
# GPU A: the diagnostics on the benchmark's hub-flip-local draws (trace, judge, landscape, grad), then the learned methods
# on the control draws; GPU B: the cheap methods on the control draws (rand-flip, hub-flip-random). Then the diagnostics
# on the control draws and the report. Every step skips what is already done, so rerunning resumes. Logs: logs/why_tinyv/.
cd "$(dirname "$0")/.." || exit 1
A=${1:-0}
B=${2:-$A}
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
L=logs/why_tinyv
mkdir -p $L
W="$PY -u scripts/why_tinyv.py"
stamp () { echo "$(date '+%F %T') $*" >> $L/night.log; }

run_bench () {  # GPU, attack, job list; twice (the second pass retries failures)
  for pass in 1 2; do
    BENCH_TINYV_ROOT=results/why_tinyv/roots/$2 $PY -u scripts/bench_tinyv.py run --device "cuda:$1" --job-list "$3" \
      >> $L/bench_$2.log 2>&1
  done
}

stamp "start: GPU A $A, GPU B $B"
$W draws --device "cuda:$A" > $L/draws.log 2>&1 || { stamp "draws FAILED"; exit 1; }
stamp "control draws built"

( for att in rand-flip hub-flip-random; do run_bench "$B" $att results/why_tinyv/jobs/cheap.txt; stamp "cheap runs done: $att"; done ) &
PB=$!

$W trace >> $L/diag.log 2>&1
for st in judge landscape grad; do
  $W $st --device "cuda:$A" >> $L/diag.log 2>&1
  stamp "hub-flip-local $st done"
done
for att in rand-flip hub-flip-random; do
  run_bench "$A" $att results/why_tinyv/jobs/learned.txt
  stamp "learned runs done: $att"
done

wait $PB
for st in trace judge landscape grad; do
  $W $st --device "cuda:$A" --attacks rand-flip hub-flip-random >> $L/diag_controls.log 2>&1
  stamp "controls $st done"
done
$W report > $L/report.log 2>&1
stamp "all done"

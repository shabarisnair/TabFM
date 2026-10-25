#!/bin/bash
# After the control runs (scripts/why_tinyv_controls.sh) finish: the diagnostics of scripts/why_tinyv.py on the control
# draws (rand-flip: all 18 instances; hub-flip-random: gas-drift and students). GPU for the forward-only stages, CPU for
# the rest. Logs: logs/why_tinyv/. Rerunning resumes (the judge is redone once, to add greedy's and the methods' entries).
#   setsid nohup bash scripts/why_tinyv_controls_diag.sh <GPU> > logs/why_tinyv/controls_diag.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
G=${1:-1}
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
L=logs/why_tinyv
J=results/why_tinyv/jobs
W="$PY -u scripts/why_tinyv.py"
left () {  # unfinished jobs of a list in a root
  BENCH_TINYV_ROOT=results/why_tinyv/roots/$1 $PY scripts/bench_tinyv.py status --job-list "$2" 2>/dev/null | head -1 \
    | sed -E 's/.*: ([0-9]+) of ([0-9]+) jobs done/\2 \1/' | awk '{print $1 - $2}'
}
until [ "$(left rand-flip $J/cpu_rand-flip.txt)" = 0 ] && [ "$(left hub-flip-random $J/cpu_hub-flip-random.txt)" = 0 ] \
   && [ "$(left rand-flip $J/gpu_metasift.txt)" = 0 ] && [ "$(left hub-flip-random $J/gpu_metasift_hfr.txt)" = 0 ]; do
  sleep 120
done
echo "$(date '+%F %T') control runs complete; diagnostics start" >> $L/night.log
$W trace --attacks rand-flip >> $L/diag_controls.log 2>&1
$W trace --attacks hub-flip-random --datasets gas-drift students_dropout >> $L/diag_controls.log 2>&1
( $W removed --device cpu --threads 16 --attacks rand-flip >> $L/diag_controls.log 2>&1
  $W removed --device cpu --threads 16 --attacks hub-flip-random --datasets gas-drift students_dropout >> $L/diag_controls.log 2>&1
  $W grad --device cpu --threads 24 --attacks rand-flip --datasets is_this_a_good_customer heloc students_dropout ecommerce_shipping --cs 5 \
    >> $L/grad_randflip_cpu.log 2>&1 ) &
$W judge --device "cuda:$G" --chunk 512 --attacks rand-flip --force >> $L/diag_controls.log 2>&1
$W judge --device "cuda:$G" --chunk 512 --attacks hub-flip-random --datasets gas-drift students_dropout --force >> $L/diag_controls.log 2>&1
$W landscape --device "cuda:$G" --attacks rand-flip >> $L/diag_controls.log 2>&1
$W landscape --device "cuda:$G" --attacks hub-flip-random --datasets gas-drift students_dropout >> $L/diag_controls.log 2>&1
wait
echo "$(date '+%F %T') control diagnostics done" >> $L/night.log

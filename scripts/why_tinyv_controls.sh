#!/bin/bash
# Control runs of the tiny-V "why" study (scripts/why_tinyv.py) on this machine while its GPUs are busy: the cheap
# methods on CPU workers (rand-flip: all 18 instances; hub-flip-random: the multiclass gas-drift and students, since on
# binary data it equals hub-flip-local; a CPU rerun of the V-filter and greedy pct10 on the benchmark's own draws to check
# CPU against GPU), and Meta-Sift (their code needs CUDA) on one GPU. Rerunning resumes. Logs: logs/why_tinyv/.
#   setsid nohup bash scripts/why_tinyv_controls.sh <GPU for Meta-Sift> > logs/why_tinyv/controls.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
G=${1:-0}
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
L=logs/why_tinyv
J=results/why_tinyv/jobs
R=results/why_tinyv/roots
mkdir -p $L

pool () {  # root, job list, workers, device, threads, log name
  for w in $(seq 1 "$3"); do
    ( for pass in 1 2; do
        BENCH_TINYV_ROOT=$R/$1 $PY -u scripts/bench_tinyv.py run --device "$4" --threads "$5" --job-list "$2" >> $L/ctl_$6_w$w.log 2>&1
      done ) &
  done
}

pool rand-flip $J/cpu_rand-flip.txt 4 cpu 10 randflip
pool hub-flip-random $J/cpu_hub-flip-random.txt 2 cpu 10 hfr
pool hub-flip-local-cpu $J/cpu_devcheck.txt 2 cpu 10 devcheck
( for pass in 1 2; do
    BENCH_TINYV_ROOT=$R/rand-flip $PY -u scripts/bench_tinyv.py run --device "cuda:$G" --job-list $J/gpu_metasift.txt >> $L/ctl_metasift.log 2>&1
    BENCH_TINYV_ROOT=$R/hub-flip-random $PY -u scripts/bench_tinyv.py run --device "cuda:$G" --job-list $J/gpu_metasift_hfr.txt >> $L/ctl_metasift.log 2>&1
  done ) &
wait
echo "$(date '+%F %T') controls done" >> $L/night.log

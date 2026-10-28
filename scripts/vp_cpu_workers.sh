#!/bin/bash
# Run V + P benchmark jobs (scripts/bench_vp.py) or the repair methods' development study (scripts/vp_tune_repair.py)
# on this machine's CPU: N workers of T torch threads each share one list (locks), and every worker passes over it 3
# times (a later pass picks up a job skipped earlier, e.g. a repair job whose other-poisoning twin was running).
# Returns when all workers are done. Logs: logs/vp_cpu/<name>_wNN.log.
#   bash scripts/vp_cpu_workers.sh bench <name> <job-list> <N> <T>   # W&B online (project TFM_experiments, group vp_bench)
#   bash scripts/vp_cpu_workers.sh tune <name> <N> <T>
mode=$1 name=$2
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-"conda run -n tabfm --no-capture-output python"}
export PYTHONPATH="$PWD/TabPFN/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore
mkdir -p logs/vp_cpu
if [ "$mode" = bench ]; then
  list=$3 N=$4 T=$5
  . scripts/load_env.sh "$PWD/.env"  # WANDB_API_KEY
  cmd="scripts/bench_vp.py run --device cpu --threads $T --job-list $list --wandb-mode online --wandb-group vp_bench"
else
  N=$3 T=$4
  cmd="scripts/vp_tune_repair.py run --threads $T"
fi
for i in $(seq -w 0 $((N - 1))); do
  (for pass in 1 2 3; do OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T $PY -u $cmd; done) >> "logs/vp_cpu/${name}_w$i.log" 2>&1 &
  sleep 2
done
wait
echo "$name: all $N workers done"

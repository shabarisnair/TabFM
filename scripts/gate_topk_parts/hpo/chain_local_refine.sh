#!/usr/bin/env bash
# Start a refine_local.txt worker on each GPU as soon as that GPU's stage-1 worker has
# finished its list (its log ends with ALL REQUESTED RUNS FINISHED).
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
for w in 0a 0b 1a 2a; do
  (
    until grep -q "ALL REQUESTED RUNS FINISHED" "logs/gate_topk_hpo_stage1_gpu$w.log" 2>/dev/null; do sleep 30; done
    GPUS=${w:0:1} JOBS_FILE=scripts/gate_topk_parts/hpo/refine_local.txt \
      bash scripts/run_gate_topk_sweep.sh > "logs/gate_topk_hpo_refine_gpu$w.log" 2>&1
  ) &
done
wait

#!/usr/bin/env bash
# =============================================================================
#  Our sift method across the full grid, at the tuned learning rate
#
#  5 datasets x {random label-flip, hub-flip} at R=30% = 10 conditions.
#
#  lr=5 comes from the 9-point sweep on url_unique/hub-flip (0.1 ... 50): precision
#  rises to lr=5 and falls monotonically above it, so it is an interior optimum, not the
#  edge of a grid. 20 iterations, matching that sweep -- every run there had plateaued
#  by iteration ~8, and the whole X trajectory is saved either way.
#
#  Extraction at m=250 and at the (1-R) full budget, which is n-k: 3500 for the
#  5000-row datasets, 320 for credit-approval (457 rows). Each subset is then scored
#  with TabPFNv2 on the held-out test set, against clean / poisoned / oracle / pure-m.
#
#  Cost: ~1 min/iteration on a 5000-row context, so ~20 min per 5000-row condition and
#  seconds for credit-approval. Split across two GPUs below, ~80 min wall clock.
#
#    GPU=0 DATASETS="url_unique heloc credit-approval" bash scripts/run_sift_all.sh &
#    GPU=3 DATASETS="coil2000_insurance_policies lcld_v2" bash scripts/run_sift_all.sh &
# =============================================================================
set -u
REPO=/home/ssn899/Desktop/TabFM
GPU=${GPU:-0}
LR=${LR:-5}
N_ITER=${N_ITER:-20}
cd "$REPO" || exit 1
PY=(conda run --no-capture-output -n tabfm python)

DATASETS=${DATASETS:-"credit-approval url_unique coil2000_insurance_policies heloc lcld_v2"}
ATTACKS=${ATTACKS:-"label-flip-random hub-flip"}

for ds in $DATASETS; do
  for atk in $ATTACKS; do
    tag=$(printf "%s_r030_lr%s_it%s" "$atk" "$LR" "$N_ITER")
    if [[ -f "$REPO/results/sift/$ds/$tag/summary.json" ]]; then
      echo "[skip] $ds/$atk"; continue
    fi
    echo "[$(date +'%F %T')] START $ds / $atk"
    "${PY[@]}" "$REPO/scripts/sift_context.py" --gpu "$GPU" \
      --dataset "$ds" --attack "$atk" --row-percent 30 \
      --lr "$LR" --n-iter "$N_ITER" --sizes 250 --out "$REPO/results/sift" \
      > "$REPO/logs/sift_all_${ds}_${atk}.log" 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && echo "[$(date +'%F %T')] DONE  $ds / $atk" \
                    || echo "[$(date +'%F %T')] FAIL  $ds / $atk (rc=$rc)"
  done
done
echo "ALL SIFT CONDITIONS FINISHED"

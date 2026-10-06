#!/usr/bin/env bash
# =============================================================================
#  Meta-Sift baseline across the full grid, at the tuned hyperparameters
#
#  5 datasets x {random label-flip, hub-flip} at R=30% = 10 conditions.
#
#  HYPERPARAMETERS: their published defaults EXCEPT the four established by the
#  tuning sweep on url_unique/hub-flip (see logs/metasift_sweep.log):
#     res_epochs  1 -> 5      decisive: at 1 the weight-net is INVERTED (AUC 0.401)
#                             and selects worse than random; at 20 it collapses again
#     v_lr    5e-4 -> 0.01    purity@100 0.90 -> 0.98
#     model_lr 1e-4 -> 1e-3   purity@100 0.98 -> 0.99
#     top_k     15 -> 6       FORCED, not tuned: the MLP backbone has 8 parameter
#                             blocks, so [-15:-2] would run past the start
#
#  Sizes: 250, plus the (1-R) full budget, which is n-k -- 3500 for the 5000-row
#  datasets and 320 for credit-approval (457 rows, where 3500 is impossible).
#
#  Each run also scores every selected subset with TabPFNv2 on the held-out test set,
#  against clean / poisoned / oracle / pure-m controls.
# =============================================================================
set -u
REPO=/home/ssn899/Desktop/TabFM
GPU=${GPU:-0}
cd "$REPO" || exit 1
PY=(conda run --no-capture-output -n tabfm python)

DATASETS=${DATASETS:-"credit-approval url_unique coil2000_insurance_policies heloc lcld_v2"}
ATTACKS=${ATTACKS:-"label-flip-random hub-flip"}

for ds in $DATASETS; do
  for atk in $ATTACKS; do
    out="$REPO/results/meta_sift/$ds/${atk}_r030_tuned/summary.json"
    if [[ -f "$out" ]]; then echo "[skip] $ds/$atk"; continue; fi
    echo "[$(date +'%F %T')] START $ds / $atk"
    "${PY[@]}" "$REPO/scripts/run_meta_sift_baseline.py" --gpu "$GPU" \
      --dataset "$ds" --attack "$atk" --row-percent 30 --tag tuned \
      --warmup-epochs 1 --res-epochs 5 --v-lr 0.01 --model-lr 0.001 --sizes 250 \
      > "$REPO/logs/metasift_${ds}_${atk}.log" 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && echo "[$(date +'%F %T')] DONE  $ds / $atk" \
                    || echo "[$(date +'%F %T')] FAIL  $ds / $atk (rc=$rc)"
  done
done
echo "ALL META-SIFT CONDITIONS FINISHED"

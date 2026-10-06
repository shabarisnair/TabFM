#!/usr/bin/env bash
# =============================================================================
#  ICLAD unsupervised anomaly-detection baseline, full grid
#
#  5 datasets x {random label-flip, hub-flip} at R=30% = 10 conditions.
#
#  Pretrained iclad_mixedprior_unsup checkpoint, library defaults throughout -- no
#  tuning. At our sizes nothing is subsampled: the model takes 512 features (we have at
#  most 86 including the appended label) and max_train_len is 10000 (contexts <= 5000).
#
#  The contaminated set is both the context and the query set, which is ICLAD's
#  unsupervised regime. The poisoned label is appended as an extra FEATURE, since ICLAD
#  has no separate target. Each run also scores a features-only control: the attacks
#  never touch features, so that variant should have no signal, and it isolates what the
#  label column contributes.
#
#  Selection keeps the LOWEST anomaly scores (= most likely clean), at m=250 and at the
#  (1-R) full budget -- 3500 for the 5000-row datasets, 320 for credit-approval.
#
#  Cost: ~20 s per condition including the TabPFNv2 downstream scoring.
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
    out="$REPO/results/iclad/$ds/${atk}_r030_default/summary.json"
    if [[ -f "$out" ]]; then echo "[skip] $ds/$atk"; continue; fi
    echo "[$(date +'%F %T')] START $ds / $atk"
    "${PY[@]}" "$REPO/scripts/run_iclad_baseline.py" --gpu "$GPU" \
      --dataset "$ds" --attack "$atk" --row-percent 30 --sizes 250 \
      > "$REPO/logs/iclad_${ds}_${atk}.log" 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && echo "[$(date +'%F %T')] DONE  $ds / $atk" \
                    || echo "[$(date +'%F %T')] FAIL  $ds / $atk (rc=$rc)"
  done
done
echo "ALL ICLAD CONDITIONS FINISHED"

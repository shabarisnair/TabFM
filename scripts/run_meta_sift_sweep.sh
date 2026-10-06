#!/usr/bin/env bash
# Fair-shot tuning for the Meta-Sift baseline on url_unique / hub-flip / R=30%.
#
# Their published defaults (warmup 1, res_epochs 1, v_lr 5e-4, backbone lr 2.5e-5*bs/32)
# are set for CIFAR-10/GTSRB: 50k images through a ResNet18. Our context is 5000 tabular
# rows, so one epoch is 39 batches -- far too few for the weight-assigning network to
# learn the right monotonicity, which is why the default run selected WORSE than random.
# This sweep gives the method enough steps before its numbers are reported.
set -u
REPO=/home/ssn899/Desktop/TabFM
GPU=${GPU:-3}
cd "$REPO" || exit 1
PY=(conda run --no-capture-output -n tabfm python)

run () {  # run <tag> <warmup> <res_epochs> <v_lr> <model_lr>
  local tag=$1 wu=$2 re=$3 vlr=$4 mlr=$5
  if [[ -f "$REPO/results/meta_sift/url_unique/hub-flip_r030_${tag}/summary.json" ]]; then
    echo "[skip] $tag"; return 0
  fi
  echo "[$(date +'%F %T')] START $tag  (warmup $wu, res $re, v_lr $vlr, model_lr $mlr)"
  "${PY[@]}" "$REPO/scripts/run_meta_sift_baseline.py" --gpu "$GPU" \
    --dataset url_unique --attack hub-flip --row-percent 30 --tag "$tag" \
    --warmup-epochs "$wu" --res-epochs "$re" --v-lr "$vlr" --model-lr "$mlr" \
    > "$REPO/logs/metasift_${tag}.log" 2>&1
  echo "[$(date +'%F %T')] DONE  $tag"
}

#    tag              warmup  res   v_lr    model_lr
run default              1     1    0.0005  0.0001
run more_epochs          1     5    0.0005  0.0001
run vlr_hi               1     5    0.01    0.0001
run vlr_hi_long          1    20    0.01    0.0001
run warm5_vlr_hi         5    20    0.01    0.0001
run fastmodel            1     5    0.01    0.001
run fastmodel_long       1    20    0.01    0.001
echo "ALL META-SIFT CONFIGS FINISHED"

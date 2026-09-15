#!/usr/bin/env bash
# usage: run_grid.sh <dataset> <gpu> <out_root> <context file> "<label>|<extra args>" ...
# x-capgd, selected/natural/<context>, test_attack_1000.csv, 30% rows, one trial per spec, constraints full.
set -u
DS=$1; GPU=$2; ROOT=$3; CTX=$4; shift 4
cd /home/ssn899/Desktop/TabFM
for spec in "$@"; do
  LABEL=${spec%%|*}; EXTRA=${spec#*|}
  OUT=$ROOT/${DS}_${LABEL}
  if [ ! -f $OUT/summary.json ]; then
    conda run -n tabfm python scripts/attack_context.py --gpu $GPU --attack x-capgd \
      --train datasets/$DS/splits/selected/natural/$CTX --test datasets/$DS/splits/test_attack_1000.csv \
      --out $OUT --row-percent 30 --n-runs 1 --n-row-subsamples 1 --constraints full $EXTRA > $ROOT/${DS}_${LABEL}.stdout 2>&1
  fi
  echo "$(date +%T) $DS $LABEL: $(python3 -c "
import json
t=json.load(open('$OUT/trials/run0_sub0.json'))
print('dce=%+.4f dacc=%+.4f dauc=%+.4f pre_repair_obj=%.4f within_eps=%s maxL2=%.3f viol=%s %.0fs'%(t['delta']['ce'],t['delta']['accuracy'],t['delta']['roc_auc'],t['restarts'][0]['objective_pre_repair'],t['within_eps_after_repair'],t['max_l2_delta_scaled'],round(t.get('relation_violation_final_mean') or 0,4),t['seconds']))" 2>/dev/null || (echo FAILED; grep -E 'Error|error' $ROOT/${DS}_${LABEL}.stdout | tail -1))"
done

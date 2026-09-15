#!/usr/bin/env bash
# usage: run_sweep.sh <dataset> <gpu>
# n_iter ablation: selected/natural/context_5000.csv vs test_attack_1000.csv, 30% rows, one trial
# (run 0 = clean start, subsample 0), all other args at CLI defaults.
set -u
DS=$1; GPU=$2
cd /home/ssn899/Desktop/TabFM
for C in full box; do
  for IT in 10 20 30 50 75 100; do
    OUT=results/ablation_niter/${DS}_nat5000_${C}_it${IT}
    [ -f $OUT/summary.json ] && continue
    conda run -n tabfm python scripts/attack_context.py --gpu $GPU --attack x-capgd \
      --train datasets/$DS/splits/selected/natural/context_5000.csv \
      --test datasets/$DS/splits/test_attack_1000.csv \
      --out $OUT --row-percent 30 --n-iter $IT --n-runs 1 --n-row-subsamples 1 --constraints $C > /dev/null 2>&1
    if [ ! -f $OUT/summary.json ]; then
      echo "$DS $C it$IT failed unchunked; retrying with --test-batch-size 250"
      conda run -n tabfm python scripts/attack_context.py --gpu $GPU --attack x-capgd \
        --train datasets/$DS/splits/selected/natural/context_5000.csv \
        --test datasets/$DS/splits/test_attack_1000.csv \
        --out $OUT --row-percent 30 --n-iter $IT --n-runs 1 --n-row-subsamples 1 --constraints $C --test-batch-size 250 > /dev/null 2>&1
    fi
    echo "$(date +%T) done $DS $C it$IT: $(python3 -c "import json;d=json.load(open('$OUT/summary.json'))['trials'][0];print('dce=%+.4f dacc=%+.4f dauc=%+.4f  %.0fs'%(d['delta']['ce'],d['delta']['accuracy'],d['delta']['roc_auc'],d['seconds']))" 2>/dev/null || echo FAILED)"
  done
done

#!/usr/bin/env bash
# CAPGD context-poisoning sweep: TabPFNv2, n_estimators=1, --simple,
# context_5000.csv -> test_attack_1000.csv, GPU 1.
#
#   Phase 1  1 row, 10 iters : constraints full | none | none @ eps 5 | none @ eps 50
#   Phase 2  1 row, 100 iters: constraints full | none   (= the n=1 entries of phase 3)
#            plus none @ eps 50 for 100 iters -- at eps 50 CAPGD starts with a step
#            of 2*eps = 95 feature-ranges, which overshoots so hard that nothing
#            improves within 10 iterations and the attack returns the clean row.
#            The extra iterations let the adaptive halving anneal down to a usable
#            scale, so the high-eps result reflects the budget and not that artifact.
#   Phase 3  ablation over the number of poisoned rows {1,2,4,8,16,32,64,128},
#            each with constraints full and none, 100 iters
#
# The whole 1000-row test set fits in one batch on all three datasets, so no
# --test-batch-size is passed and every loss is over the full test set.
# Completed runs are skipped, so the script can be re-run to resume.
set -u

cd /home/ssn899/Desktop/TabFM
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tabfm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=1
DATASETS=(url_unique wids lcld_v2)
ROWS=(1 2 4 8 16 32 64 128)
HIGH_EPS=50
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
START=$(date +%s)
TOTAL=0; DONE=0; SKIP=0; FAIL=0

run () {                      # run <dataset> <name> <extra args...>
  local ds=$1 name=$2; shift 2
  local out="results/$ds/attack/$name"
  TOTAL=$((TOTAL+1))
  if [ -f "$out/attack.json" ]; then
    SKIP=$((SKIP+1)); echo "  [skip] $ds/$name"; return
  fi
  local t0=$(date +%s)
  python scripts/attack_context.py \
      --gpu "$GPU" --simple --n-estimators 1 \
      --train "datasets/$ds/splits/context_5000.csv" \
      --test  "datasets/$ds/splits/test_attack_1000.csv" \
      "$@" --out "$out" > "$LOGDIR/${ds}__${name}.log" 2>&1
  local rc=$? t1=$(date +%s)
  if [ $rc -eq 0 ] && [ -f "$out/attack.json" ]; then
    DONE=$((DONE+1))
    echo "  [ok]   $ds/$name  ($((t1-t0))s)  $(grep -m1 'delta    loss' "$LOGDIR/${ds}__${name}.log" | sed 's/^ *//')"
  else
    FAIL=$((FAIL+1))
    echo "  [FAIL] $ds/$name  (rc=$rc)  -- see $LOGDIR/${ds}__${name}.log"
    tail -3 "$LOGDIR/${ds}__${name}.log" | sed 's/^/           /'
  fi
}

echo "################ PHASE 1: 1 row, 10 iterations ################"
for ds in "${DATASETS[@]}"; do
  run "$ds" p1_full            --n-rows 1 --constraints full
  run "$ds" p1_none            --n-rows 1 --constraints none
  run "$ds" p1_none_eps5           --n-rows 1 --constraints none --eps 5
  run "$ds" p1_none_eps${HIGH_EPS} --n-rows 1 --constraints none --eps "$HIGH_EPS"
done

echo "################ PHASE 2b: very high eps, 100 iterations ################"
for ds in "${DATASETS[@]}"; do
  run "$ds" p2_none_eps${HIGH_EPS}_it100 --n-rows 1 --constraints none --eps "$HIGH_EPS" --n-iter 100
done

echo "################ PHASE 2+3: row ablation, 100 iterations ################"
echo "# (the n=1 entries are the requested 100-iteration reruns of phase 1)"
for ds in "${DATASETS[@]}"; do
  for n in "${ROWS[@]}"; do
    printf -v nn "%03d" "$n"
    run "$ds" "abl_rows${nn}_full_it100" --n-rows "$n" --constraints full --n-iter 100
    run "$ds" "abl_rows${nn}_none_it100" --n-rows "$n" --constraints none --n-iter 100
  done
done

echo "################ DONE ################"
echo "  total $TOTAL   ok $DONE   skipped $SKIP   failed $FAIL   elapsed $(( ($(date +%s)-START)/60 )) min"

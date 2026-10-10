#!/usr/bin/env bash
# =============================================================================
#  TabPFNv2 robustness to label perturbation on the MULTICLASS datasets
#
#  2 datasets (students_dropout: 3 classes; gas-drift: 6 classes)
#  x 3 attacks x R in {0, 4, 8, ..., 40}  =  66 conditions.
#
#    label-flip-bernoulli  each row, with probability R, relabelled to a uniformly
#                          random other class                     -> 10 runs, averaged
#    hub-flip-local        the R% hubs, each relabelled to its locally closest other
#                          class (deterministic)                  -> 1 run
#    hub-flip-random       the SAME hub rows, relabelled to uniformly random other
#                          classes                                -> 10 runs, averaged
#
#  hub-flip-local vs hub-flip-random poison identical rows, so their gap is what the
#  COHERENCE of the poison contributes (see tabfm_experiments/multiclass_attacks.py).
#
#  Context: datasets/<ds>/splits/context_<M>.csv (M = 3000 / 5000). Scored on
#  test_attack_<N>.csv (N = 500 / 1000). The validation set is NOT used.
#
#  Reported per condition: cross-entropy, accuracy, balanced accuracy, multiclass MCC,
#  one-vs-rest macro ROC-AUC -- mean +- std over the runs. Audit afterwards with
#      python scripts/verify_multiclass.py
#
#  Cost: a few seconds per condition (model load dominates); the whole grid is
#  ~15-20 min on one GPU. Re-running skips every condition whose summary.json exists
#  and parses; each condition takes a lock dir so two shells cannot collide on one.
#
#    bash scripts/run_multiclass_attacks.sh                  # everything, GPU 1
#    DRY_RUN=1 bash scripts/run_multiclass_attacks.sh        # preview
#    GPU=0 DATASETS=gas-drift bash scripts/run_multiclass_attacks.sh
# =============================================================================

GPU=${GPU:-1}
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
OUT_ROOT=${OUT_ROOT:-$REPO/results/multiclass}
N_RUNS_STOCHASTIC=10
DATASETS=${DATASETS:-"students_dropout gas-drift"}
ATTACKS=${ATTACKS:-"label-flip-bernoulli hub-flip-local hub-flip-random"}
ROW_PERCENTS=${ROW_PERCENTS:-"0 4 8 12 16 20 24 28 32 36 40"}

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

done_ok () {   # summary.json exists and is valid JSON
  [[ -f "$1" ]] && "${PY[@]}" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1
}

run () {   # run <dataset> <attack> <row_percent>
  local ds=$1 atk=$2 rp=$3
  local tag; tag=$(printf "%s_r%03d" "$atk" "$rp")
  local out="$OUT_ROOT/$ds/$tag"
  local nruns=$N_RUNS_STOCHASTIC
  [[ "$atk" == "hub-flip-local" ]] && nruns=1      # deterministic: extra runs are identical

  if done_ok "$out/summary.json"; then echo "[skip] $ds/$tag (already done)"; return 0; fi

  local cmd=("${PY[@]}" "$REPO/scripts/attack_multiclass.py" --gpu "$GPU"
             --dataset "$ds" --attack "$atk" --row-percent "$rp" --n-runs "$nruns"
             --out "$OUT_ROOT")
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

  mkdir -p "$out"
  if ! mkdir "$out/.lock" 2>/dev/null; then
    echo "[busy] $ds/$tag (another process holds $out/.lock; delete it if that process died)"
    return 0
  fi
  echo "[$(date +'%F %T')] START $ds/$tag"
  "${cmd[@]}" > "$out/stdout.log" 2>&1
  local rc=$?
  rmdir "$out/.lock" 2>/dev/null
  if [[ $rc -eq 0 ]]; then echo "[$(date +'%F %T')] DONE  $ds/$tag"
  else echo "[$(date +'%F %T')] FAIL  $ds/$tag (rc=$rc, see $out/stdout.log)"; fi
  return 0
}

for ds in $DATASETS; do
  for atk in $ATTACKS; do
    for rp in $ROW_PERCENTS; do
      run "$ds" "$atk" "$rp"
    done
  done
done
echo "ALL REQUESTED RUNS FINISHED"

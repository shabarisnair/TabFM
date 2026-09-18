#!/usr/bin/env bash
# =============================================================================
#  XGBoost transfer of the TabPFNv2 context poison
#
#  For every finished attack run under results/main_experiments, rebuild its five
#  per-run-best poisoned contexts, train XGBoost on each from scratch (optuna HPO on
#  validation logloss), and score on the same test_attack_1000.csv.
#
#  4 datasets x { x-capgd , label-flip } x R in {1,4,16,64} % = 32 commands,
#  each covering BOTH validation protocols:
#     A  validation is an 80/20 internal split of the given context
#        -> a poisoned context also gives a poisoned validation set
#     B  validation is the separate clean datasets/<ds>/splits/val_2000.csv
#
#  CPU only -- this does not touch the GPU, so it is safe to run alongside the
#  TabPFNv2 attacks.
#
#  HOW TO USE
#    1. Edit the CONFIG block below.
#    2. bash scripts/run_xgb_transfer.sh          (DRY_RUN=1 to preview)
#    3. To split across servers: copy CONFIG + run() and paste only the `run` lines
#       you want. Each line is independent.
#
#  Re-running is safe: a command whose output already has summary.json is skipped.
#  An attack dir that has not finished yet (no summary.json) is skipped with a note,
#  so this file can be run repeatedly while the attacks are still going.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
ATTACK_ROOT=$REPO/results/main_experiments      # where attack_context.py wrote its runs
OUT_ROOT=$REPO/results/xgb_transfer

HPO_TRIALS=30           # optuna trials per fitted model (objective: validation logloss)
N_JOBS=16               # xgboost threads per fit
DEVICE=cpu              # or cuda
VAL_RATIO=0.2           # protocol A split -> 80/20

# Clean baselines depend only on (train, test, val, protocol, hpo settings), so they are
# identical for all 8 runs of a dataset. Caching them cuts the work roughly in half.
# Set to "" to recompute per command.
CLEAN_CACHE="--clean-cache-dir $OUT_ROOT/_clean_cache"

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

run () {   # run <dataset_dir> <x-capgd|label-flip> <row_percent>
  local ds=$1 atk=$2 rp=$3
  local tag; tag=$(printf "%s_random_r%03d" "$atk" "$rp")
  local adir="$ATTACK_ROOT/$ds/$tag"
  local out="$OUT_ROOT/$ds/$tag"

  if [[ ! -f "$adir/summary.json" ]]; then
    echo "[wait] $ds/$tag (attack not finished: no $adir/summary.json)"; return 0
  fi
  if [[ -f "$out/summary.json" ]]; then
    echo "[skip] $ds/$tag (already done)"; return 0
  fi

  local cmd=("${PY[@]}" "$REPO/scripts/xgb_transfer.py"
             --model xgboost --attack-dir "$adir" --out "$out"
             --val "$REPO/datasets/$ds/splits/val_2000.csv"
             --val-ratio "$VAL_RATIO" --protocols A B
             --hpo-trials "$HPO_TRIALS" --n-jobs "$N_JOBS" --device "$DEVICE"
             $CLEAN_CACHE)

  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

  mkdir -p "$out"
  echo "[$(date +'%F %T')] START $ds/$tag"
  "${cmd[@]}" > "$out/stdout.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +'%F %T')] DONE  $ds/$tag"
  else
    echo "[$(date +'%F %T')] FAIL  $ds/$tag (rc=$rc, see $out/stdout.log)"
  fi
  return 0        # keep going even if one run fails
}

# ============================== COMMANDS =====================================
# Cost per command: 2 protocols x (5 clean + 5 poisoned) fits, each fit = HPO_TRIALS
# short XGBoost trainings plus a refit. MEASURED on lcld_v2 (5000 x 28) at
# HPO_TRIALS=30, N_JOBS=16: ~13 s per fit -> ~2 min per command with the clean cache
# warm, ~4.5 min cold. wids (5000 x 108) is roughly 3x that. Whole file well under
# 2 CPU-hours.

# --------------------------------------------------------------------------
# BLOCK 1of8: coil2000  X_train CAPGD
# --------------------------------------------------------------------------
run coil2000_insurance_policies x-capgd 1
run coil2000_insurance_policies x-capgd 4
run coil2000_insurance_policies x-capgd 16
run coil2000_insurance_policies x-capgd 32
run coil2000_insurance_policies x-capgd 50
run coil2000_insurance_policies x-capgd 64

# --------------------------------------------------------------------------
# BLOCK 2of8: coil2000  Y_train label-flip (GA)
# --------------------------------------------------------------------------
run coil2000_insurance_policies label-flip 1
run coil2000_insurance_policies label-flip 4
run coil2000_insurance_policies label-flip 16
run coil2000_insurance_policies label-flip 32
run coil2000_insurance_policies label-flip 50
run coil2000_insurance_policies label-flip 64

# --------------------------------------------------------------------------
# BLOCK 3of8: lcld_v2   X_train CAPGD
# --------------------------------------------------------------------------
run lcld_v2 x-capgd 1
run lcld_v2 x-capgd 4
run lcld_v2 x-capgd 16
run lcld_v2 x-capgd 32
run lcld_v2 x-capgd 50
run lcld_v2 x-capgd 64

# --------------------------------------------------------------------------
# BLOCK 4of8: lcld_v2   Y_train label-flip (GA)
# --------------------------------------------------------------------------
run lcld_v2 label-flip 1
run lcld_v2 label-flip 4
run lcld_v2 label-flip 16
run lcld_v2 label-flip 32
run lcld_v2 label-flip 50
run lcld_v2 x-capgd 32
run lcld_v2 x-capgd 50
run lcld_v2 label-flip 64

# --------------------------------------------------------------------------
# BLOCK 5of8: url_unique  X_train CAPGD
# --------------------------------------------------------------------------
run url_unique x-capgd 1
run url_unique x-capgd 4
run url_unique x-capgd 16
run url_unique x-capgd 32
run url_unique x-capgd 50
run url_unique x-capgd 64

# --------------------------------------------------------------------------
# BLOCK 6of8: url_unique  Y_train label-flip (GA)
# --------------------------------------------------------------------------
run url_unique label-flip 1
run url_unique label-flip 4
run url_unique label-flip 16
run url_unique label-flip 32
run url_unique label-flip 50
run url_unique label-flip 64

run wids x-capgd 1
run wids x-capgd 4
run wids x-capgd 16
run wids x-capgd 32
run wids x-capgd 50
run wids x-capgd 64

run wids label-flip 1
run wids label-flip 4
run wids label-flip 16
run wids label-flip 32
run wids label-flip 50
run wids label-flip 64

echo "ALL REQUESTED RUNS FINISHED"

#!/usr/bin/env bash
# =============================================================================
#  Test-agnostic random label-flip grid
#
#  4 datasets x R in {1,4,16,32,50,64} % = 24 commands.
#  Each command flips a uniformly random R% of the context labels, with a FRESH random
#  subset for each of the N_RUNS runs. The test set is never used to choose rows --
#  only to score the poisoned context.
#
#  Context: datasets/<ds>/splits/selected/natural/context_5000.csv  (same as main grid)
#  Test:    datasets/<ds>/splits/$TEST_FILE  (default test_attack_1000.csv, so the
#           numbers line up with the main tables)
#
#  AGGREGATION: one draw per run (n_row_subsamples is fixed at 1 for this attack), so
#  summary.json's aggregate_delta / aggregate_poisoned are the mean +- std over all
#  N_RUNS independent draws. No best-of selection -- a best-of pick by test CE would
#  let the test set back in.
#
#  HOW TO USE
#    1. Edit the CONFIG block below (GPU id is the main one).
#    2. bash scripts/run_random_flip_experiments.sh        (DRY_RUN=1 to preview)
#    3. To split across servers/GPUs: copy CONFIG + run() and paste only the `run`
#       lines you want. Each line is independent.
#
#  Re-running is safe: a command is skipped when its summary.json exists AND parses.
#  A truncated summary.json (process killed mid-write) is treated as not done. Each
#  command also takes a lock dir, so two shells cannot run the same cell at once.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=0                                   # <-- GPU id on THIS server
REPO=/home/ssn899/Desktop/TabFM         # <-- repo root on THIS server
CONDA_ENV=tabfm                         # <-- conda env name
OUT_ROOT=$REPO/results/main_experiments # <-- same root as the main grid, so the
                                        #     table scripts find these runs too
N_RUNS=15                               # independent random draws per (dataset, R)
TEST_FILE=test_attack_1000.csv          # or test.csv to score on the full held-out split
                                        # (then also change OUT_ROOT, the numbers differ)

# Optional: write the single highest-dCE draw as context_poisoned.csv. Off by default:
# picking "the highest" uses the test set, and every draw is already saved as
# trials/run<r>_sub0_delta.npz (which is what the XGBoost transfer rebuilds from).
SAVE_CONTEXT=""                         # or "--save-full-context"

# How python is invoked. Change if conda lives elsewhere on that server, e.g.
#   PY=(/path/to/envs/tabfm/bin/python)
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

done_ok () {   # summary.json exists and is valid JSON
  [[ -f "$1" ]] && "${PY[@]}" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1
}

run () {   # run <dataset_dir> <row_percent>
  local ds=$1 rp=$2
  local tag; tag=$(printf "label-flip-random_random_r%03d" "$rp")
  local out="$OUT_ROOT/$ds/$tag"

  if done_ok "$out/summary.json"; then
    echo "[skip] $ds/$tag (already done)"; return 0
  fi

  local cmd=("${PY[@]}" "$REPO/scripts/attack_context.py"
             --gpu "$GPU" --attack label-flip-random --row-percent "$rp" --n-runs "$N_RUNS"
             --train "$REPO/datasets/$ds/splits/selected/natural/context_5000.csv"
             --test  "$REPO/datasets/$ds/splits/$TEST_FILE"
             --out   "$out" $SAVE_CONTEXT)

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
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +'%F %T')] DONE  $ds/$tag"
  else
    echo "[$(date +'%F %T')] FAIL  $ds/$tag (rc=$rc, see $out/stdout.log)"
  fi
  return 0        # keep going even if one run fails
}

# ============================== COMMANDS =====================================
# Cost: one no-grad forward per draw, so 15 draws + the clean baseline take well under
# a minute per command on lcld_v2; wids is the slowest. Peak memory is the no-grad
# forward only (wids ~2-8 GB), so no --recompute-layers is needed here -- the 74 GB
# wids peak seen before came from the GA's influence-seeding GRADIENT pass, which this
# attack does not run.

# --------------------------------------------------------------------------
# BLOCK 1of4: coil2000
# --------------------------------------------------------------------------
run coil2000_insurance_policies 1
run coil2000_insurance_policies 4
run coil2000_insurance_policies 16
run coil2000_insurance_policies 32
run coil2000_insurance_policies 50
run coil2000_insurance_policies 64

# --------------------------------------------------------------------------
# BLOCK 2of4: lcld_v2
# --------------------------------------------------------------------------
run lcld_v2 1
run lcld_v2 4
run lcld_v2 16
run lcld_v2 32
run lcld_v2 50
run lcld_v2 64

# --------------------------------------------------------------------------
# BLOCK 3of4: url_unique
# --------------------------------------------------------------------------
run url_unique 1
run url_unique 4
run url_unique 16
run url_unique 32
run url_unique 50
run url_unique 64

# --------------------------------------------------------------------------
# BLOCK 4of4: wids
# --------------------------------------------------------------------------
run wids 1
run wids 4
run wids 16
run wids 32
run wids 50
run wids 64

echo "ALL REQUESTED RUNS FINISHED"

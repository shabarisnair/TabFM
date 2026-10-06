#!/usr/bin/env bash
# =============================================================================
#  Random label-flip: fill url_unique + coil2000 onto the DENSE R ladder
#
#  WHY: heloc, credit-approval and lcld_v2 were swept at R = 0,2,4,...,40, but
#  url_unique and coil2000 only ever got the sparse early grid R in {1,4,16,32,50,64}.
#  That leaves them with 3 usable points below R=40 -- too few to draw next to the
#  21-point curves. This script runs only the ladder values they are missing.
#
#  Already present for both (skipped automatically): R = 4, 16, 32.
#  Run here: R = 0,2,6,8,10,12,14,18,20,22,24,26,28,30,34,36,38,40  = 18 per dataset.
#
#  R=0 flips nothing (k=0) and reproduces the clean baseline -- it is the ladder's
#  left anchor, and is only meaningful for this attack. Note the older R=1/50/64 runs
#  are left alone; they stay in the CSV and are simply outside the 0-40 window.
#
#  Settings are pinned to match the runs already on disk, so the new points are
#  directly comparable: same context/test splits, N_RUNS=15, same OUT_ROOT.
#
#  HOW TO USE
#    bash scripts/run_url_coil_random_flips.sh                     # both, GPU 0
#    DRY_RUN=1 bash scripts/run_url_coil_random_flips.sh           # preview only
#    # two GPUs, in parallel:
#    GPU=0 DATASETS=url_unique                  bash scripts/run_url_coil_random_flips.sh &
#    GPU=3 DATASETS=coil2000_insurance_policies bash scripts/run_url_coil_random_flips.sh &
#
#  Cost: ~25 s per command (dominated by model load; the 15 draws are sub-second),
#  so ~8 min per dataset, ~15 min for both on one GPU.
#
#  Re-running is safe: a command is skipped when its summary.json exists AND parses,
#  and each command takes a lock dir so two shells cannot collide on one cell.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=${GPU:-0}                           # <-- GPU id (env-overridable, see above)
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
OUT_ROOT=$REPO/results/main_experiments # same root as every other random-flip run
N_RUNS=15                               # MUST stay 15: matches the runs already on disk
TEST_FILE=test_attack_1000.csv

DATASETS=${DATASETS:-"url_unique coil2000_insurance_policies"}
ROW_PERCENTS=${ROW_PERCENTS:-"0 2 6 8 10 12 14 18 20 22 24 26 28 30 34 36 38 40"}

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
             --out   "$out")

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
  return 0        # keep going even if one cell fails
}

# ============================== COMMANDS =====================================
for ds in $DATASETS; do
  echo "=== $ds  (GPU $GPU) ==="
  for rp in $ROW_PERCENTS; do
    run "$ds" "$rp"
  done
done

echo "ALL REQUESTED RUNS FINISHED"

#!/usr/bin/env bash
# =============================================================================
#  Label-flip attacks on heloc and credit-approval  --  PART B of 2
#
#  This half runs R = 2, 6, 10, 14, 18, 22, 26, 30, 34, 38.
#  The other half is run_heloc_credit_flips_a.sh. The two R sets are disjoint,
#  so the scripts can run concurrently on different GPUs and never collide.
#  Both write to the same OUT_ROOT; each condition also takes a .lock directory,
#  so even an accidental overlap would be skipped rather than raced.
#
#  2 datasets x { random flip , hub flip } x 10 R values = 40 commands (84 across both halves).
#  R=0 is the clean baseline (k=0 rows flipped) and is a real condition for both
#  attacks, so the sweep starts from an exact zero rather than an implied one.
#
#  RANDOM FLIP  test-agnostic, uniformly random rows, a fresh draw per run.
#               Stochastic -> N_RUNS_RANDOM=10, reported as mean +- std over the draws.
#  HUB FLIP     test-agnostic, the R% of rows with the smallest mean k-NN distance.
#               Deterministic -> 1 run; more would re-record identical flips.
#
#  PATHS DIFFER BY DATASET. heloc follows the usual protocol (selected 5000-row
#  context, 1000-row attack test set). credit-approval is only 653 rows and was
#  prepared with a plain 70/30 split, so it uses train.csv / test.csv directly.
#
#  OUTPUT ROOTS are chosen so the existing table scripts read them unchanged:
#    $OUT_ROOT/main_experiments  ->  scripts/make_main_tables.py --root ...
#    $OUT_ROOT/hub_flip          ->  scripts/make_hub_tables.py  --root ...
#
#  HOW TO USE
#    1. Edit the CONFIG block.
#    2. bash scripts/run_heloc_credit_flips.sh       (DRY_RUN=1 to preview)
#  Re-running is safe: a condition whose summary.json exists AND parses is skipped.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=3
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
OUT_ROOT=$REPO/results

DATASETS=(heloc credit-approval)
R_VALUES=(2 6 10 14 18 22 26 30 34 38)

N_RUNS_RANDOM=15         # random flip is stochastic; hub flip is deterministic (always 1)
HUB_METRIC=knn           # mean distance to the k nearest others
HUB_K=10                 # neighbours used by that score

# Attention recording for the hub runs. OFF by default: this sweep is about the
# metrics, and recording costs ~38 GiB and roughly doubles the runtime.
# To turn it on:  RECORD_ATTENTION="--columns all --topk 256 --full-target-col"
RECORD_ATTENTION="--no-attention"

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

done_ok () { [[ -f "$1" ]] && "${PY[@]}" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

ctx_for () {   # the context / train file for a dataset
  case "$1" in
    credit-approval) echo "$REPO/datasets/$1/splits/train.csv" ;;
    *)               echo "$REPO/datasets/$1/splits/selected/natural/context_5000.csv" ;;
  esac
}
test_for () {  # the query / test file for a dataset
  case "$1" in
    credit-approval) echo "$REPO/datasets/$1/splits/test.csv" ;;
    *)               echo "$REPO/datasets/$1/splits/test_attack_1000.csv" ;;
  esac
}

launch () {    # launch <label> <out_dir> <cmd...>
  local label=$1 out=$2; shift 2
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "$@"; echo; return 0; fi
  mkdir -p "$out"
  if ! mkdir "$out/.lock" 2>/dev/null; then
    echo "[busy] $label (another process holds $out/.lock)"; return 0
  fi
  echo "[$(date +'%F %T')] START $label"
  "$@" > "$out/stdout.log" 2>&1
  local rc=$?
  rmdir "$out/.lock" 2>/dev/null
  if [[ $rc -eq 0 ]]; then echo "[$(date +'%F %T')] DONE  $label"
  else echo "[$(date +'%F %T')] FAIL  $label (rc=$rc, see $out/stdout.log)"; fi
  return 0
}

run_random () {   # run_random <dataset> <row_percent>
  local ds=$1 rp=$2
  local tag; tag=$(printf "label-flip-random_random_r%03d" "$rp")
  local out="$OUT_ROOT/main_experiments/$ds/$tag"
  if done_ok "$out/summary.json"; then echo "[skip] $ds random r$rp"; return 0; fi
  launch "$ds random r$rp" "$out" \
    "${PY[@]}" "$REPO/scripts/attack_context.py" \
      --gpu "$GPU" --attack label-flip-random --row-percent "$rp" --n-runs "$N_RUNS_RANDOM" \
      --train "$(ctx_for "$ds")" --test "$(test_for "$ds")" --out "$out"
}

run_hub () {      # run_hub <dataset> <row_percent>
  local ds=$1 rp=$2
  local out; out="$OUT_ROOT/hub_flip/$ds/$(printf 'r%03d' "$rp")"
  if done_ok "$out/summary.json"; then echo "[skip] $ds hub r$rp"; return 0; fi
  launch "$ds hub r$rp" "$out" \
    "${PY[@]}" "$REPO/scripts/hub_flip_attention.py" \
      --gpu "$GPU" --dataset "$ds" --row-percent "$rp" --n-runs 1 \
      --train "$(ctx_for "$ds")" --test "$(test_for "$ds")" \
      --hub-metric "$HUB_METRIC" --hub-k "$HUB_K" \
      --out "$OUT_ROOT/hub_flip" $RECORD_ATTENTION
}

# ============================== COMMANDS =====================================
# MEASURED per condition on GPU 0, including model load (conda run adds ~1-2 s):
#   heloc  random (10 runs) 8.7 s      heloc  hub (1 run) 4.8 s
#   credit random (10 runs) 4.9 s      credit hub (1 run) 4.5 s
# This half: 40 commands => about 5 minutes. Run both halves in parallel on
# two GPUs and the whole sweep finishes in roughly that time.
# With attention on, add roughly 38 GiB and expect the runtime to roughly double.

for ds in "${DATASETS[@]}"; do
  for rp in "${R_VALUES[@]}"; do
    run_random "$ds" "$rp"
  done
  for rp in "${R_VALUES[@]}"; do
    run_hub "$ds" "$rp"
  done
done

echo "PART B FINISHED  (R = 2 6 10 14 18 22 26 30 34 38)"
echo
echo "Tables (run only after BOTH halves have finished):"
echo "  python scripts/make_main_tables.py --root $OUT_ROOT/main_experiments --datasets heloc credit-approval"
echo "  python scripts/make_hub_tables.py  --root $OUT_ROOT/hub_flip"

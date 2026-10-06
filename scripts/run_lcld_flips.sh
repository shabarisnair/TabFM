#!/usr/bin/env bash
# =============================================================================
#  Label-flip attacks on lcld_v2  --  the same sweep run on heloc / credit-approval
#
#  { random flip , hub flip } x R in {0,2,4,...,40} = 42 commands.
#  R=0 is the clean baseline (k=0 rows flipped), a real condition for both attacks.
#
#  RANDOM FLIP  test-agnostic, uniformly random rows, a fresh draw per run. Stochastic,
#               so N_RUNS_RANDOM draws are averaged (mean +- std, no selection).
#  HUB FLIP     test-agnostic, the R% of rows with the smallest mean k-NN distance.
#               Deterministic -> 1 run; more would re-record identical flips.
#
#  WHY N_RUNS_RANDOM=15 AND NOT 10
#  lcld_v2 ALREADY has label-flip-random results at R = 1, 4, 16, 32, 50, 64, all with
#  n_runs=15, in this same RANDOM_ROOT. R = 4, 16 and 32 fall inside this ladder, so at
#  15 they are correctly reused (skipped) and the whole lcld_v2 curve stays uniform --
#  and R = 1, 50, 64 come along as free extra points. Setting 10 here would instead
#  produce a curve mixing 15-run and 10-run conditions. If you do want 10, also change
#  RANDOM_ROOT to somewhere fresh so the existing runs are not silently reused.
#
#  HOW TO USE
#    1. Edit the CONFIG block.
#    2. bash scripts/run_lcld_flips.sh          (DRY_RUN=1 to preview)
#  Re-running is safe: a condition whose summary.json exists AND parses is skipped.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=0
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm

DATASET=lcld_v2
R_VALUES=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40)

RANDOM_ROOT=$REPO/results/main_experiments   # read by scripts/make_main_tables.py
HUB_ROOT=$REPO/results/hub_flip              # read by scripts/make_hub_tables.py

N_RUNS_RANDOM=15         # see the note above before changing this
HUB_METRIC=knn           # mean distance to the k nearest others
HUB_K=10                 # neighbours used by that score

# Attention recording for the hub runs. OFF by default: this sweep is about the metrics.
# To turn it on:  RECORD_ATTENTION="--columns all --topk 256 --full-target-col"
RECORD_ATTENTION="--no-attention"

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

CTX="$REPO/datasets/$DATASET/splits/selected/natural/context_5000.csv"
TST="$REPO/datasets/$DATASET/splits/test_attack_1000.csv"
for f in "$CTX" "$TST"; do [[ -f "$f" ]] || { echo "missing input: $f"; exit 1; }; done

done_ok () { [[ -f "$1" ]] && "${PY[@]}" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

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

run_random () {   # run_random <row_percent>
  local rp=$1
  local tag; tag=$(printf "label-flip-random_random_r%03d" "$rp")
  local out="$RANDOM_ROOT/$DATASET/$tag"
  if done_ok "$out/summary.json"; then echo "[skip] random r$rp"; return 0; fi
  launch "random r$rp" "$out" \
    "${PY[@]}" "$REPO/scripts/attack_context.py" \
      --gpu "$GPU" --attack label-flip-random --row-percent "$rp" --n-runs "$N_RUNS_RANDOM" \
      --train "$CTX" --test "$TST" --out "$out"
}

run_hub () {      # run_hub <row_percent>
  local rp=$1
  local out; out="$HUB_ROOT/$DATASET/$(printf 'r%03d' "$rp")"
  if done_ok "$out/summary.json"; then echo "[skip] hub r$rp"; return 0; fi
  launch "hub r$rp" "$out" \
    "${PY[@]}" "$REPO/scripts/hub_flip_attention.py" \
      --gpu "$GPU" --dataset "$DATASET" --row-percent "$rp" --n-runs 1 \
      --train "$CTX" --test "$TST" \
      --hub-metric "$HUB_METRIC" --hub-k "$HUB_K" \
      --out "$HUB_ROOT" $RECORD_ATTENTION
}

# ============================== COMMANDS =====================================
# MEASURED per condition on GPU 0, including model load (conda run adds ~1-2 s):
#   random (15 runs) 10.9 s        hub (1 run) 4.6 s
# 42 commands, of which 3 random ones (R=4,16,32) already exist and are skipped:
#   18 x 10.9 + 21 x 4.6 ~= 5 minutes, a few MB on disk.

for rp in "${R_VALUES[@]}"; do run_random "$rp"; done
for rp in "${R_VALUES[@]}"; do run_hub    "$rp"; done

echo "ALL REQUESTED RUNS FINISHED"
echo
echo "Tables:"
echo "  python scripts/make_main_tables.py --root $RANDOM_ROOT --datasets $DATASET"
echo "  python scripts/make_hub_tables.py  --root $HUB_ROOT"

#!/usr/bin/env bash
# =============================================================================
#  k-fold self-consistency defence against label-poisoned contexts
#
#  QUESTION: knowing only the corruption rate R -- not which rows were hit, nor how the
#  attacker chose them, and with no access to the test set -- can we repair the context?
#
#  METHOD (scripts/tabfm_experiments/defense.py): split the context into 20 folds, score
#  every row by p(it carries the label it was given) with its own fold held out, drop the
#  R% that score lowest, refit, and only THEN touch the test set.
#
#  GRID: 5 datasets x R in {5,10,20,30,40} x {random, hub} = 50 commands.
#
#  Each command is a CROSSED design over the two independent randomness sources, so the
#  aggregate can attribute the spread instead of confounding it. Every poison draw is
#  defended under the same set of fold seeds:
#    * label-flip-random -- N_POISON_DRAWS x N_FOLD_DRAWS = 5 x 5 = 25 cells.
#    * hub-flip          -- the attack is deterministic, so 1 x 5 = 5 cells; all of the
#                           spread there is the fold draw's doing.
#  summary.json reports, per metric: overall mean/std, "between_poison_std" (how much the
#  ATTACK draw moves it) and "within_poison_fold_std" (how much the DEFENCE's own fold
#  draw moves it with the poison held fixed).
#
#  Each command writes, per run: the detection scores (precision / recall / accuracy /
#  ranking AUC, overall AND per class) and the test-set metrics for four contexts --
#  clean, poisoned, defended, and oracle (perfect removal: the ceiling for any detector
#  at this R, and the control for removal also shrinking the context).
#
#  R=5 is NOT in the attack sweeps. The poison is regenerated from the attacks' own code
#  paths, so any R works; where a sweep run does exist the regenerated rows are checked
#  against the stored ones and the verdict lands in summary.json under
#  "poison_reproduction_check". Spot-checked: credit-approval hub R=10 and url_unique
#  random R=10 both MATCH.
#
#  HOW TO USE
#    bash scripts/run_defense_kfold.sh                    # everything, GPU 0
#    DRY_RUN=1 bash scripts/run_defense_kfold.sh          # preview the commands
#    # split across two GPUs (disjoint work):
#    GPU=0 ATTACKS=label-flip-random bash scripts/run_defense_kfold.sh &
#    GPU=3 ATTACKS=hub-flip          bash scripts/run_defense_kfold.sh &
#
#  Cost: ~15 s per run on a 5000-row context (20 folds + 4 scored contexts), so ~80 s per
#  command at N_RUNS=5. Whole grid ~60-75 min on one GPU, ~35 min split across two;
#  credit-approval (457 rows) is far faster than the rest, coil2000 (85 features) slowest.
#
#  Re-running is safe: a command is skipped when its summary.json exists AND parses, and
#  each command takes a lock dir so two shells cannot collide on one cell.
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=${GPU:-0}                             # <-- GPU id (env-overridable, see above)
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
OUT_ROOT=${OUT_ROOT:-$REPO/results/defense_kfold}

N_FOLDS=50                                # the k in "k-fold"
N_POISON_DRAWS=5                          # attack randomness (forced to 1 for hub-flip)
N_FOLD_DRAWS=5                            # defence randomness: fold reshuffles per poison
FOLD_SEED=0
ROW_SEED=1                                # MUST be 1: matches Seeds.row_seed, so the
                                          # regenerated flips equal the sweeps' flips
HUB_METRIC=knn
HUB_K=10

DATASETS=${DATASETS:-"credit-approval url_unique coil2000_insurance_policies heloc lcld_v2"}
ROW_PERCENTS=${ROW_PERCENTS:-"5 10 20 30 40"}
ATTACKS=${ATTACKS:-"label-flip-random hub-flip"}

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

done_ok () {   # summary.json exists and is valid JSON
  [[ -f "$1" ]] && "${PY[@]}" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1
}

run () {   # run <dataset> <attack> <row_percent>
  local ds=$1 atk=$2 rp=$3
  local tag; tag=$(printf "%s_r%03d" "$atk" "$rp")
  local out="$OUT_ROOT/$ds/$tag"

  # hub-flip is deterministic, so extra poison draws would be byte-identical;
  # defend_context.py rejects --n-poison-draws > 1 for it outright.
  local npoison=$N_POISON_DRAWS
  [[ "$atk" == "hub-flip" ]] && npoison=1

  if done_ok "$out/summary.json"; then
    echo "[skip] $ds/$tag (already done)"; return 0
  fi

  local cmd=("${PY[@]}" "$REPO/scripts/defend_context.py"
             --gpu "$GPU" --dataset "$ds" --attack "$atk" --row-percent "$rp"
             --method kfold --n-folds "$N_FOLDS"
             --n-poison-draws "$npoison" --n-fold-draws "$N_FOLD_DRAWS"
             --fold-seed "$FOLD_SEED" --row-seed "$ROW_SEED"
             --hub-metric "$HUB_METRIC" --hub-k "$HUB_K"
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
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +'%F %T')] DONE  $ds/$tag"
  else
    echo "[$(date +'%F %T')] FAIL  $ds/$tag (rc=$rc, see $out/stdout.log)"
  fi
  return 0        # keep going even if one cell fails
}

# ============================== COMMANDS =====================================
# Ordered dataset-major so a partial run still gives complete curves for the datasets it
# reached, rather than a fragment of every dataset.
for ds in $DATASETS; do
  for atk in $ATTACKS; do
    echo "=== $ds / $atk  (GPU $GPU) ==="
    for rp in $ROW_PERCENTS; do
      run "$ds" "$atk" "$rp"
    done
  done
done

echo "ALL REQUESTED RUNS FINISHED"

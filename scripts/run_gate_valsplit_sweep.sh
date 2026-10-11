#!/usr/bin/env bash
# =============================================================================
#  Context gating vs simply using the clean rows (scripts/gate_context.py)
#
#  Is gating worth it? If the small clean validation set, used directly as the
#  ICL context, recovers as much accuracy as gating the poisoned context does,
#  gating is pointless. Each run therefore:
#
#    1. draws a class-stratified random validation set of V rows from the
#       CLEAN context (--val-from-context V --val-seed S);
#    2. poisons the other rows (hub-flip-local, R%);
#    3. learns one gate per poisoned-context row on clean validation rows;
#    4. scores the test set every TEST_EVERY updates, soft (current gates) and
#       hard (the R% lowest-gated rows discarded);
#    5. scores, once, the references, among them
#         val_as_context     the V clean rows alone as the context  <- the baseline
#         poisoned_plus_val  the poisoned rows + the V clean rows
#         clean_context      the remaining rows with their clean labels
#         poisoned_no_gate, random removal, oracle gate, oracle removal
#       val_as_context is drawn as a flat line on every test curve (plots and
#       W&B overlay charts); test_metrics.json has vs_val_as_context =
#       (best soft gates) - (val_as_context) per metric.
#
#  Two strategies for choosing the best iteration:
#    full     the gates are fitted on all V rows and the best iteration is the
#             one with the highest accuracy on them (ties: lower CE, earlier)
#    holdout  a class-stratified HOLDOUT share (default 0.2) of the V rows is held
#             out as a validation-test set (--val-holdout): the gates are fitted on
#             the rest, the held-out rows are scored with the test set every
#             TEST_EVERY updates, and the best iteration is the one with the highest
#             held-out accuracy (ties: lower held-out CE, earlier). Run names end in
#             _H<percent>.
#  Either way, test_metrics.json's selection_rules records the iteration that
#  each stopping rule would pick and the test scores logged there.
#
#  Datasets (V up to MAX_V of each; a larger V would leave too small a context):
#    gas-drift        6 classes  context_5000.csv, test_attack_1000.csv
#    url_unique       binary     selected/natural/context_5000.csv, test_attack_1000.csv
#    heloc            binary     selected/natural/context_5000.csv, test_attack_1000.csv
#    credit-approval  binary     train.csv (457 rows), test.csv (196 rows); V <= 200
#    students_dropout 3 classes  context_3000.csv, test_attack_500.csv (50/32/18%)
#
#  Grid (when no JOBS_FILE is given): DATASETS x R in ROW_PERCENTS x V in VAL_SIZES
#  x VAL_SEEDS x STRATEGIES. One seed gives nested draws (V=200 contains V=100).
#  Optimiser: PGD, L-inf step scaling, lr 0.05, 300 updates (the best setting of
#  run_gate_sweep.sh).
#
#  JOBS_FILE: run an explicit list instead, one run per line,
#      <part> <dataset> <R> <V> <seed> <full|holdout>
#  (# comments and blank lines skipped). With PART=k only the lines of part k
#  run. scripts/gate_valsplit_parts/round2_jobs.txt is the second round, split
#  into 11 parts by scripts/gate_valsplit_parts/make_round2_jobs.py, and
#  run_part<k>.sh run them, one script per GPU.
#  Without JOBS_FILE, PART=k NPARTS=m runs the k-th of m shares of the grid,
#  dealt in snake order (1..m, m..1, ...).
#
#  Each run is logged to W&B project TFM_experiments, group gate-valsplit, as
#      gate-valsplit_<dataset>_<attack>_R<R>_V<V>s<seed>[_H20]_pgd-linf_lr<lr>_it<updates>
#  and writes its files to
#      results/gate_valsplit/<dataset>/<attack>_r<RRR>/<that name>/
#  (split.npz there holds the validation, held-out and context row indices into
#  the context file).
#
#  GPUs: one queue per GPU in GPUS (default "1 3"); each runs its share of the
#  runs one after another. A run needs up to ~11 GiB of GPU memory.
#
#  Re-running resumes: a run whose test_metrics.json exists is skipped. A run
#  being worked on holds <run dir>/.lock; delete it if that process died.
#
#  W&B key (online mode only): WANDB_API_KEY if exported, else the line
#  WANDB_API_KEY=<key> in .env in the repo root, which scripts/load_env.sh loads
#  (git-ignored, owner-only; never commit it, copy it to each machine). .env may
#  also set any of the settings below; an exported variable still wins.
#
#    bash scripts/run_gate_valsplit_sweep.sh                    # the whole grid, GPUs 1 and 3
#    DRY_RUN=1 bash scripts/run_gate_valsplit_sweep.sh          # print the commands only
#    VAL_SEEDS="0 1 2" bash scripts/run_gate_valsplit_sweep.sh  # three draws per size
#    GPUS="1" DATASETS="heloc" STRATEGIES="holdout" bash scripts/run_gate_valsplit_sweep.sh
#    JOBS_FILE=scripts/gate_valsplit_parts/round2_jobs.txt PART=3 GPUS="2" \
#        bash scripts/run_gate_valsplit_sweep.sh                # part 3 of round 2 on GPU 2
#
#  Every setting below can be overridden the same way (e.g. STEPS, TEST_EVERY,
#  WANDB_MODE=offline, OUT_ROOT, CONDA_ENV). REPO defaults to the checkout this script
#  sits in, so the script also runs from a copy of the repo on another machine.
# =============================================================================

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" "$REPO/.env"   # WANDB_API_KEY, and any setting below
CONDA_ENV=${CONDA_ENV:-tabfm}
OUT_ROOT=${OUT_ROOT:-$REPO/results/gate_valsplit}
GPUS=${GPUS:-"1 3"}
DATASETS=${DATASETS:-"gas-drift url_unique heloc credit-approval students_dropout"}
ROW_PERCENTS=${ROW_PERCENTS:-"10 20"}
VAL_SIZES=${VAL_SIZES:-"50 100 200 300 400 500"}
VAL_SEEDS=${VAL_SEEDS:-"0"}
STRATEGIES=${STRATEGIES:-"full holdout"}
HOLDOUT=${HOLDOUT:-0.2}
ATTACK=hub-flip-local
LR=${LR:-0.05}
STEPS=${STEPS:-300}
TEST_EVERY=${TEST_EVERY:-5}
WANDB_PROJECT=${WANDB_PROJECT:-TFM_experiments}
WANDB_GROUP=${WANDB_GROUP:-gate-valsplit}
WANDB_MODE=${WANDB_MODE:-online}
JOBS_FILE=${JOBS_FILE:-}  # explicit run list (see above); otherwise the grid
PART=${PART:-}            # with JOBS_FILE: that part's lines; with NPARTS: that share of the grid
NPARTS=${NPARTS:-1}

declare -A CONTEXT=(
  [gas-drift]=datasets/gas-drift/splits/context_5000.csv
  [url_unique]=datasets/url_unique/splits/selected/natural/context_5000.csv
  [heloc]=datasets/heloc/splits/selected/natural/context_5000.csv
  [credit-approval]=datasets/credit-approval/splits/train.csv
  [students_dropout]=datasets/students_dropout/splits/context_3000.csv
)
declare -A TEST=(
  [gas-drift]=datasets/gas-drift/splits/test_attack_1000.csv
  [url_unique]=datasets/url_unique/splits/test_attack_1000.csv
  [heloc]=datasets/heloc/splits/test_attack_1000.csv
  [credit-approval]=datasets/credit-approval/splits/test.csv
  [students_dropout]=datasets/students_dropout/splits/test_attack_500.csv
)
declare -A MAX_V=(  # the grid skips larger V: the context must stay at least as large as V
  [credit-approval]=200
)

PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)
HPCT=$(awk -v f="$HOLDOUT" 'BEGIN { printf "%d", f * 100 + 0.5 }')

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

if [[ $WANDB_MODE == online && -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env (git-ignored) or export WANDB_API_KEY"
  exit 1
fi

run_one () {   # run_one <gpu> <dataset> <R> <V> <seed> <full|holdout>
  local gpu=$1 ds=$2 r=$3 v=$4 s=$5 strategy=$6
  local hsuf="" hold=()
  if [[ $strategy == holdout ]]; then hsuf="_H$HPCT"; hold=(--val-holdout "$HOLDOUT"); fi
  local name="gate-valsplit_${ds}_${ATTACK}_R${r}_V${v}s${s}${hsuf}_pgd-linf_lr${LR}_it${STEPS}"
  local out; out="$OUT_ROOT/$ds/$(printf '%s_r%03d' "$ATTACK" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --test "${TEST[$ds]}"
             --val-from-context "$v" --val-seed "$s" "${hold[@]}"
             --attack "$ATTACK" --row-percent "$r"
             --optimizer pgd --pgd-scale linf --lr "$LR" --steps "$STEPS" --test-every "$TEST_EVERY"
             --run-name "$name" --out "$OUT_ROOT"
             --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-mode "$WANDB_MODE")
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

  if [[ -f "$out/test_metrics.json" ]]; then echo "[skip] $name (already done)"; return 0; fi
  mkdir -p "$out"
  if ! mkdir "$out/.lock" 2>/dev/null; then
    echo "[busy] $name (another process holds $out/.lock; delete it if that process died)"
    return 0
  fi
  echo "[$(date +'%F %T')] START GPU $gpu  $name"
  "${cmd[@]}" > "$out/stdout.log" 2>&1
  local rc=$?
  rmdir "$out/.lock" 2>/dev/null
  if [[ $rc -eq 0 ]]; then echo "[$(date +'%F %T')] DONE  GPU $gpu  $name"
  else echo "[$(date +'%F %T')] FAIL  GPU $gpu  $name (rc=$rc, see $out/stdout.log)"; fi
  return 0
}

# every run, as "<dataset> <R> <V> <seed> <strategy>"
JOBS=()
if [[ -n "$JOBS_FILE" ]]; then
  [[ -f $JOBS_FILE ]] || { echo "JOBS_FILE not found: $JOBS_FILE"; exit 1; }
  while read -r part ds r v s strategy rest; do
    [[ -z ${part:-} || $part == \#* ]] && continue
    [[ -z ${strategy:-} || -n ${rest:-} ]] && { echo "bad line in $JOBS_FILE: $part $ds $r $v $s $strategy $rest"; exit 1; }
    [[ -z "$PART" || $part == "$PART" ]] && JOBS+=("$ds $r $v $s $strategy")
  done < "$JOBS_FILE"
  echo "${JOBS_FILE}${PART:+, part $PART}: ${#JOBS[@]} runs"
else
  for ds in $DATASETS; do
    for r in $ROW_PERCENTS; do
      for v in $VAL_SIZES; do
        (( v <= ${MAX_V[$ds]:-1000000000} )) || continue
        for s in $VAL_SEEDS; do
          for strategy in $STRATEGIES; do JOBS+=("$ds $r $v $s $strategy"); done
        done
      done
    done
  done
  if [[ -n "$PART" ]]; then  # this share of the grid, dealt in snake order so the shares balance
    (( PART >= 1 && PART <= NPARTS )) || { echo "PART must be in 1..NPARTS ($NPARTS), got $PART"; exit 1; }
    MINE=()
    for i in "${!JOBS[@]}"; do
      q=$(( i % (2 * NPARTS) ))
      p=$(( q < NPARTS ? q + 1 : 2 * NPARTS - q ))
      (( p == PART )) && MINE+=("${JOBS[$i]}")
    done
    echo "part $PART of $NPARTS: ${#MINE[@]} of ${#JOBS[@]} runs"
    JOBS=("${MINE[@]}")
  fi
fi

for job in "${JOBS[@]}"; do  # check every run before starting any
  read -r ds r v s strategy <<< "$job"
  [[ -n "${CONTEXT[$ds]:-}" ]] || { echo "unknown dataset $ds (known: ${!CONTEXT[*]})"; exit 1; }
  [[ $strategy == full || $strategy == holdout ]] || { echo "unknown strategy $strategy (full or holdout)"; exit 1; }
done

run_queue () {   # run_queue <queue index> <gpu>: every NQ-th run, starting at the index
  local q=$1 gpu=$2 i
  for i in "${!JOBS[@]}"; do
    if (( i % NQ == q )); then
      # shellcheck disable=SC2086  # the job string is split into its five fields on purpose
      run_one "$gpu" ${JOBS[$i]}
    fi
  done
}

GPU_LIST=($GPUS)
NQ=${#GPU_LIST[@]}
echo "${#JOBS[@]} runs on GPU(s) ${GPU_LIST[*]}; results in $OUT_ROOT; W&B $WANDB_MODE, project $WANDB_PROJECT, group $WANDB_GROUP"
for q in "${!GPU_LIST[@]}"; do
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "# queue $q: GPU ${GPU_LIST[$q]}"
    run_queue "$q" "${GPU_LIST[$q]}"
  else
    run_queue "$q" "${GPU_LIST[$q]}" &
  fi
done
wait
echo "ALL REQUESTED RUNS FINISHED"

#!/usr/bin/env bash
# =============================================================================
#  Context gating vs simply using the clean rows (scripts/gate_context.py)
#
#  Is gating worth it? If the small clean validation set, used directly as the
#  ICL context, recovers as much accuracy as gating the poisoned context does,
#  gating is pointless. Each run therefore:
#
#    1. draws a class-stratified random validation set of V rows from the
#       CLEAN 5000-row context (--val-from-context V --val-seed S);
#    2. poisons the other 5000 - V rows (hub-flip-local, R%);
#    3. learns one gate per poisoned-context row on the V clean rows;
#    4. scores the test set (1000 rows) every TEST_EVERY updates, soft (current
#       gates) and hard (the R% lowest-gated rows discarded), to see where the
#       gates start over-fitting the V rows;
#    5. scores, once, the references, among them
#         val_as_context     the V clean rows alone as the context  <- the baseline
#         poisoned_plus_val  the poisoned rows + the V clean rows
#         clean_context      the 5000 - V rows with their clean labels
#         poisoned_no_gate, random removal, oracle gate, oracle removal
#       val_as_context is drawn as a flat line on every test curve (plots and
#       W&B overlay charts), and test_metrics.json has vs_val_as_context =
#       (best soft gates) - (val_as_context) per metric.
#
#  Grid: 2 datasets x R in {10, 20}% x V in {100, ..., 500} x val seeds
#        = 20 runs per seed. One seed gives nested draws (V=200 contains V=100),
#        so the V ablation compares sets that grow, not unrelated sets.
#  Optimiser: the best setting of the learning-rate sweep (run_gate_sweep.sh):
#        PGD, L-inf step scaling, lr 0.05, 300 updates.
#
#    gas-drift    6 classes   context_5000.csv, test_attack_1000.csv
#    url_unique   binary      selected/natural/context_5000.csv, test_attack_1000.csv
#
#  Each run is logged to W&B project TFM_experiments, group gate-valsplit, as
#      gate-valsplit_<dataset>_<attack>_R<R>_V<V>s<seed>_pgd-linf_lr<lr>_it<updates>
#  and writes its files to
#      results/gate_valsplit/<dataset>/<attack>_r<RRR>/<that name>/
#  (split.npz there holds the validation and context row indices into the
#  context file).
#
#  GPUs: one queue per GPU in GPUS (default "1 3"); each runs its share of the
#  runs one after another. A run needs ~11 GiB of GPU memory.
#
#  Re-running resumes: a run whose test_metrics.json exists is skipped. A run
#  being worked on holds <run dir>/.lock; delete it if that process died.
#
#  W&B key (online mode only): WANDB_API_KEY if exported, else the line
#  WANDB_API_KEY=<key> in .env in the repo root, which scripts/load_env.sh loads
#  (git-ignored, owner-only; never commit it, copy it to each machine). .env may
#  also set any of the settings below; an exported variable still wins.
#
#  Splitting the grid across machines/GPUs: PART=k NPARTS=m runs only the k-th of m
#  shares (k = 1..m). Runs are dealt to the shares in snake order (1..m, m..1, 1..m, ...)
#  over the job list, which starts with the slow dataset (gas-drift), so every share gets
#  a similar total time. scripts/gate_valsplit_parts/run_part<k>.sh are 10 shares
#  as one script per GPU.
#
#    bash scripts/run_gate_valsplit_sweep.sh                    # everything, GPUs 1 and 3
#    DRY_RUN=1 bash scripts/run_gate_valsplit_sweep.sh          # print the commands only
#    VAL_SEEDS="0 1 2" bash scripts/run_gate_valsplit_sweep.sh  # three draws per size
#    GPUS="1" DATASETS="url_unique" VAL_SIZES="100 200" bash scripts/run_gate_valsplit_sweep.sh
#    PART=3 NPARTS=7 GPUS="2" bash scripts/run_gate_valsplit_sweep.sh   # share 3 of 7 on GPU 2
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
DATASETS=${DATASETS:-"gas-drift url_unique"}
ROW_PERCENTS=${ROW_PERCENTS:-"10 20"}
VAL_SIZES=${VAL_SIZES:-"100 200 300 400 500"}
VAL_SEEDS=${VAL_SEEDS:-"0"}
ATTACK=hub-flip-local
LR=${LR:-0.05}
STEPS=${STEPS:-300}
TEST_EVERY=${TEST_EVERY:-5}
WANDB_PROJECT=${WANDB_PROJECT:-TFM_experiments}
WANDB_GROUP=${WANDB_GROUP:-gate-valsplit}
WANDB_MODE=${WANDB_MODE:-online}
PART=${PART:-}          # with NPARTS: run only the PART-th share (1..NPARTS)
NPARTS=${NPARTS:-1}

declare -A CONTEXT=(
  [gas-drift]=datasets/gas-drift/splits/context_5000.csv
  [url_unique]=datasets/url_unique/splits/selected/natural/context_5000.csv
)
declare -A TEST=(
  [gas-drift]=datasets/gas-drift/splits/test_attack_1000.csv
  [url_unique]=datasets/url_unique/splits/test_attack_1000.csv
)

PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

for ds in $DATASETS; do
  [[ -n "${CONTEXT[$ds]:-}" ]] || { echo "unknown dataset $ds (known: ${!CONTEXT[*]})"; exit 1; }
done

if [[ $WANDB_MODE == online && -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env (git-ignored) or export WANDB_API_KEY"
  exit 1
fi

run_one () {   # run_one <gpu> <dataset> <R> <V> <seed>
  local gpu=$1 ds=$2 r=$3 v=$4 s=$5
  local name="gate-valsplit_${ds}_${ATTACK}_R${r}_V${v}s${s}_pgd-linf_lr${LR}_it${STEPS}"
  local out; out="$OUT_ROOT/$ds/$(printf '%s_r%03d' "$ATTACK" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --test "${TEST[$ds]}"
             --val-from-context "$v" --val-seed "$s"
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

# every run of the sweep, as "<dataset> <R> <V> <seed>"
JOBS=()
for ds in $DATASETS; do
  for r in $ROW_PERCENTS; do
    for v in $VAL_SIZES; do
      for s in $VAL_SEEDS; do JOBS+=("$ds $r $v $s"); done
    done
  done
done

# PART/NPARTS: keep only this share, dealt in snake order so the shares balance
if [[ -n "$PART" ]]; then
  (( PART >= 1 && PART <= NPARTS )) || { echo "PART must be in 1..NPARTS ($NPARTS), got $PART"; exit 1; }
  MINE=()
  for i in "${!JOBS[@]}"; do
    r=$(( i % (2 * NPARTS) ))
    p=$(( r < NPARTS ? r + 1 : 2 * NPARTS - r ))
    (( p == PART )) && MINE+=("${JOBS[$i]}")
  done
  echo "part $PART of $NPARTS: ${#MINE[@]} of ${#JOBS[@]} runs"
  JOBS=("${MINE[@]}")
fi

run_queue () {   # run_queue <queue index> <gpu>: every NQ-th run, starting at the index
  local q=$1 gpu=$2 i
  for i in "${!JOBS[@]}"; do
    if (( i % NQ == q )); then
      # shellcheck disable=SC2086  # the job string is split into its four fields on purpose
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

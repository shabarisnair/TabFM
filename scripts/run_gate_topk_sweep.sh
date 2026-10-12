#!/usr/bin/env bash
# =============================================================================
#  Top-k gating ablation (scripts/gate_context.py --gate topk)
#
#  Exactly R% of the context rows are switched off: each row has a free score S,
#  the forward pass masks the R% lowest scores (gate 0, the rest 1), and the gate
#  gradient reaches the scores through a straight-through estimator. Ablated:
#
#    STES        surrogate in the backward pass: identity (no sigmoid) | sigmoid
#                (sigmoid-centered, centred on the keep/remove boundary, is also
#                implemented: STES="identity sigmoid sigmoid-centered")
#    INIT_STDS   initial scores N(0, std^2): 1 | 0.01
#    LRS         PGD step with L-inf scaling: the largest score change per update
#    INIT_SEEDS  4 random initialisations (restarts) per setting. The best restart
#                is the one with the lowest validation-test loss (CE) at its
#                chosen iteration, picked afterwards from the runs' logs.
#
#  Every run: V = 500 clean rows drawn from the clean context (validation seed 0),
#  100 of them held out as validation-test (--val-holdout 0.2) and the other 400
#  fitted; the other 4500 context rows poisoned (hub-flip-local, R%); 300 updates;
#  the validation-test and test sets scored every TEST_EVERY = 5 updates; the
#  best iteration is the one with the lowest validation-test CE (--select-on ce).
#  Validation seed 0 with V = 500 and a 20% holdout is the same draw, holdout and
#  poison as the soft-gate runs of results/gate_valsplit, so the two compare
#  directly.
#
#  Grid: gas-drift, url_unique, students_dropout x R in {10, 20}% x STES x
#  INIT_STDS x LRS x INIT_SEEDS = 3 x 2 x 2 x 2 x 3 x 4 = 288 runs.
#
#  Each run is logged to W&B project TFM_experiments, group gate-topk, as
#      gate-topk_<dataset>_<attack>_R<R>_V500s0_H20_<ste>_std<std>_lr<lr>_init<seed>_it<updates>
#  and writes its files to results/gate_topk/<dataset>/<attack>_r<RRR>/<that name>/.
#
#  JOBS_FILE: run an explicit list instead of the grid, one run per line,
#      <part> <dataset> <R> <ste> <init_std> <lr> <init_seed>
#  (# comments and blank lines skipped); with PART=k only the lines of part k.
#  scripts/gate_topk_parts/<round>/run_part<k>.sh run those parts, one per GPU
#  (written by scripts/gate_topk_parts/make_jobs.py).
#
#  Re-running resumes: a run whose test_metrics.json exists is skipped; a run in
#  progress holds <run dir>/.lock (delete it if that process died).
#  W&B key: WANDB_API_KEY if exported, else WANDB_API_KEY=<key> in .env in the
#  repo root (scripts/load_env.sh; git-ignored, never commit it).
#
#    bash scripts/run_gate_topk_sweep.sh                    # the whole grid, GPUs 1 and 3
#    DRY_RUN=1 bash scripts/run_gate_topk_sweep.sh          # print the commands only
#    GPUS="0" DATASETS="students_dropout" LRS="0.2" bash scripts/run_gate_topk_sweep.sh
#
#  Every setting below can be overridden the same way (e.g. STEPS, WANDB_MODE=offline,
#  OUT_ROOT, CONDA_ENV). REPO defaults to the checkout this script sits in.
# =============================================================================

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" "$REPO/.env"   # WANDB_API_KEY, and any setting below
CONDA_ENV=${CONDA_ENV:-tabfm}
OUT_ROOT=${OUT_ROOT:-$REPO/results/gate_topk}
GPUS=${GPUS:-"1 3"}
DATASETS=${DATASETS:-"gas-drift url_unique students_dropout"}
ROW_PERCENTS=${ROW_PERCENTS:-"10 20"}
STES=${STES:-"identity sigmoid"}
INIT_STDS=${INIT_STDS:-"1 0.01"}
LRS=${LRS:-"0.05 0.2 1"}
INIT_SEEDS=${INIT_SEEDS:-"0 1 2 3"}
VAL_SIZE=${VAL_SIZE:-500}
VAL_SEED=${VAL_SEED:-0}
HOLDOUT=${HOLDOUT:-0.2}
ATTACK=hub-flip-local
STEPS=${STEPS:-300}
TEST_EVERY=${TEST_EVERY:-5}
WANDB_PROJECT=${WANDB_PROJECT:-TFM_experiments}
WANDB_GROUP=${WANDB_GROUP:-gate-topk}
WANDB_MODE=${WANDB_MODE:-online}
JOBS_FILE=${JOBS_FILE:-}  # explicit run list (see above); otherwise the grid
PART=${PART:-}            # with JOBS_FILE: only that part's lines

declare -A CONTEXT=(
  [gas-drift]=datasets/gas-drift/splits/context_5000.csv
  [url_unique]=datasets/url_unique/splits/selected/natural/context_5000.csv
  [heloc]=datasets/heloc/splits/selected/natural/context_5000.csv
  [students_dropout]=datasets/students_dropout/splits/context_3000.csv
)
declare -A TEST=(
  [gas-drift]=datasets/gas-drift/splits/test_attack_1000.csv
  [url_unique]=datasets/url_unique/splits/test_attack_1000.csv
  [heloc]=datasets/heloc/splits/test_attack_1000.csv
  [students_dropout]=datasets/students_dropout/splits/test_attack_500.csv
)

PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)
HPCT=$(awk -v f="$HOLDOUT" 'BEGIN { printf "%d", f * 100 + 0.5 }')

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

if [[ $WANDB_MODE == online && -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env (git-ignored) or export WANDB_API_KEY"
  exit 1
fi

run_one () {   # run_one <gpu> <dataset> <R> <ste> <init_std> <lr> <init_seed>
  local gpu=$1 ds=$2 r=$3 ste=$4 std=$5 lr=$6 seed=$7
  local name="gate-topk_${ds}_${ATTACK}_R${r}_V${VAL_SIZE}s${VAL_SEED}_H${HPCT}_${ste}_std${std}_lr${lr}_init${seed}_it${STEPS}"
  local out; out="$OUT_ROOT/$ds/$(printf '%s_r%03d' "$ATTACK" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --test "${TEST[$ds]}"
             --val-from-context "$VAL_SIZE" --val-seed "$VAL_SEED" --val-holdout "$HOLDOUT"
             --attack "$ATTACK" --row-percent "$r"
             --gate topk --ste "$ste" --score-init-std "$std" --init-seed "$seed"
             --optimizer pgd --pgd-scale linf --lr "$lr" --steps "$STEPS" --test-every "$TEST_EVERY"
             --select-on ce
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

# every run, as "<dataset> <R> <ste> <init_std> <lr> <init_seed>"
JOBS=()
if [[ -n "$JOBS_FILE" ]]; then
  [[ -f $JOBS_FILE ]] || { echo "JOBS_FILE not found: $JOBS_FILE"; exit 1; }
  while read -r part ds r ste std lr seed rest; do
    [[ -z ${part:-} || $part == \#* ]] && continue
    [[ -z ${seed:-} || -n ${rest:-} ]] && { echo "bad line in $JOBS_FILE: $part $ds $r $ste $std $lr $seed $rest"; exit 1; }
    [[ -z "$PART" || $part == "$PART" ]] && JOBS+=("$ds $r $ste $std $lr $seed")
  done < "$JOBS_FILE"
  echo "${JOBS_FILE}${PART:+, part $PART}: ${#JOBS[@]} runs"
else
  for ds in $DATASETS; do
    for r in $ROW_PERCENTS; do
      for ste in $STES; do
        for std in $INIT_STDS; do
          for lr in $LRS; do
            for seed in $INIT_SEEDS; do JOBS+=("$ds $r $ste $std $lr $seed"); done
          done
        done
      done
    done
  done
fi

for job in "${JOBS[@]}"; do  # check every run before starting any
  read -r ds r ste std lr seed <<< "$job"
  [[ -n "${CONTEXT[$ds]:-}" ]] || { echo "unknown dataset $ds (known: ${!CONTEXT[*]})"; exit 1; }
  [[ $ste == identity || $ste == sigmoid || $ste == sigmoid-centered ]] || { echo "unknown surrogate $ste"; exit 1; }
done

run_queue () {   # run_queue <queue index> <gpu>: every NQ-th run, starting at the index
  local q=$1 gpu=$2 i
  for i in "${!JOBS[@]}"; do
    if (( i % NQ == q )); then
      # shellcheck disable=SC2086  # the job string is split into its six fields on purpose
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

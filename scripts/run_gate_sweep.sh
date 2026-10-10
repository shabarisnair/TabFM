#!/usr/bin/env bash
# =============================================================================
#  Context gating (scripts/gate_context.py): learning-rate sweep
#
#  2 datasets x hub-flip-local at R in {10, 20}% x {PGD, Adam} x 3 learning rates
#  = 24 runs, 300 gate updates each.
#
#    gas-drift    6 classes   context_5000.csv
#                             val_1000.csv  (all of it optimised on)
#                             test_attack_1000.csv
#    url_unique   binary      selected/natural/context_5000.csv
#                             val_2000.csv  (all of it optimised on)
#                             test_attack_1000.csv
#    (the same contexts and test sets as the earlier experiments)
#
#    PGD   --pgd-scale linf; lr = the largest change of any gate per update
#          (PGD_SCALE=none: plain projected gradient descent, step = lr * grad)
#    Adam  on the gates directly, clipped to [0, 1] after each update
#
#  Every gate starts at 1. The test set is scored every 5 updates two ways:
#    soft   with the current gates
#    hard   with the R% lowest-gated rows discarded and the rest run as usual
#  Nothing enforces an R% budget while the gates are learned.
#
#  Each run is logged to W&B project TFM_experiments under its own name,
#      gate_<dataset>_<attack>_R<R>_<optimizer>_lr<lr>_it<updates>
#  and writes its files to results/gate/<dataset>/<attack>_r<RRR>/<that name>/.
#
#  GPUs: one queue per GPU in GPUS (default "1 3"); each queue runs its share of the
#  runs one after another. Each run needs ~11 GiB of GPU memory.
#  Time: ~7.5 s per update on gas-drift (measured), so ~40 min per run. url_unique
#  should be faster but was not timed. The whole sweep is roughly 6-7 hours on two GPUs.
#
#  Re-running resumes: a run whose test_metrics.json exists is skipped. A run being
#  worked on holds <run dir>/.lock; delete it if that process died.
#
#  W&B key: WANDB_API_KEY if exported, else the line WANDB_API_KEY=<key> in .env in
#  the repo root, which scripts/load_env.sh loads (git-ignored, owner-only; never commit
#  it, copy it to each machine). .env may also set any of the settings below; an
#  exported variable still wins. REPO defaults to the checkout this script sits in.
#
#    bash scripts/run_gate_sweep.sh                    # everything, GPUs 1 and 3
#    DRY_RUN=1 bash scripts/run_gate_sweep.sh          # print the commands only
#    GPUS="3" DATASETS="gas-drift" bash scripts/run_gate_sweep.sh
# =============================================================================

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" "$REPO/.env"   # WANDB_API_KEY, and any setting below
CONDA_ENV=${CONDA_ENV:-tabfm}
OUT_ROOT=${OUT_ROOT:-$REPO/results/gate}
GPUS=${GPUS:-"1 3"}
DATASETS=${DATASETS:-"gas-drift url_unique"}
ROW_PERCENTS=${ROW_PERCENTS:-"10 20"}
OPTIMIZERS=${OPTIMIZERS:-"pgd adam"}
PGD_LRS=${PGD_LRS:-"0.02 0.05 0.1"}
ADAM_LRS=${ADAM_LRS:-"0.003 0.01 0.03"}
PGD_SCALE=${PGD_SCALE:-linf}
ATTACK=hub-flip-local
STEPS=300
TEST_EVERY=5
WANDB_PROJECT=TFM_experiments

declare -A CONTEXT=(
  [gas-drift]=datasets/gas-drift/splits/context_5000.csv
  [url_unique]=datasets/url_unique/splits/selected/natural/context_5000.csv
)
declare -A VAL=(
  [gas-drift]=datasets/gas-drift/splits/val_1000.csv
  [url_unique]=datasets/url_unique/splits/val_2000.csv
)
declare -A TEST=(
  [gas-drift]=datasets/gas-drift/splits/test_attack_1000.csv
  [url_unique]=datasets/url_unique/splits/test_attack_1000.csv
)

PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

if [[ -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env (git-ignored) or export WANDB_API_KEY"
  exit 1
fi

run_one () {   # run_one <gpu> <dataset> <R> <optimizer> <lr>
  local gpu=$1 ds=$2 r=$3 opt=$4 lr=$5
  local optname=$opt
  [[ $opt == pgd ]] && optname=pgd-$PGD_SCALE
  local name="gate_${ds}_${ATTACK}_R${r}_${optname}_lr${lr}_it${STEPS}"
  local out; out="$OUT_ROOT/$ds/$(printf '%s_r%03d' "$ATTACK" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --val "${VAL[$ds]}" --test "${TEST[$ds]}"
             --attack "$ATTACK" --row-percent "$r"
             --optimizer "$opt" --lr "$lr" --steps "$STEPS" --test-every "$TEST_EVERY"
             --run-name "$name" --wandb-project "$WANDB_PROJECT" --out "$OUT_ROOT")
  [[ $opt == pgd ]] && cmd+=(--pgd-scale "$PGD_SCALE")
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

# every run of the sweep, as "<dataset> <R> <optimizer> <lr>"
JOBS=()
for ds in $DATASETS; do
  for r in $ROW_PERCENTS; do
    for opt in $OPTIMIZERS; do
      lrs=$PGD_LRS
      [[ $opt == adam ]] && lrs=$ADAM_LRS
      for lr in $lrs; do JOBS+=("$ds $r $opt $lr"); done
    done
  done
done

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
echo "${#JOBS[@]} runs on GPU(s) ${GPU_LIST[*]}; results in $OUT_ROOT; W&B project $WANDB_PROJECT"
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

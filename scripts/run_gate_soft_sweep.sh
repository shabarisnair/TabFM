#!/usr/bin/env bash
# =============================================================================
#  Soft-gate variants (scripts/gate_context.py --gate soft | attn), one run per line of
#  JOBS_FILE:
#      <part> <dataset> <R> <V> <seed> <tag> [<extra args to gate_context.py>...]
#  (# comments and blank lines skipped; with PART=k only the lines of part k). Every run
#  draws V clean rows from the context (validation seed <seed>), holds out 20% of them
#  (--val-holdout 0.2), poisons the rest (ATTACK, default hub-flip-local, R%), and runs 300 updates,
#  scoring the held-out rows and the test set every 5; the extra args (e.g. "--gate attn
#  --lr 0.5") override the defaults below (argparse keeps the last value). The tag names
#  the variant:
#      gate-soft_<dataset>_<attack>_R<R>_V<V>s<seed>_H20_<tag>_it300
#  under results/gate_soft/<dataset>/<attack>_r<RRR>/, logged to W&B project
#  TFM_experiments, group gate-soft-night. The same draws as results/gate_valsplit.
#
#  Several workers (one per GPU) can share a JOBS_FILE: a run in progress holds
#  <run dir>/.lock and a finished run (test_metrics.json) is skipped.
#
#    JOBS_FILE=scripts/gate_soft_parts/r2.txt GPUS="1 2" bash scripts/run_gate_soft_sweep.sh
#    DRY_RUN=1 JOBS_FILE=... bash scripts/run_gate_soft_sweep.sh
# =============================================================================

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" "$REPO/.env"   # WANDB_API_KEY, and any setting below
CONDA_ENV=${CONDA_ENV:-tabfm}
OUT_ROOT=${OUT_ROOT:-$REPO/results/gate_soft}
GPUS=${GPUS:-"1"}
HOLDOUT=${HOLDOUT:-0.2}
ATTACK=${ATTACK:-hub-flip-local}  # the label attack of every run of this worker
STEPS=${STEPS:-300}
TEST_EVERY=${TEST_EVERY:-5}
WANDB_PROJECT=${WANDB_PROJECT:-TFM_experiments}
WANDB_GROUP=${WANDB_GROUP:-gate-soft-night}
WANDB_MODE=${WANDB_MODE:-online}
JOBS_FILE=${JOBS_FILE:?set JOBS_FILE (one run per line, see the header)}
PART=${PART:-}

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

PY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)
HPCT=$(awk -v f="$HOLDOUT" 'BEGIN { printf "%d", f * 100 + 0.5 }')

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }
if [[ $WANDB_MODE == online && -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env or export WANDB_API_KEY"; exit 1
fi

run_one () {   # run_one <gpu> <dataset> <R> <V> <seed> <tag> [extra args...]
  local gpu=$1 ds=$2 r=$3 v=$4 s=$5 tag=$6
  local extra=("${@:7}")
  local name="gate-soft_${ds}_${ATTACK}_R${r}_V${v}s${s}_H${HPCT}_${tag}_it${STEPS}"
  local out; out="$OUT_ROOT/$ds/$(printf '%s_r%03d' "$ATTACK" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --test "${TEST[$ds]}"
             --val-from-context "$v" --val-seed "$s" --val-holdout "$HOLDOUT"
             --attack "$ATTACK" --row-percent "$r"
             --optimizer pgd --pgd-scale linf --lr 0.05 --steps "$STEPS" --test-every "$TEST_EVERY"
             "${extra[@]}"
             --run-name "$name" --out "$OUT_ROOT"
             --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-mode "$WANDB_MODE")
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  if [[ -f "$out/test_metrics.json" ]]; then echo "[skip] $name (already done)"; return 0; fi
  mkdir -p "$out"
  if ! mkdir "$out/.lock" 2>/dev/null; then echo "[busy] $name"; return 0; fi
  echo "[$(date +'%F %T')] START GPU $gpu  $name"
  "${cmd[@]}" > "$out/stdout.log" 2>&1
  local rc=$?
  rmdir "$out/.lock" 2>/dev/null
  if [[ $rc -eq 0 ]]; then echo "[$(date +'%F %T')] DONE  GPU $gpu  $name"
  else echo "[$(date +'%F %T')] FAIL  GPU $gpu  $name (rc=$rc, see $out/stdout.log)"; fi
  return 0
}

JOBS=()
[[ -f $JOBS_FILE ]] || { echo "JOBS_FILE not found: $JOBS_FILE"; exit 1; }
while read -r part ds r v s tag rest; do
  [[ -z ${part:-} || $part == \#* ]] && continue
  [[ -z ${tag:-} ]] && { echo "bad line in $JOBS_FILE: $part $ds $r $v $s"; exit 1; }
  [[ -n "${CONTEXT[$ds]:-}" ]] || { echo "unknown dataset $ds"; exit 1; }
  [[ -z "$PART" || $part == "$PART" ]] && JOBS+=("$ds $r $v $s $tag${rest:+ $rest}")
done < "$JOBS_FILE"

run_queue () {   # every run, claimed through its .lock: workers on several GPUs share the list
  local gpu=$1 i
  for i in "${!JOBS[@]}"; do
    # shellcheck disable=SC2086  # the job string is split into its fields on purpose
    run_one "$gpu" ${JOBS[$i]}
  done
}

GPU_LIST=($GPUS)
echo "${JOBS_FILE}${PART:+, part $PART}: ${#JOBS[@]} runs on GPU(s) ${GPU_LIST[*]}; results in $OUT_ROOT; W&B $WANDB_MODE, group $WANDB_GROUP"
for gpu in "${GPU_LIST[@]}"; do
  if [[ -n "${DRY_RUN:-}" ]]; then run_queue "$gpu"; break; fi
  run_queue "$gpu" &
  sleep 2
done
wait
echo "ALL REQUESTED RUNS FINISHED"

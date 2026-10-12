#!/usr/bin/env bash
# =============================================================================
#  Does the best top-k configuration generalise? (scripts/gate_context.py --gate topk)
#
#  The configuration found on gas-drift and url_unique at R = 20%, V = 500
#  (results/tables/gate_topk/night_log.md), unchanged:
#    hard top-k gate: exactly R% of the context rows switched off, gates 0/1;
#    identity straight-through, scores N(0, 1), L-inf score step with lr 5 (constant),
#    300 updates; the straight-through gradient taken at the relaxed point alpha = 0.05
#    (--ste-probe 0.05: kept rows at 0.95, masked rows at 0.05 in the backward pass
#    only); each restart's iteration chosen by the lowest held-out CE (--select-on ce);
#    4 restarts (INIT_SEEDS) combined by a vote (scripts/vote_topk.py): the k rows
#    masked by the most restarts are switched off.
#
#  Run on the grid of the soft-gate experiment (run_gate_valsplit_sweep.sh, holdout
#  strategy): DATASETS x V in VAL_SIZES (up to MAX_V) x VAL_SEEDS (default: seed 0 only),
#  at R in ROW_PERCENTS (default 20; "10 20" for both). Each draw (dataset, R, V, validation seed; 20% of the V rows held out)
#  is the same draw, holdout and poison as the soft-gate run
#  results/gate_valsplit/<dataset>/<attack>_r<RRR>/gate-valsplit_..._R<R>_V<V>s<seed>_H20_...,
#  so the two compare directly.
#
#  One job is one draw: its restarts one after another, then their vote, on one GPU.
#  Runs are named
#      gate-topk_<dataset>_<attack>_R<R>_V<V>s<seed>_H20_identity_std1_lr5_init<i>_probe0.05_it300
#  the vote
#      gate-topk-vote4_<dataset>_<attack>_R<R>_V<V>s<seed>_H20_identity_std1_lr5_probe0.05_it300
#  both under results/gate_topk_gen/<dataset>/<attack>_r<RRR>/, and both logged to W&B
#  project TFM_experiments, group gate-topk-gen (the vote as a run of job type "vote").
#  The vote writes vote.json (vote, best restart, every restart, references) and
#  vote.npz (mask, votes, test probabilities).
#
#  JOBS_FILE: run an explicit list of draws instead of the grid, one per line,
#      <part> <dataset> <R> <V> <seed>
#  (# comments and blank lines skipped); with PART=k only the lines of part k.
#  scripts/gate_topk_parts/<name>/run_part<k>.sh run those parts, one per GPU
#  (written by scripts/gate_topk_parts/make_gen_jobs.py).
#
#  Re-running resumes: a run whose test_metrics.json exists is skipped, and so is a
#  vote whose vote.json exists; a run or vote in progress holds <its dir>/.lock
#  (delete it if that process died). A draw with a failed restart gets no vote.
#  W&B key: WANDB_API_KEY if exported, else WANDB_API_KEY=<key> in .env in the repo
#  root (scripts/load_env.sh; git-ignored, never commit it).
#
#    bash scripts/run_gate_topk_gen_sweep.sh                  # the whole grid, GPUs 1 and 3
#    DRY_RUN=1 bash scripts/run_gate_topk_gen_sweep.sh        # print the commands only
#    GPUS="0" DATASETS="heloc" VAL_SIZES="50 100" bash scripts/run_gate_topk_gen_sweep.sh
#
#  Every setting below can be overridden the same way (e.g. WANDB_MODE=offline,
#  OUT_ROOT, CONDA_ENV, VOTE=0). REPO defaults to the checkout this script sits in.
# =============================================================================

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" "$REPO/.env"   # WANDB_API_KEY, and any setting below
CONDA_ENV=${CONDA_ENV:-tabfm}
OUT_ROOT=${OUT_ROOT:-$REPO/results/gate_topk_gen}
GPUS=${GPUS:-"1 3"}
DATASETS=${DATASETS:-"gas-drift url_unique heloc credit-approval students_dropout"}
ROW_PERCENTS=${ROW_PERCENTS:-"20"}
VAL_SIZES=${VAL_SIZES:-"50 100 200 300 400 500"}
VAL_SEEDS=${VAL_SEEDS:-"0"}
INIT_SEEDS=${INIT_SEEDS:-"0 1 2 3"}
HOLDOUT=${HOLDOUT:-0.2}
ATTACK=hub-flip-local
STE=${STE:-identity}
INIT_STD=${INIT_STD:-1}
LR=${LR:-5}
PROBE=${PROBE:-0.05}
STEPS=${STEPS:-300}
TEST_EVERY=${TEST_EVERY:-5}
VOTE=${VOTE:-1}           # 1: vote each draw's restarts once they are all done
WANDB_PROJECT=${WANDB_PROJECT:-TFM_experiments}
WANDB_GROUP=${WANDB_GROUP:-gate-topk-gen}
WANDB_MODE=${WANDB_MODE:-online}
JOBS_FILE=${JOBS_FILE:-}  # explicit list of draws (see above); otherwise the grid
PART=${PART:-}            # with JOBS_FILE: only that part's lines

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
TAG=$(awk -v p="$PROBE" 'BEGIN { if (p + 0 > 0) print "probe" p }')
NRESTARTS=$(wc -w <<< "$INIT_SEEDS")

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

if [[ $WANDB_MODE == online && -z "${WANDB_API_KEY:-}" && -z "${DRY_RUN:-}" ]]; then
  echo "no W&B key: put WANDB_API_KEY=<key> in $REPO/.env (git-ignored) or export WANDB_API_KEY"
  exit 1
fi

draw_dir () {   # draw_dir <dataset> <R>: the folder of that dataset and R
  printf '%s/%s/%s_r%03d' "$OUT_ROOT" "$1" "$ATTACK" "$2"
}
run_name () {   # run_name <dataset> <R> <V> <seed> <init_seed>
  echo "gate-topk_${1}_${ATTACK}_R${2}_V${3}s${4}_H${HPCT}_${STE}_std${INIT_STD}_lr${LR}_init${5}${TAG:+_$TAG}_it${STEPS}"
}
vote_name () {  # vote_name <dataset> <R> <V> <seed>
  echo "gate-topk-vote${NRESTARTS}_${1}_${ATTACK}_R${2}_V${3}s${4}_H${HPCT}_${STE}_std${INIT_STD}_lr${LR}${TAG:+_$TAG}_it${STEPS}"
}

run_one () {   # run_one <gpu> <dataset> <R> <V> <seed> <init_seed>
  local gpu=$1 ds=$2 r=$3 v=$4 s=$5 seed=$6
  local name; name=$(run_name "$ds" "$r" "$v" "$s" "$seed")
  local out; out="$(draw_dir "$ds" "$r")/$name"
  local cmd=("${PY[@]}" "$REPO/scripts/gate_context.py" --gpu "$gpu"
             --dataset "$ds" --context "${CONTEXT[$ds]}" --test "${TEST[$ds]}"
             --val-from-context "$v" --val-seed "$s" --val-holdout "$HOLDOUT"
             --attack "$ATTACK" --row-percent "$r"
             --gate topk --ste "$STE" --score-init-std "$INIT_STD" --init-seed "$seed" --ste-probe "$PROBE"
             --optimizer pgd --pgd-scale linf --lr "$LR" --steps "$STEPS" --test-every "$TEST_EVERY"
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

vote_one () {   # vote_one <gpu> <dataset> <R> <V> <seed>: vote the draw's restarts if all are done
  local gpu=$1 ds=$2 r=$3 v=$4 s=$5 seed
  local name; name=$(vote_name "$ds" "$r" "$v" "$s")
  local out; out="$(draw_dir "$ds" "$r")/$name"
  local runs=() missing=0
  for seed in $INIT_SEEDS; do
    runs+=("$(draw_dir "$ds" "$r")/$(run_name "$ds" "$r" "$v" "$s" "$seed")")
    [[ -f "${runs[-1]}/test_metrics.json" ]] || missing=$((missing + 1))
  done
  local cmd=("${PY[@]}" "$REPO/scripts/vote_topk.py" --gpu "$gpu" --out "$out"
             --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-mode "$WANDB_MODE" "${runs[@]}")
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

  if [[ -f "$out/vote.json" ]]; then echo "[skip] $name (already done)"; return 0; fi
  if (( missing > 0 )); then echo "[no vote] $name: $missing of $NRESTARTS restarts not finished"; return 0; fi
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

# every draw, as "<dataset> <R> <V> <seed>"
DRAWS=()
if [[ -n "$JOBS_FILE" ]]; then
  [[ -f $JOBS_FILE ]] || { echo "JOBS_FILE not found: $JOBS_FILE"; exit 1; }
  while read -r part ds r v s rest; do
    [[ -z ${part:-} || $part == \#* ]] && continue
    [[ -z ${s:-} || -n ${rest:-} ]] && { echo "bad line in $JOBS_FILE: $part $ds $r $v $s $rest"; exit 1; }
    [[ -z "$PART" || $part == "$PART" ]] && DRAWS+=("$ds $r $v $s")
  done < "$JOBS_FILE"
  echo "${JOBS_FILE}${PART:+, part $PART}: ${#DRAWS[@]} draws"
else
  for s in $VAL_SEEDS; do  # seed-major: the first seed's whole grid comes first
    for ds in $DATASETS; do
      for r in $ROW_PERCENTS; do
        for v in $VAL_SIZES; do
          (( v <= ${MAX_V[$ds]:-1000000000} )) || continue
          DRAWS+=("$ds $r $v $s")
        done
      done
    done
  done
fi

for draw in "${DRAWS[@]}"; do  # check every draw before starting any
  read -r ds r v s <<< "$draw"
  [[ -n "${CONTEXT[$ds]:-}" ]] || { echo "unknown dataset $ds (known: ${!CONTEXT[*]})"; exit 1; }
  (( v <= ${MAX_V[$ds]:-1000000000} )) || { echo "V = $v is too large for $ds (at most ${MAX_V[$ds]})"; exit 1; }
done

run_queue () {   # run_queue <queue index> <gpu>: every NQ-th draw, its restarts, then its vote
  local q=$1 gpu=$2 i seed
  for i in "${!DRAWS[@]}"; do
    if (( i % NQ == q )); then
      read -r ds r v s <<< "${DRAWS[$i]}"
      for seed in $INIT_SEEDS; do run_one "$gpu" "$ds" "$r" "$v" "$s" "$seed"; done
      [[ $VOTE == 1 ]] && vote_one "$gpu" "$ds" "$r" "$v" "$s"
    fi
  done
}

GPU_LIST=($GPUS)
NQ=${#GPU_LIST[@]}
echo "${#DRAWS[@]} draws x $NRESTARTS restarts on GPU(s) ${GPU_LIST[*]}; results in $OUT_ROOT;" \
     "W&B $WANDB_MODE, project $WANDB_PROJECT, group $WANDB_GROUP"
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

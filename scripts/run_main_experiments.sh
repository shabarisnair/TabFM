#!/usr/bin/env bash
# =============================================================================
#  Main context-poisoning experiment grid
#
#  4 datasets x { X_train CAPGD , Y_train label-flip (GA) } x R in {1,4,16,64} %
#  = 32 commands.   Rows attacked are sampled at RANDOM (no class targeting).
#
#  Context: datasets/<ds>/splits/selected/natural/context_5000.csv  (unbalanced 5000)
#  Test:    datasets/<ds>/splits/test_attack_1000.csv
#
#  Everything except --gpu/--attack/--row-percent/--train/--test is left at the
#  script defaults (n-iter, constraints full, budgeted repair, n-runs,
#  n-row-subsamples, eps, lambda, GA population/generations ...).
#  => if a default changes later, re-running this file picks it up automatically.
#
#  HOW TO USE
#    1. Edit the CONFIG block below (GPU id is the main one).
#    2. Run everything:            bash run_main_experiments.sh
#       Preview without running:   DRY_RUN=1 bash run_main_experiments.sh
#    3. To split across servers/GPUs: copy the CONFIG + run() block (everything
#       above the "COMMANDS" line) into a new file, then paste only the `run ...`
#       lines you want under it. Each `run` line is independent.
#
#  Re-running is safe: a command whose output already has summary.json is skipped.
#
#  Each command saves ONE poisoned context (SAVE_CONTEXT below): the best trial of its
#  n_runs x n_row_subsamples grid, chosen by largest CE increase.
#
#  NOTE: class-targeted attacks (--attack-class 0/1) and R in {2,8,32,100} were
#  removed on request. To re-add a targeted run, use e.g.
#      run wids x-capgd class1 16
#  (the run() helper still supports class0/class1).
# =============================================================================

# ----------------------------- CONFIG: EDIT ME -------------------------------
GPU=1                                   # <-- GPU id on THIS server
REPO=/home/ssn899/Desktop/TabFM         # <-- repo root on THIS server
CONDA_ENV=tabfm                         # <-- conda env name
OUT_ROOT=$REPO/results/main_experiments # <-- where results are written

# Memory savers (activation checkpointing). Identical loss/gradient, ~20% slower,
# but cuts peak GPU memory a lot. Peak memory per run at context_5000 + 1000 test:
#   wids        ~74 GB  ->  ~10 GB with --recompute-layers
#   url_unique  ~46 GB  ->  ~10 GB with --recompute-layers
#   lcld_v2     ~23 GB      coil2000 ~20 GB   (usually fine as-is)
# Set to "" to disable, "--recompute-layers" to enable.
#
# This applies to BOTH attacks: the peak is any GRADIENT forward over the context.
# For label-flip that is the influence-seed scoring run at the top of every trial
# (MEASURED on wids: 73.7 GiB -> 9.6 GiB), not the GA itself -- the GA's batched
# fitness forward is no-grad and peaks at only 7.6 GiB at --ga-batch-size 16, and
# --recompute-layers does not change it. --test-batch-size barely helps the gradient
# forward (wids 73.7 -> 65.2 GiB) and is ~6x slower, so it is not the lever to reach for.
#
# 73.7 GiB fits on an idle 93 GiB card but NOT next to a neighbour: the 2026-09-16
# wids label-flip runs died with CUDA OOM against an 18.6 GiB co-tenant on GPU 1 and
# an 81.9 GiB one on GPU 3. Leaving this ON makes the runs immune to that.
MEM_SAVER_WIDS="--recompute-layers"     # <-- keep ON unless you have a >80GB GPU to spare
MEM_SAVER_URL=""                        # <-- set to "--recompute-layers" on GPUs <48GB
MEM_SAVER_LCLD=""
MEM_SAVER_COIL=""

# Save the poisoned context as CSV. Writes ONE file per command: the single strongest
# trial (largest CE increase) out of the n_runs x n_row_subsamples grid, saved as
# <out>/context_poisoned.csv, with the winning trial recorded in summary.json.
# ~55 MB for the whole 32-command grid. Set to "" to disable; the per-trial deltas are
# saved in trials/run{r}_sub{s}_delta.npz either way.
SAVE_CONTEXT="--save-full-context"

# How python is invoked. Change if conda lives elsewhere on that server, e.g.
#   PY=(/path/to/envs/tabfm/bin/python)
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
# -----------------------------------------------------------------------------

set -u
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }

run () {   # run <dataset_dir> <x-capgd|label-flip> <random|class0|class1> <row_percent>
  local ds=$1 atk=$2 tgt=$3 rp=$4
  local cls=() extra=()
  [[ $tgt == class* ]] && cls=(--attack-class "${tgt#class}")
  case "$ds" in
    wids)                        extra=($MEM_SAVER_WIDS) ;;
    url_unique)                  extra=($MEM_SAVER_URL)  ;;
    lcld_v2)                     extra=($MEM_SAVER_LCLD) ;;
    coil2000_insurance_policies) extra=($MEM_SAVER_COIL) ;;
  esac
  local tag; tag=$(printf "%s_%s_r%03d" "$atk" "$tgt" "$rp")
  local out="$OUT_ROOT/$ds/$tag"

  if [[ -f "$out/summary.json" ]]; then
    echo "[skip] $ds/$tag (already done)"; return 0
  fi

  local cmd=("${PY[@]}" "$REPO/scripts/attack_context.py"
             --gpu "$GPU" --attack "$atk"
             --train "$REPO/datasets/$ds/splits/selected/natural/context_5000.csv"
             --test  "$REPO/datasets/$ds/splits/test_attack_1000.csv"
             --out   "$out" --row-percent "$rp" $SAVE_CONTEXT "${cls[@]}" "${extra[@]}")

  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

  mkdir -p "$out"
  echo "[$(date +'%F %T')] START $ds/$tag"
  "${cmd[@]}" > "$out/stdout.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +'%F %T')] DONE  $ds/$tag"
  else
    echo "[$(date +'%F %T')] FAIL  $ds/$tag (rc=$rc, see $out/stdout.log)"
  fi
  return 0        # keep going even if one run fails
}

# ============================== COMMANDS =====================================
# coil2000 has NO relation constraints. Its command is identical: --constraints
# full still applies the type/immutability/eps-budget repair, there is simply
# nothing for the relation penalty to act on (--constraint-penalty is inert).
#
# Cost per command. x-capgd = 15 trials (5 runs x 3 subsamples) x n-iter 40 (the default)
# = 600 forward+backward passes; label-flip = 5 trials x ~600 GA evals (forward only,
# independent of n-iter). Per-iteration cost scales ~linearly with the number of feature
# groups (features/2), MEASURED at context_5000 + 1000 test, best-effort:
#     lcld_v2     28 feat / 15 groups -> 1.29 s/iter ->  52 s/trial -> ~13 min per command
#     url_unique  63 feat / 32 groups -> 2.31 s/iter ->  92 s/trial -> ~23 min per command
#     wids       108 feat / 55 groups -> 4.88 s/iter -> 195 s/trial -> ~49 min per command
#                (wids figure includes the ~25% --recompute-layers overhead; 3.9 s/iter without)
#     coil2000    85 feat / 43 groups -> ~3.0 s/iter (ESTIMATED, not yet measured)
# Whole file ~15 GPU-hours (x-capgd ~7.7 h, label-flip ~7.3 h).
# Note: cost is independent of R -- every iteration is a full forward+backward over the
# entire context+test regardless of how many rows are being attacked.

# --------------------------------------------------------------------------
# BLOCK 1of8: coil2000  X_train CAPGD              (~2.0 h total, ESTIMATED)
# --------------------------------------------------------------------------
run coil2000_insurance_policies x-capgd random 1
run coil2000_insurance_policies x-capgd random 4
run coil2000_insurance_policies x-capgd random 16
run coil2000_insurance_policies x-capgd random 64

# --------------------------------------------------------------------------
# BLOCK 2of8: coil2000  Y_train label-flip (GA)    (~1.5 h total, ESTIMATED)
# --------------------------------------------------------------------------
run coil2000_insurance_policies label-flip random 1
run coil2000_insurance_policies label-flip random 4
run coil2000_insurance_policies label-flip random 16
run coil2000_insurance_policies label-flip random 64

# --------------------------------------------------------------------------
# BLOCK 3of8: lcld_v2   X_train CAPGD              (~0.9 h total)
# --------------------------------------------------------------------------
run lcld_v2 x-capgd random 1
run lcld_v2 x-capgd random 4
run lcld_v2 x-capgd random 16
run lcld_v2 x-capgd random 64

# --------------------------------------------------------------------------
# BLOCK 4of8: lcld_v2   Y_train label-flip (GA)    (~1.1 h total)
# --------------------------------------------------------------------------
run lcld_v2 label-flip random 1
run lcld_v2 label-flip random 4
run lcld_v2 label-flip random 16
run lcld_v2 label-flip random 64

# --------------------------------------------------------------------------
# BLOCK 5of8: url_unique  X_train CAPGD            (~1.5 h total)
# --------------------------------------------------------------------------
run url_unique x-capgd random 1
run url_unique x-capgd random 4
run url_unique x-capgd random 16
run url_unique x-capgd random 64

# --------------------------------------------------------------------------
# BLOCK 6of8: url_unique  Y_train label-flip (GA)  (~1.4 h total)
# --------------------------------------------------------------------------
run url_unique label-flip random 1
run url_unique label-flip random 4
run url_unique label-flip random 16
run url_unique label-flip random 64

# --------------------------------------------------------------------------
# BLOCK 7of8: wids  X_train CAPGD    (~3.3 h total  <-- the expensive block)
# --------------------------------------------------------------------------
run wids x-capgd random 1
run wids x-capgd random 4
run wids x-capgd random 16
run wids x-capgd random 64

# --------------------------------------------------------------------------
# BLOCK 8of8: wids  Y_train label-flip (GA)        (~3.3 h total)
# --------------------------------------------------------------------------
run wids label-flip random 1
run wids label-flip random 4
run wids label-flip random 16
run wids label-flip random 64

echo "ALL REQUESTED RUNS FINISHED"

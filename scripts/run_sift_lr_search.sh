#!/usr/bin/env bash
# =============================================================================
#  Learning-rate search for the sift optimiser, on url_unique
#
#  10 iterations fixed; the only thing varied is the Adam learning rate. Because Adam
#  normalises by gradient magnitude, one step moves each logit by roughly `lr`, so over
#  10 iterations the grid below spans "barely moves" (0.1 -> logits shift ~1) to "fully
#  saturated" (5.0 -> ~50). Only the SPREAD of X matters to the attention (softmax is
#  shift-invariant), so the useful range is wherever the spread grows without the
#  ranking degenerating.
#
#  Hub-flip at R=30%, the case the k-fold filter handled worst: its detection precision
#  there was 0.305, barely above the 0.30 a random pick would score.
#
#  Every iteration's full X vector and per-row cross-entropy are written to history.npz,
#  so a run that peaks early and then decays, or drifts into a degenerate solution, can
#  be told apart afterwards from one that genuinely improved.
#
#  Each run extracts the kept subset under all three class-quota rules
#  (proportional / balanced / global) from the SAME optimisation, so they cost nothing
#  extra to compare.
#
#  Cost: ~59 s per iteration on url_unique (5000x63, chunk 256), so ~20 min per run at
#  20 iterations. The 6 learning rates are split over two GPUs, ~1 h per GPU:
#
#    GPU=0 LRS="0.1 0.5 2.0" bash scripts/run_sift_lr_search.sh &
#    GPU=3 LRS="0.3 1.0 5.0" bash scripts/run_sift_lr_search.sh &
#
#  Re-running skips any cell whose summary.json already exists.
# =============================================================================

GPU=${GPU:-0}
REPO=/home/ssn899/Desktop/TabFM
CONDA_ENV=tabfm
OUT_ROOT=${OUT_ROOT:-$REPO/results/sift}
DATASET=${DATASET:-url_unique}
ROW_PERCENT=${ROW_PERCENT:-30}
N_ITER=${N_ITER:-20}
CHUNK=${CHUNK:-256}
LAYERS=${LAYERS:-all}
LRS=${LRS:-"0.1 0.3 0.5 1.0 2.0 5.0"}
ATTACKS=${ATTACKS:-"hub-flip"}

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

set -u
cd "$REPO" || { echo "REPO not found"; exit 1; }

for atk in $ATTACKS; do
  for lr in $LRS; do
    tag=$(printf "%s_r%03d_lr%s_it%s" "$atk" "$ROW_PERCENT" "$lr" "$N_ITER")
    out="$OUT_ROOT/$DATASET/$tag"
    if [[ -f "$out/summary.json" ]]; then echo "[skip] $tag"; continue; fi
    echo "[$(date +'%F %T')] START $tag"
    "${PY[@]}" "$REPO/scripts/sift_context.py" \
      --gpu "$GPU" --dataset "$DATASET" --attack "$atk" \
      --row-percent "$ROW_PERCENT" --lr "$lr" --n-iter "$N_ITER" \
      --chunk "$CHUNK" --layers "$LAYERS" --out "$OUT_ROOT" > /dev/null 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && echo "[$(date +'%F %T')] DONE  $tag" \
                    || echo "[$(date +'%F %T')] FAIL  $tag (rc=$rc)"
  done
done
echo "ALL REQUESTED RUNS FINISHED"

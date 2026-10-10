#!/usr/bin/env bash
# Gate-valsplit sweep, part 1 of 10: this part's share of the runs of
# scripts/run_gate_valsplit_sweep.sh (the experiment is described there), run one after
# another on ONE GPU. Start the 10 parts together, one per GPU; between them they cover
# all 20 runs once, one gas-drift and one url_unique run each (about an hour per part).
#
#     bash scripts/gate_valsplit_parts/run_part1.sh [gpu]      (default GPU 0)
#
# Under a scheduler that sets CUDA_VISIBLE_DEVICES to one GPU, pass 0. Any other setting
# of the sweep can be overridden through the environment or .env as usual, e.g.
#     WANDB_MODE=offline bash scripts/gate_valsplit_parts/run_part1.sh 3
#
# With the default grid, this part runs (hub-flip-local; PGD, L-inf step, lr 0.05,
# 300 updates; validation seed 0):
#     gas-drift   R=10%  V=100
#     url_unique  R=20%  V=500
GPU=${1:-0}
export PART=1 NPARTS=10 GPUS="$GPU"
exec bash "$(dirname "${BASH_SOURCE[0]}")/../run_gate_valsplit_sweep.sh"

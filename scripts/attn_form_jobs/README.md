# Running the attention-form study on two machines

The 24 configurations (param/init/update) are split so that each machine runs whole configurations (learning-rate
grids, gas-drift / coil2000 runs and log2000 checks included); no run is done on both.

| machine | share (`--only`) | how |
|---|---|---|
| main (GPUs 2, 3) | `w/*/*,sigmoid/*/adam,sigmoid/*/pgd` | `ONLY='w/*/*,sigmoid/*/adam,sigmoid/*/pgd' bash scripts/attn_form_jobs/launch.sh 2 3` |
| second (4 GPUs) | `log*/*/*,sigmoid/*/sign` | `bash scripts/attn_form_jobs/remote_gpu<0-3>.sh`, one per GPU |

The second machine needs this repo's current scripts, plus three things git does not carry:
- `results/bench_largev/draws/*_R30_V500_s0.npz` and `results/attn_form/preflight_reference/` (copy `results/`);
- the 6 datasets' split files under `datasets/` (git-ignored);
- the model checkpoint the code loads, `~/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`.

Each `remote_gpu<N>.sh` first runs `check_remote.py`, which stops everything unless the inputs and the checkpoint
are byte-identical to the main machine's (`inputs.sha256`, the checkpoint's sha256), tabpfn comes from this repo's
`TabPFN/src` with the row-weight hooks, and 3 updates of every configuration reproduce the main machine's held-out
log loss.

When the scripts have exited, bring the runs here with
`bash scripts/attn_form_jobs/merge_remote.sh <host>:<its repo>/results/attn_form/runs`, then
`python scripts/report_attn_form.py`.

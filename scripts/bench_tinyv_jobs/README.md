# Tiny-V benchmark: 11 GPU scripts

Each `gpu_XX.sh` runs a fixed list of jobs (`gpu_XX.txt`, one `<instance> <method>` per line, longest first)
on one GPU, with 3 parallel workers. The lists are disjoint and together cover every job not done yet, so the
scripts can run on different machines without a shared folder. They were made by
`python scripts/plan_bench_tinyv.py --scripts 11`. Each script is about 2.4 H100-hours (estimate, ±50%); the
estimated job counts and hours per script are in `split.md`.

## Before starting

1. **Copy the shared draws.** Every machine needs `results/bench_tinyv/draws/` (54 small files, already built).
   The scripts refuse to run without them, and they must not be rebuilt elsewhere.
2. **Python environment.** Each machine needs the `tabfm` environment. If its python is not at the path the
   scripts use by default, set `PY=/path/to/python`.

## Running

```
GPU=0 bash scripts/bench_tinyv_jobs/gpu_00.sh      # GPU = the device index; WORKERS=3 by default
GPU=1 bash scripts/bench_tinyv_jobs/gpu_01.sh
...
GPU=10 bash scripts/bench_tinyv_jobs/gpu_10.sh
```

- **Logs:** each worker logs to `logs/bench_tinyv/gpu_XX_wN.log`.
- **Progress:** `python scripts/bench_tinyv.py status --job-list scripts/bench_tinyv_jobs/gpu_XX.txt`.
- **Interruptions:** a script can simply be started again; finished jobs are skipped.

## Afterwards

1. **Merge the results.** Copy every machine's `results/bench_tinyv/runs/` and `results/bench_tinyv/grids/` into
   this machine's `results/bench_tinyv/`. There is one file per (instance, method), so the copies never collide.
2. **Make the tables.** Run `python scripts/report_bench_tinyv.py`, which writes `results/tables/bench_tinyv/`.
   The schedule study's results for 24 instances (refs, V-filter, greedy 1% and 10%) are already imported here.
   `python scripts/elo_bench_tinyv.py` adds the TabArena-style Elo (`elo.md`, `elo.csv`).

## The missing attention jobs (re-run, done)

**Done on 2026-10-19:** the three `attn_rerun` scripts below ran on ecen-a86552 (GPUs 0-2, about 17
minutes) and finished all 63 jobs, so the benchmark is complete. With 3 workers per GPU, the third worker ran out
of memory beside two gas-drift or coil2000 attention jobs (about 45 GB each); the other workers picked those jobs
up. The rest of this section is how it happened and how to repeat it.

None of the 63 `attn_topk` / `attn_soft` jobs of gpu_04-gpu_10 finished: those scripts ran on ecen-a86553
(gpu_04-07) and ecen-a86555 (gpu_08-10), where every other job finished. All 45 attention jobs of gpu_00-03
finished on ecen-a86552. Only the attention learners use the row-weight hooks of the repository's TabPFN
(`TabPFN/src`: `architectures/tabpfn_v2.py`, `preprocessing/steps/context_row_weights.py` and two other
preprocessing steps, all added in commit 86df5d0). So the likely cause is a `tabfm` environment there that
imports another TabPFN copy; the worker logs on those machines (`logs/bench_tinyv/gpu_04_w1.log` etc.) will
say. `attn_rerun/gpu_00.sh`, `gpu_01.sh` and `gpu_02.sh` split the 63 jobs (about half an hour each on an H100 with
3 workers; a final single-worker pass retries any job that ran out of memory). Run them on a
machine whose `python -c "import tabpfn; print(tabpfn.__file__)"` points into this repository's `TabPFN/src`
(ecen-a86552 does):

```
GPU=0 bash scripts/bench_tinyv_jobs/attn_rerun/gpu_00.sh    # logs: logs/bench_tinyv/attn_rerun/
GPU=1 bash scripts/bench_tinyv_jobs/attn_rerun/gpu_01.sh
GPU=2 bash scripts/bench_tinyv_jobs/attn_rerun/gpu_02.sh
```

On a machine whose TabPFN lacks the hooks, the workers now skip the attention jobs and say so in the log
(`SKIPPING the attention jobs ...`) instead of failing each one.

## What runs

54 instances, all of seed 0: gas-drift, heloc, ecommerce_shipping, students_dropout, is_this_a_good_customer and
coil2000 × c = 1, 3, 5 clean rows per class × R = 10, 20, 30% poisoned (hub-flip-local).

Every row of the report is TabPFN with V + the method's cleaned rows on the held-out test set. It reports
accuracy, ROC-AUC, log loss and the precision of the removed rows.

| job | report rows |
|---|---|
| `refs` | V only; poisoned + V; oracle + V |
| `vfilter` | V-filter |
| `greedy_pct10`, `greedy_pct1`, `greedy_m1` | greedy, 10% / 1% / 1 row per step |
| `attn_topk` | attention weights, top-k (M1), 20 updates |
| `attn_soft` | attention weights, soft (A3), 20 updates; also its soft weights + V |
| `nan_topk` | NaN-input top-k gate, 4 restarts × 20 updates, vote |
| `ieg`, `emlc`, `label_spread` | default (remove); default (relabel); V-selected |
| `knn_shapley` | default (remove); V-selected |

The four baselines are the strongest of the earlier tiny-V benchmark: IEG, KNN-Shapley, EMLC and label
spreading, with mean accuracy over its 30 cells of 79.1, 76.2, 75.3 and 74.8. The other six scored 66.7-73.7;
`bench_tinyv.py` still runs them if named with `--methods`.

V only (in `refs`) and the V-filter cover all 54 instances: 24 imported from the schedule study, and 30 in these
job lists.

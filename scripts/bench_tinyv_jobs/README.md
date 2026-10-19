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

# Larger-V benchmark: 8 GPU scripts

Each `gpu_XX.sh` runs a fixed list of jobs (`gpu_XX.txt`, one `<instance> <job>` per line, longest first) on one
GPU. The 8 lists are disjoint and together hold every job not done yet: 816 jobs, about 72 H100-hours (±50%),
9 per script, so about 9 hours on 8 GPUs. They were made by `python scripts/bench_largev.py plan --scripts 8`.
`split.md` has the counts per script. The protocol and methods are in `docs/bench_largev.md`.

Grid: 6 datasets × V 50/100/200/300/400/500 × R 10/20/30% × one sampling of V (seed 0) = 108 instances. The
NaN gates run 200 updates.

## Before starting

1. **The shared draws.** Every machine needs `results/bench_largev/draws/` (built on 2026-10-19). Copy it; never
   rebuild it elsewhere.
2. **The environment.** The `tabfm` environment, with this repository's TabPFN copy installed
   (`pip install -e TabPFN`). Workers on a machine without it skip the attention jobs and log
   `SKIPPING the attention jobs ...`. If the environment's python is not at the path the scripts use by default,
   set `PY=/path/to/python`.

## Running

```
GPU=0 bash scripts/bench_largev_jobs/gpu_00.sh     # GPU = the device index
GPU=1 bash scripts/bench_largev_jobs/gpu_01.sh
...
GPU=7 bash scripts/bench_largev_jobs/gpu_07.sh
```

- **Workers:** 2 per GPU by default (`WORKERS=`). gas-drift and coil2000 jobs take about 45 GB each.
- **After the workers finish:**
  - one worker alone retries any job that failed (for example out of memory);
  - any top-k vote still missing is computed;
  - the script prints its progress and any FAILED or SKIPPING lines.
- **Logs:** `logs/bench_largev/bench_largev_jobs/gpu_XX_wN.log` (and `_retry.log`).
- **Progress:** `python scripts/bench_largev.py status`, or `status --job-list scripts/bench_largev_jobs/gpu_XX.txt`.
- **Interruptions:** start the script again; finished jobs are skipped.

## Afterwards

1. Copy every machine's `results/bench_largev/runs/` into one; there is one file per instance and job, so
   nothing collides.
2. Run `python scripts/bench_largev.py votes --device cuda:0` in case a top-k vote is still missing. Each list
   keeps an instance's four restarts together, so normally none is.
3. Run `python scripts/report_bench_largev.py`, which writes `results/tables/bench_largev/`.

## What is already done

Imported from the attention study, not in these lists (42 results, seed 0):
- `refs` and `attn_topk` for heloc, coil2000 and students_dropout × V 50 / 500 × R 10/20/30;
- `attn_soft` for heloc and students_dropout on the same grid.

Everything else is in the lists:

| job | jobs | H100-hours (est.) |
|---|---|---|
| `nan_topk_r0`-`r3` (top-k NaN gate, 4 restarts × 200 updates) | 432 | 42.4 (59%) |
| `nan_soft` (soft NaN gate, 200 updates) | 108 | 10.6 (15%) |
| `attn_soft` (A3, 200 updates) | 96 | 10.0 (14%) |
| `attn_topk` (M1, 200 updates) | 90 | 8.9 (12%) |
| `refs` | 90 | 0.2 |

By dataset: gas-drift 32.6 H100-hours, coil2000 23.3, heloc 6.6, students 4.5, ecommerce 4.3, good_customer 0.7.

# Tiny-V poisoning benchmark: runbook

**Question.** V holds a handful of clean rows per class. The rest of the train context is poisoned with
hub-flip-local at R%. How much test performance does TabPFN recover with V + a cleaned version of the
poisoned rows?

**Code:**
- `scripts/bench_tinyv.py`: draws, jobs and methods;
- `scripts/tinyv_learned.py`: the attention-weight and NaN-gate learners;
- `scripts/report_bench_tinyv.py`: tables;
- `scripts/elo_bench_tinyv.py`: the TabArena-style Elo (1 - ROC-AUC on binary datasets, log loss on
  multi-class ones; Bradley-Terry over the pairwise battles of the 54 instances, 200 bootstrap resamples).
- `scripts/archive_bench_tinyv.py`: freezes every table, the raw run records, the draws and removal masks and
  the code into `results_archive/bench_tinyv/<date>/` (git-tracked, with a README and a manifest), since
  `results/` is git-ignored. Snapshot of the complete benchmark: `results_archive/bench_tinyv/2026-10-19/`.

Results go to `results/bench_tinyv/` (git-ignored), and tables to `results/tables/bench_tinyv/`.

## Protocol

- **Instances:** 6 datasets × c ∈ {1, 3, 5} clean rows per class × R ∈ {10, 20, 30}% × seed 0, which is
  54 instances.

  | dataset | clean train context | test set |
  |---|---|---|
  | gas-drift | `splits/context_5000.csv` | `test_attack_1000` |
  | heloc | `splits/selected/natural/context_5000.csv` | `test_attack_1000` |
  | ecommerce_shipping | `splits/selected/natural/context_5000.csv` | `test_attack_1000` |
  | students_dropout | `splits/context_3000.csv` | `test_attack_500` |
  | is_this_a_good_customer | `splits/train.csv` (1400) | `test.csv` (300) |
  | coil2000 | `splits/selected/natural/context_5000.csv` | `test_attack_1000` |

- **The draw:** V = c random rows of every class (`bench_small_v.draw_per_class`). The rest of the context is
  poisoned with hub-flip-local at R%, so k = R% of its rows carry a wrong label. This is the construction the
  schedule study and the earlier tiny-V benchmark used.
- **Scoring:** TabPFNv2 with V + the cleaned rows as the context, on the test set:
  - accuracy, balanced accuracy and log loss;
  - ROC-AUC: binary, of class 1; multi-class, one-vs-rest macro;
  - the precision of the removed rows (for relabelling: of the relabelled rows).
- **Degenerate accuracy:** on coil2000 and is_this_a_good_customer, accuracy is degenerate (TabPFN predicts
  the majority class), so use ROC-AUC there.
- **Fairness:** a method sees only V, the poisoned rows (features and given labels), k and the number of
  classes. The test set and the poisoned-row identity are used only to score the result (and by the
  oracle reference).

## Methods

| job | what | hyperparameters |
|---|---|---|
| `refs` | V only; poisoned + V; oracle + V | none |
| `vfilter` | TabPFN(V) scores each poisoned row's given label; the k least plausible are removed | none |
| `greedy_pct10` | greedy growth of a trusted set from V; 10% of the poisoned rows join per step | fixed |
| `greedy_pct1` | the same, 1% per step | fixed |
| `greedy_m1` | the same, one row per step | fixed |
| `attn_topk` | learned attention weights, hard top-k (M1): PGD-L-inf lr 0.2, probe 0.05, V-filter start | 20 updates |
| `attn_soft` | learned attention weights, log w (A3): Adam lr 0.1, V-filter start, no prior | 20 updates |
| `nan_topk` | feature-input (NaN) top-k gate: relaxed straight-through α 0.05, lr 5, random start | 20 updates per restart; vote of 4 restarts' final masks |
| 4 baselines | IEG, KNN-Shapley, EMLC, label spreading (`small_v_methods.py`) | default; V-selected (below) |

**The baselines.** These are the four strongest of the earlier tiny-V benchmark (mean accuracy over its 30
cells):

| baseline | mean accuracy |
|---|---|
| IEG | 79.1 |
| KNN-Shapley | 76.2 |
| EMLC | 75.3 |
| label spreading | 74.8 |

The other six, GLC, cleanlab, Data-OOB, kNN sanitisation, MW-Net and SimiFeat, scored 66.7-73.7. They are
still in the code and run when named with `--methods`.

**Our methods.** All run at fixed settings; nothing is tuned per instance.
- **20 updates for every learner** (per restart for the gate). With 2-30 clean rows the objective is easy to
  overfit, and there are no rows to hold out for early stopping. The earlier studies ran 200 and 300 updates
  with V = 50-500. 20 rather than 10 gives the random-start NaN gate room to move off its random mask, while
  the attention learners, which start from the V-filter ranking, stay within a few log-units of it.
- The NaN gate's configuration and its vote of final masks come from `results/tables/gate_topk/night_log.md`.
  The vote needs no held-out rows.
- Caveat: the attention study chose M1 and A3 as its best configurations by comparing test results on
  heloc, coil2000 and students. That is a meta-level choice made before this benchmark.

**The baselines' tuning.** The earlier tiny-V benchmark tuned the baselines on tuning draws. Their
evaluation set was 20% of the context file with its clean labels, and the shortlist used the poisoned rows'
identity. Neither signal is available from V + the poisoned set, so those choices are not reused. Each
baseline is reported three ways:
- **default:** the defaults of `small_v_methods.py`, with z-scored features, in removal mode;
- **default, relabel:** the same configuration in relabel mode, where the method has one;
- **V-selected:** the baseline tuned using V as its validation set.
  - Each configuration of the method's grid, in each mode it has (remove, relabel, relabel_k, flagged), gives
    one cleaned set.
  - TabPFN is fitted on that cleaned set alone (V is not in its context) and predicts V's labels.
  - The configuration and mode with the lowest log loss on V win, and the winner's cleaned set + V is then
    scored on the test set.
  - It is admissible: only V and the poisoned set are used.
  - The catch: V is tiny (2-30 rows), so the choice is noisy, and V is also an input of most of these methods,
    so configurations that echo V's labels look better than they are.
  - `--no-vsel` drops it (default configurations only), which saves about 1.7 H100-hours with the four
    baselines.

## Already done (imported, not re-run)

`import-sched` copies the schedule study's seed-0 results whose draw is identical (it checks the arrays):
- **instances:** gas-drift, heloc, students_dropout and coil2000 × c ∈ {1, 5} × R ∈ {10, 20, 30}, which is
  24 instances;
- **methods:** `refs`, `vfilter`, `greedy_pct1`, `greedy_pct10` (these were computed on H100s with the same
  code).

Nothing else is reusable. The earlier attention, gate and baseline runs used other draws or inadmissible
tuning.

## Running it

**11 ready-made GPU scripts:** `scripts/bench_tinyv_jobs/gpu_00.sh` ... `gpu_10.sh`. Each runs a fixed, disjoint job list of about 2.4 H100-hours with 3 workers. Copy `results/bench_tinyv/draws/` to every machine first; `scripts/bench_tinyv_jobs/README.md` has the details. To re-split for another number of GPUs, run `python scripts/plan_bench_tinyv.py --scripts N`. The commands below are the manual alternative.

TabPFN's arithmetic differs slightly between CPU and GPU. A CPU re-run of one V-filter moved accuracy by
0.7 points. So run every job that uses TabPFN on GPUs of one type (H100, like the imported results). Only
the neural baselines' grid stage is meant for CPUs.

```
# 1. once, on one machine; then copy results/bench_tinyv/draws/ to every machine
python scripts/bench_tinyv.py draws --device cuda:1
python scripts/bench_tinyv.py import-sched

# 2. cheap GPU jobs (minutes per instance)
python scripts/bench_tinyv.py run --device cuda:1 --methods refs vfilter greedy_pct10 greedy_pct1

# 3. learned methods (20 updates each; a few minutes per instance, gas-drift and coil2000 the most)
python scripts/bench_tinyv.py run --device cuda:1 --methods attn_topk attn_soft nan_topk

# 4. one row per step: the most expensive (n - k TabPFN calls, up to 4500); one instance takes about
#    60 H100-minutes on gas-drift, 40 on coil2000, 15 on heloc, 10 on ecommerce, 6 on students, 1 on
#    good_customer. Give it its own GPUs and shard it
python scripts/bench_tinyv.py run --device cuda:1 --methods greedy_m1 --shard 0/2
python scripts/bench_tinyv.py run --device cuda:2 --methods greedy_m1 --shard 1/2

# 5. baseline grids: the neural ones on CPU (all four together 1-7 CPU minutes per instance);
#    the others on a GPU (cleanlab needs TabPFN)
python scripts/bench_tinyv.py grid --device cpu --methods glc mwnet emlc ieg --threads 8
python scripts/bench_tinyv.py grid --device cuda:1 --methods cleanlab knn_sanitize label_spread knn_shapley data_oob simifeat

# 6. baselines' V-selection and test scores (GPU, about 0.5-2 min per instance and baseline; needs step 5)
python scripts/bench_tinyv.py run --device cuda:1 --methods knn_sanitize label_spread knn_shapley data_oob cleanlab simifeat glc mwnet emlc ieg

# progress and tables (partial results are fine)
python scripts/bench_tinyv.py status
python scripts/report_bench_tinyv.py
python scripts/elo_bench_tinyv.py
python scripts/archive_bench_tinyv.py --tag $(date +%F)   # a frozen, git-tracked copy
```

**Workers.** Several workers can share one machine or a shared folder: jobs are claimed with lock files.

**Machines without a shared folder:**
- give each one the same draws/ folder, and split the jobs with `--shard i/N` (or `--datasets` / `--methods`);
- afterwards, copy their `runs/` and `grids/` folders together (one file per instance and method, so the copies
  never collide);
- a baseline's run job needs its grid file. If the grid ran elsewhere, copy `grids/` over first, or pass
  `--grid-if-missing`.

## Cost estimate

From measured timings: the schedule study's greedy runs, the attention study, the NaN-gate night and the
earlier benchmark's grids. Those ran with about 3 jobs per H100, and a job's share of one GPU is its time
divided by the measured 2.8 jobs per GPU. ecommerce and good_customer are extrapolated by rows and features.
Expect ±50%.

| H100-hours | gas | coil2000 | heloc | students | ecommerce | good_customer | total |
|---|---|---|---|---|---|---|---|
| greedy_m1 | 9.5 | 6.2 | 2.3 | 0.9 | 1.5 | 0.1 | **20.4** |
| greedy_pct1, pct10, vfilter, refs (c = 3 and the new datasets; the rest imported) | 0.07 | 0.05 | 0.02 | 0.01 | 0.03 | 0.01 | 0.2 |
| attn_topk + attn_soft (20 updates) | 0.47 | 0.31 | 0.09 | 0.05 | 0.06 | 0.01 | 1.0 |
| nan_topk (4 × 20 updates) | 0.31 | 0.20 | 0.07 | 0.05 | 0.05 | 0.01 | 0.7 |
| 4 baselines: grids + V-selection + test scores | 1.23 | 1.16 | 0.65 | 0.42 | 0.60 | 0.18 | 4.2 |
| **all GPU jobs** | 11.6 | 7.9 | 3.1 | 1.4 | 2.2 | 0.2 | **26.5** |

- **CPU:** none separately; in the 10 GPU scripts the baselines' grids run inside their GPU jobs (counted above).
- **Wall-clock:** with the 11 GPU scripts (about 2.4 H100-hours each), roughly 3 hours. With two H100s, about 13 hours.
- **Longest single job:** one greedy_m1 instance on gas-drift, about 3 hours when sharing a GPU with 2 other
  jobs, about 1 hour alone.

**Failures and testing flags.** A failed job leaves no output and is retried by the next worker.
`--max-steps`, `--restarts` and `--grid-limit` are for testing only: they shorten the learned methods and the
grids.

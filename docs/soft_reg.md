# Soft attention weights on all of V: learning rate and regularisation

Question (2026-10-21): can the best soft form (w in [0, 1], V-filter start, PGD) train for a long, fixed number of
updates on all of V (no held-out part, no early stopping) and beat its V-filter start, and the benchmarks' hard
V-filter, if the learning rate is lowered or the training is regularised?

Setup: R = 30%, development draws (seed 1, not the benchmarks' seed-0 instances): V = 50 and 100
(results/bench_largev/draws) and V = 5 rows per class (results/soft_reg/draws), on heloc, students_dropout and
is_this_a_good_customer (the informative datasets: room above the V-filter and a harmful poison; gas-drift is at the
ceiling, ecommerce's poison is harmless, coil2000 is expensive): 9 instances. 300 updates per arm, test metrics (soft +
V and soft) at updates 0-300 along the way. Two forms, both from the V-filter start with PGD steps:

- w in [0, 1] at lr 0.05 (its held-out choice 0.8, cut 16x for 300 updates); a row at w = 0 is deleted exactly and
  stays deleted (no gradient reaches it; kept explicitly under a pull toward the start);
- log w in [-1000, 0] at lr 0.0625 (the better of log w and sigmoid in the form study, its held-out choice 1.0 cut by
  the same 16x); weights never reach exactly 0, so nothing is permanent.

| arm (for each form) | what changes |
|---|---|
| control | nothing |
| mb0.5, mb0.25 | each update's loss on a class-stratified random half / quarter of V |
| l1x5 | proximal L1 toward the start: a row moves only while pushed harder than 5x the median row |
| l2 | proximal L2 toward the start: the deviation shrinks by 1 / (1 + g) per update (w: g = 0.05; log w: g = 0.003, so the hardest-pushed row can still cover the form's span) |
| knn10 | the gradient averaged over each row and its 10 nearest same-label neighbours |
| pseudo0.3 | extra queries: 200 of the 30% most plausible poisoned rows per update |

```
bash scripts/soft_reg_jobs/gpu_0N.sh [GPU]                 # N = 0, 1, 2; checks the machine first
bash scripts/soft_reg_jobs/merge.sh <host>:<repo path>     # bring the runs back
python scripts/report_soft_reg.py                          # -> results/tables/soft_reg/report.md, results/plots/soft_reg/
```

Code: scripts/soft_reg_study.py (draws, work, plan, status), scripts/report_soft_reg.py.

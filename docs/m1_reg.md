# M1 on all of V: regularisation

Question (2026-10-22): can M1 (hard top-k, V-filter start, PGD at lr 0.2) train for a long, fixed number of updates
on all of V (no held-out part, no early stopping) and beat its start, if the training is regularised? Deployed hard +
V, M1's start (update 0) is exactly the benchmarks' V-filter, so every arm is compared with the V-filter directly.
The M1 counterpart of docs/soft_reg.md.

Setup: R = 30%, development draws (seed 1, not the benchmarks' seed-0 instances): V = 50 and 100 (bench_largev's
draws) and V = 5 rows per class (the soft study's), on 5 datasets, 15 instances (results/m1_reg/draws, copied, so
they are the same instances as in those studies):

- the soft study's three: heloc (48/52), students_dropout (3 classes, 32/18/50), is_this_a_good_customer (89/11);
- gas-drift (6 classes, 12-22% each; 128 features). Its V-filter is close to the oracle in ROC-AUC but not in
  accuracy or log loss, so it is informative on those two.
- ecommerce_shipping (41/59). Its poison does no harm (poisoned + V = oracle + V in ROC-AUC); the V-filter's
  removals do (0.021 ROC-AUC below no cleaning), so it shows whether training undoes needless removals.

Room above the V-filter (oracle + V minus V-filter hard + V; log loss: V-filter minus oracle), and the poison's harm
(oracle + V minus poisoned + V), mean over each dataset's 3 instances:

| dataset | room: ROC-AUC | accuracy | log loss | harm: ROC-AUC | accuracy |
|---|---|---|---|---|---|
| heloc | 0.032 | 0.034 | 0.61 | 0.147 | 0.120 |
| students_dropout | 0.061 | 0.093 | 0.63 | 0.118 | 0.197 |
| is_this_a_good_customer | 0.138 | 0.094 | 0.60 | 0.151 | 0.137 |
| gas-drift | 0.015 | 0.107 | 0.70 | 0.044 | 0.272 |
| ecommerce_shipping | 0.021 | 0.041 | 1.01 | -0.001 | 0.042 |

300 updates per arm, all at M1's lr 0.2 (lr * g / max|g| per update, so the hardest-pushed score moves 0.2 nats).

| arm | what changes |
|---|---|
| m1 | nothing (the control) |
| m1_mb0.5, m1_mb0.25 | each update's loss on a class-stratified random half / quarter of V |
| m1_l1x5 | proximal L1 toward the start scores: a row's score moves only while pushed harder than 5x the median row |
| m1_l2_0.01 | proximal L2 toward the start scores: the deviation shrinks by 1 / 1.01 per update, so the hardest-pushed row can still move about 20 nats |
| m1_knn10 | the gradient averaged over each row and its 10 nearest same-label neighbours |
| m1_pseudo0.3 | extra queries: 200 of the 30% most plausible poisoned rows (start score) per update, each held out of the context for its own prediction |

At updates 0-300 (CKPTS): test metrics hard + V (the headline) and hard alone, V's log loss at the relaxed point (the
objective) and with the hard context, the precision of the removed rows, their overlap with the V-filter's removed
set, the rows that left the removed set so far, and the scores' AUROC against the poison. The test probabilities and
the removed sets are saved (.npz).

```
bash scripts/m1_reg_jobs/gpu_0N.sh [GPU]                  # N = 0..8; checks the machine first
bash scripts/m1_reg_jobs/merge.sh <host>:<repo path>      # bring the runs back
python scripts/report_m1_reg.py                           # -> results/tables/m1_reg/report.md, results/plots/m1_reg/
```

Cost: 105 arm runs (the 15 refs jobs are done on the main machine), about 19 H100-hours, 2.2 on each of the 9
scripts. ecommerce_shipping is cheap (10 features: about 4 min per arm); gas-drift dominates (128 features: about 29 min per arm, 57 with pseudo-queries, and about 44 GiB of GPU
memory; at most 9 GiB elsewhere). On a GPU shared with other jobs a gas-drift run can run out of memory; the script's
second pass retries it, and rerunning the script resumes.

Code: scripts/m1_reg_study.py (draws, work, plan, status), scripts/report_m1_reg.py.

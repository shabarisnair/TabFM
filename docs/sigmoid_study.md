# Sigmoid soft attention weights with t unbounded (R = 30%)

Question (2026-10-23): with w = sigmoid(t) per poisoned row, t any real number (never clipped) and trained by Adam,
which start and learning rate work at V = 50, 100 and 5 rows per class? And how does this compare with the best soft
form and with the form study's sigmoid? Unrelated to the sifting experiments.

Code: scripts/sigmoid_study.py (on attn_form_study.train, with the parameterisation `sigmoid_free` and the starts
`vf_hard` and `random<seed>` added there); jobs in scripts/sigmoid_jobs/.

## Setup

- **Weights:** w = sigmoid(t), log w = logsigmoid(t) in the attention; w in the label mean and the z-norms. t is never
  clipped.
- **Loss:** the log loss of the fitted 80% of V, with the weighted poisoned rows as the context. Adam (betas 0.9, 0.999)
  at lr 0.01, 0.1, 1, 5 and 10.
- **Stopping:** the held-out 20% of V is scored after every update up to 10, then every 5. A run stops 50 updates after
  its held-out log loss last improved by 1e-3 (at most 300 updates). The update with the lowest held-out log loss is
  deployed.
- **Deployment:** soft + V (every poisoned row at its weight, all of V at weight 1: the headline) and soft (the
  poisoned rows alone), on the test set.
- **Instances:** seed 0, the 6 datasets, three V sizes:
  - V = 50 and 100: the larger-V benchmark's draws and their 80/20 split of V;
  - 5 rows per class: the tiny-V benchmark's draws, with 1 row per class held out.

| start | t at the start |
|---|---|
| ones | 4 for every row (w = 0.982) |
| vf_soft | sigmoid(t) = the V-filter's soft weight p(given label \| fitted V) itself (not divided by its max; p kept in [1e-12, 1 - 1e-6]) |
| vf_hard | -30 for the k = R% rows the V-filter finds least plausible, 4 for the rest |
| random | N(0, 1), seeds 0-4; `random` is seed 0 |
| random_ens | the five random runs' weights sigmoid(t) averaged, each at its own chosen update, then deployed |

## Comparison

- **The best soft form (w / V-filter / PGD lr 0.8) with a 20% holdout:** the larger-V benchmark's attn_wsoft at V = 50
  and 100. At 5 per class it is run here (`best_soft`) with the same holdout.
- **The benchmarks' references:** V only, poisoned + V, V-filter (hard) and oracle + V.
- **At 5 per class, also:** the tiny-V benchmark's best soft (lr 0.2, 4 updates on all of V, no holdout).
- **Context:** the form study's sigmoid at V = 500 (t clipped to [-1000, 20]).

## Running

```
bash scripts/sigmoid_jobs/gpu_0N.sh [GPU]                 # N = 0..8; checks the machine first, resumes when rerun
bash scripts/sigmoid_jobs/merge.sh <host>:<repo path>     # bring the runs back
python scripts/sigmoid_study.py report                    # -> results/tables/sigmoid_study/report.md
```

The other machines need results/bench_largev/draws, results/bench_tinyv/draws, results/attn_form/preflight_reference
(for the check) and datasets/.

**Cost:** 366 jobs (18 instances x 4 starts x 5 learning rates; the random job runs 5 seeds and the ensemble; plus 6
best_soft references), about 56 H100-hours, about 6.3 hours on each of the 9 scripts. This assumes runs stop at about
120 updates on average; the form study's sigmoid runs peaked on the holdout at a median of update 20. On a GPU that
another busy job shares, it takes two to four times longer.

# Soft attention weights: which parameterisation, start and update rule

Question (2026-10-19): for the soft attention-weight learner (learn soft per-row weights, deploy them soft, nothing
removed), which parameterisation, start and update rule is best?

| factor | levels |
|---|---|
| parameterisation | w in [0, 1] (projected); w = sigmoid(t); log w in [-1000, 0]; log w in [-2000, 0] |
| start | w = 1 (sigmoid: 0.982); the V-filter start w = p(given label \| fitted V rows) / max |
| update | Adam; PGD (lr * g / max\|g\|); sign (lr * sign g) |

Setup: the larger-V benchmark's draws (`results/bench_largev/draws`), V = 500 (400 fitted, 100 held out), R = 30%,
seed 0, on gas-drift, coil2000, heloc, students_dropout, ecommerce_shipping and is_this_a_good_customer. All six
can hold out V = 500 from their clean train rows; is_this_a_good_customer is the tight one (1,400 train rows, so the
poisoned pool is 900 rows against 2,500-4,500 elsewhere). Each run stops once the held-out log loss has not improved
by 1e-3 for 30 updates (at most 200) and keeps the update with the lowest held-out log loss. The learning rate is
chosen per configuration on the held-out log loss of the four cheaper datasets (x4 grid, extended at an edge) and
reused on gas-drift and coil2000.

```
bash scripts/attn_form_jobs/launch.sh 2 3                 # 3 workers per GPU, one shared queue (lock files)
# on two machines without a shared disk: scripts/attn_form_jobs/README.md
python scripts/attn_form_study.py status
python scripts/report_attn_form.py                        # -> results/tables/attn_form/report.md, runs.csv
python scripts/attn_soft_response.py sweep|channels       # how TabPFN responds to a soft weight
```

Results: `results/attn_form/runs/<instance>/<param>_<init>_<opt>_lr<lr>.{json,npz}` (the npz holds the deployed and
the starting log w). Report and findings: `results/tables/attn_form/`.

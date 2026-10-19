# Tiny-V poisoning benchmark: frozen results (2026-10-19)

TabPFN v2 with V (c clean rows per class) + a method's cleaned version of the poisoned train context, scored on each dataset's held-out test set. 6 datasets x c in {1, 3, 5} x R in {10, 20, 30}% hub-flip-local poisoning, seed 0: 54 instances. The runbook is `docs/bench_tinyv.md`; the analysis is `findings.md`.

## Provenance

- Made 2026-10-19T03:08:52-05:00 on ecen-a86552 by `scripts/archive_bench_tinyv.py` from `results/bench_tinyv`.
- Git commit 86df5d0141e17da1e590d49efb7403e645134229, with uncommitted changes to docs/bench_tinyv.md, scripts/bench_tinyv.py, scripts/bench_tinyv_jobs/README.md, scripts/plan_bench_tinyv.py, scripts/report_bench_tinyv.py, scripts/archive_bench_tinyv.py, scripts/bench_tinyv_jobs/attn_rerun/, scripts/elo_bench_tinyv.py.
  `code/` holds the scripts as they were when this snapshot was made.
- 54 instances, 22 entries, 648 run files. Runs per host: ecen-a86552 267, ecen-a86553 160, ecen-a86555 125, imported 96.

## Files

| file | what |
|---|---|
| `results.csv` | one row per instance x entry: every metric (the source of every other table) |
| `by_v_r.md`, `by_v_r.csv` | one table per V size x poisoning rate: per dataset, precision, ROC-AUC, accuracy, log loss |
| `report.md` | summary over the instances and per-dataset tables of each metric |
| `per_dataset.csv` | means over each dataset's 9 instances, per entry |
| `elo.md`, `elo.csv` | TabArena-style Elo (main fit and sensitivity fits) |
| `datasets.csv` | per dataset: type, classes, class shares, balance, split files and sizes |
| `instances.csv` | per instance: V size, poisoned-set size and number of poisoned rows k |
| `findings.md` | the written analysis |
| `raw/runs.jsonl.gz` | every run record (configurations, timings, all rows), one JSON object per line |
| `raw/draws_and_masks.tar.gz` | the draws (`draws/<instance>.npz`: v_rows, ctx_rows, y_given, pois_rows, k) and the removal masks (`runs/<instance>/<method>.npz`) |
| `code/` | the scripts that ran the benchmark and made these tables |
| `manifest.json` | git commit, counts, sha256 of every file |

## Columns

`results.csv`:
- `instance` (`<dataset>_R<R>_c<c>_s<seed>`), `dataset`, `R` (% of the poisoned set given a wrong label), `c` (clean rows per class in V), `seed`;
- `entry`: the method (legend below);
- `acc`, `bacc`: test accuracy and balanced accuracy (0-1); `auc`: test ROC-AUC (binary: of class 1; multi-class: one-vs-rest macro); `ce`: test log loss;
- `prec`: share of the removed rows that were poisoned (relabel modes: of the relabelled rows); empty when nothing is removed. Each method removes k rows, so removing at random scores R/100 on average;
- `n_removed`: rows removed; `cfg`, `mode`: a baseline's configuration and mode (V-selected rows: the chosen ones); `imported`: taken from the schedule study's identical runs.

`by_v_r.csv`: `c`, `R`, `entry`, `method`, then `<tag>_<metric>` per dataset tag (below) and metric (`prec`, `auc`, `acc`, `logloss`).

`elo.csv`: `variant` (`main` is TabArena's procedure; the others are sensitivity fits), `entry`, `elo`, `lo_tasks`/`hi_tasks` (95% CI, task bootstrap), `lo_datasets`/`hi_datasets` (dataset bootstrap), `gap_lo_*`/`gap_hi_*` (95% interval of the Elo gap to poisoned + V), `win_rate`, `improvability`, `tasks`, `W`/`T`/`L` (wins, ties, losses against poisoned + V). Error: 1 - ROC-AUC on binary datasets, log loss on multi-class ones; Bradley-Terry fit; poisoned + V calibrated to 1000; oracle + V not rated.

## Datasets

| tag | dataset | type | balance | class shares (context) | context / test rows |
|---|---|---|---|---|---|
| gas | gas-drift | multi-class (6 classes) | roughly balanced | 0.184 / 0.210 / 0.118 / 0.139 / 0.216 / 0.132 | 5000 / 1000 |
| heloc | heloc | binary (2 classes) | roughly balanced | 0.478 / 0.522 | 5000 / 1000 |
| ecom | ecommerce_shipping | binary (2 classes) | roughly balanced | 0.407 / 0.593 | 5000 / 1000 |
| stud | students_dropout | multi-class (3 classes) | moderately imbalanced | 0.321 / 0.180 / 0.499 | 3000 / 500 |
| good | is_this_a_good_customer | binary (2 classes) | imbalanced | 0.886 / 0.114 | 1400 / 300 |
| coil | coil2000_insurance_policies | binary (2 classes) | imbalanced | 0.930 / 0.070 | 5000 / 1000 |

Balance: largest over smallest class share up to 2 is roughly balanced, up to 4 moderately imbalanced.

## Entries

| entry | method |
|---|---|
| `v_only` | V only |
| `poisoned_plus_v` | poisoned + V |
| `oracle_plus_v` | oracle + V |
| `vfilter` | V-filter |
| `greedy_pct10` | greedy, 10% per step |
| `greedy_pct1` | greedy, 1% per step |
| `greedy_m1` | greedy, 1 row per step |
| `attn_topk` | attention weights, top-k (M1) |
| `attn_soft` | attention weights, soft (A3) |
| `attn_soft_softw` | A3, soft weights + V |
| `nan_topk` | NaN gate, top-k vote |
| `label_spread` | label_spread (default) |
| `label_spread_relabel` | label_spread (default, relabel) |
| `label_spread_vsel` | label_spread (V-selected) |
| `knn_shapley` | knn_shapley (default) |
| `knn_shapley_vsel` | knn_shapley (V-selected) |
| `emlc` | emlc (default) |
| `emlc_relabel` | emlc (default, relabel) |
| `emlc_vsel` | emlc (V-selected) |
| `ieg` | ieg (default) |
| `ieg_relabel` | ieg (default, relabel) |
| `ieg_vsel` | ieg (V-selected) |

## Loading

```python
import gzip, json, tarfile
import numpy as np, pandas as pd

r = pd.read_csv('results.csv')                         # one row per instance x entry
cell = r[(r.c == 1) & (r.R == 10) & (r.dataset == 'heloc')]
elo = pd.read_csv('elo.csv').query("variant == 'main'")
runs = [json.loads(l) for l in gzip.open('raw/runs.jsonl.gz', 'rt')]
tarfile.open('raw/draws_and_masks.tar.gz').extractall('raw/')  # then np.load('raw/draws/<instance>.npz')
```

To make a new snapshot after more runs: `python scripts/archive_bench_tinyv.py --tag <date>`.

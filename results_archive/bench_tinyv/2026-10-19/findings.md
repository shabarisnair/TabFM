# Tiny-V poisoning benchmark: results and TabArena-style Elo (2026-10-19)

## Setup

- **Instances:** 54. Six datasets × c ∈ {1, 3, 5} clean rows per class in V × R ∈ {10, 20, 30}% hub-flip-local
  poisoning, seed 0.
- **Scoring:** TabPFN with V + each method's cleaned rows as its context, on the held-out test set.
- **Code:** `scripts/bench_tinyv.py`; tables from `report_bench_tinyv.py` (`report.md`, `by_v_r.md`,
  `results.csv`); Elo from `elo_bench_tinyv.py` (`elo.md`, `elo.csv`).
- **Completeness:** every method finished all 54 instances. The 63 attention jobs that failed on ecen-a86553
  and ecen-a86555 were re-run on ecen-a86552 on 2026-10-19 (`scripts/bench_tinyv_jobs/attn_rerun/`), so every
  attention run comes from that host and the same code.

## The Elo: TabArena's procedure, confirmed from its source

Source: github.com/autogluon/tabarena, `packages/bencheval/src/bencheval/elo_utils.py` and `evaluator.py`
(`compute_elo`), `packages/tabarena/src/tabarena/evaluation/leaderboard_reporter.py`.

| | TabArena | here |
|---|---|---|
| error | binary: 1 - ROC-AUC; multi-class: log loss; regression: RMSE | the same |
| battle | every two methods on every (dataset, fold); lower error wins, an exact tie is half a win; a dataset's folds weighted 1/n_folds | every two entries on every instance (task = dataset × c × R; one seed) |
| model | Bradley-Terry by maximum likelihood: logistic regression without intercept, x = ±ln 10, C = 1e6, tol 1e-10; Elo = 400 · coefficient + 1000 | the same (re-implemented; matches an independent scipy fit to 0.0 Elo) |
| point estimate | the full-data fit | the same |
| 95% CI | 200 bootstrap resamples of the tasks, 2.5/97.5% quantiles, before calibration | the same; also resampling the 6 datasets, and the interval of the gap to the anchor |
| anchor (1000) | RF (default) | poisoned + V (no cleaning) |
| missing results | imputed with RF (default)'s | none are missing |

oracle + V is not rated: it knows the poisoned rows.

## Leaderboard

Gap: the 95% interval of (entry's Elo − poisoned + V's Elo) over the 200 task resamples; an interval above 0
would mean the entry beats no cleaning beyond the noise. Accuracy, AUC, log loss and precision are means over
the 54 instances.

| # | entry | Elo | 95% CI | gap to no cleaning (tasks) | W-L vs no cleaning | accuracy | AUC (binary) | log loss | precision |
|---|---|---|---|---|---|---|---|---|---|
| 1 | emlc (default) | 1020 | 981–1065 | -77 to +101 | 30-24 | 72.6 | 69.1 | 0.798 | 0.26 |
| 2 | attention weights, soft (A3) | 1013 | 955–1073 | -40 to +67 | 21-33 | 73.1 | 67.9 | 0.635 | 0.27 |
| 3 | ieg (default) | 1000 | 953–1056 | -73 to +71 | 26-28 | 73.0 | 69.1 | 0.835 | 0.28 |
| 4 | poisoned + V | 1000 | 945–1076 | . | . | 73.5 | 68.6 | 0.609 | . |
| 5 | NaN gate, top-k vote | 989 | 941–1066 | -53 to +32 | 22-32 | 72.1 | 68.3 | 0.693 | 0.23 |
| 6 | knn_shapley (default) | 957 | 913–1006 | -111 to +8 | 24-30 | 74.1 | 68.0 | 0.762 | 0.36 |
| 7 | attention weights, top-k (M1) | 955 | 921–997 | -131 to +26 | 22-32 | 72.9 | 68.4 | 0.798 | 0.34 |
| 8 | V-filter | 938 | 906–979 | -155 to +17 | 23-31 | 72.8 | 68.4 | 0.847 | 0.32 |
| 9 | knn_shapley (V-selected) | 927 | 877–976 | -132 to -19 | 19-35 | 73.8 | 67.3 | 0.778 | 0.34 |
| 10 | ieg (default, relabel) | 894 | 840–954 | -205 to -2 | 22-32 | 68.2 | 66.9 | 2.124 | 0.37 |
| 11 | A3, soft weights + V | 893 | 842–953 | -189 to -34 | 15-39 | 72.2 | 66.3 | 0.698 | . |
| 12 | greedy, 1% per step | 878 | 813–958 | -238 to -25 | 18-36 | 76.9 | 63.8 | 1.159 | 0.52 |
| 13 | greedy, 1 row per step | 876 | 800–957 | -234 to -15 | 20-34 | 77.5 | 64.4 | 1.134 | 0.53 |
| 14 | label_spread (default) | 876 | 826–921 | -193 to -81 | 14-40 | 71.6 | 66.3 | 0.827 | 0.25 |
| 15 | greedy, 10% per step | 871 | 817–940 | -212 to -48 | 15-39 | 76.0 | 66.3 | 1.071 | 0.49 |
| 16 | ieg (V-selected) | 849 | 803–906 | -265 to -40 | 19-35 | 66.7 | 65.2 | 2.392 | 0.36 |
| 17 | V only | 712 | 649–767 | -396 to -203 | 11-43 | 55.7 | 60.3 | 0.803 | . |
| 18 | emlc (V-selected) | 647 | 566–718 | -490 to -243 | 10-44 | 52.6 | 62.1 | 3.598 | 0.24 |
| 19 | emlc (default, relabel) | 642 | 553–712 | -512 to -244 | 10-44 | 51.2 | 62.3 | 3.753 | 0.24 |
| 20 | label_spread (V-selected) | 565 | 452–654 | -618 to -304 | 8-46 | 51.7 | 58.5 | 3.887 | 0.24 |
| 21 | label_spread (default, relabel) | 484 | 372–573 | -658 to -411 | 4-50 | 50.8 | 55.1 | 2.982 | 0.23 |
| – | oracle + V (reference) | – | – | – | – | 84.3 | 77.0 | 0.350 | 1.00 |

**Sensitivity of the Elo** (`elo.md`; its "imputed" column now equals the main fit):

| entry | main | binary only | multi-class only | multi-class by accuracy | without ecommerce |
|---|---|---|---|---|---|
| emlc (default) | 1020 | 1052 | 956 | 1054 | 1091 |
| A3 | 1013 | 986 | 1076 | 1021 | 1040 |
| ieg (default) | 1000 | 1059 | 882 | 1039 | 1032 |
| poisoned + V | 1000 | 1000 | 1000 | 1000 | 1000 |
| NaN gate | 989 | 1000 | 966 | 1024 | 1000 |
| knn_shapley (default) | 957 | 950 | 969 | 995 | 1014 |
| M1 | 955 | 995 | 865 | 1000 | 1006 |
| V-filter | 938 | 989 | 821 | 991 | 993 |
| greedy, 1% per step | 878 | 784 | 1082 | 980 | 945 |
| greedy, 1 row per step | 876 | 820 | 990 | 984 | 923 |
| greedy, 10% per step | 871 | 849 | 908 | 1012 | 913 |

## Per dataset (means over the 9 instances)

LL = log loss (lower is better). The bold value is the best non-oracle entry in its column.

| entry | gas LL | students LL | gas acc | students acc | heloc AUC | ecom AUC | good_cust AUC | coil AUC |
|---|---|---|---|---|---|---|---|---|
| V only | 1.068 | 0.993 | 59.6 | 53.5 | 68.0 | 57.6 | 60.9 | 54.8 |
| poisoned + V | 0.908 | 0.714 | 80.5 | 68.4 | 71.7 | **75.5** | 70.3 | 56.9 |
| oracle + V | 0.018 | 0.532 | 99.3 | 80.5 | 80.2 | 75.4 | 79.8 | 72.4 |
| V-filter | 0.815 | 1.005 | 83.8 | 66.4 | 74.9 | 70.7 | 69.6 | 58.3 |
| greedy, 10% per step | 0.700 | 0.923 | 92.0 | 74.8 | 75.8 | 71.1 | 64.9 | 53.6 |
| greedy, 1% per step | **0.280** | 1.033 | **96.5** | **75.0** | **77.0** | 67.5 | 58.6 | 52.0 |
| greedy, 1 row per step | 0.315 | 1.167 | 96.2 | 73.4 | 73.9 | 69.6 | 58.3 | 55.9 |
| attention weights, top-k (M1) | 0.799 | 0.973 | 84.7 | 66.0 | 75.1 | 71.1 | 69.6 | 57.9 |
| attention weights, soft (A3) | 0.675 | 0.757 | 85.3 | 68.1 | 70.7 | 73.0 | **71.0** | 57.1 |
| A3, soft weights + V | 0.626 | 0.981 | 86.0 | 64.6 | 69.3 | 72.7 | 65.7 | 57.4 |
| NaN gate, top-k vote | 0.922 | **0.706** | 82.3 | 69.2 | 71.7 | 74.2 | 69.6 | 58.0 |
| label_spread (default) | 0.908 | 0.908 | 81.5 | 67.0 | 68.5 | 70.9 | 69.2 | 56.4 |
| label_spread (default, relabel) | 4.359 | 2.677 | 60.6 | 47.2 | 57.1 | 50.4 | 61.5 | 51.5 |
| label_spread (V-selected) | 3.953 | 6.786 | 67.5 | 28.3 | 65.1 | 50.4 | 59.7 | 58.6 |
| knn_shapley (default) | 0.889 | 0.803 | 83.9 | 69.7 | 74.6 | 70.4 | 70.1 | 57.2 |
| knn_shapley (V-selected) | 0.855 | 0.865 | 84.1 | 68.3 | 72.0 | 70.4 | 68.8 | 58.2 |
| emlc (default) | 0.661 | 0.944 | 87.5 | 64.0 | 75.3 | 70.7 | 70.5 | **60.0** |
| emlc (default, relabel) | 3.011 | 4.660 | 67.0 | 38.1 | 71.6 | 58.1 | 62.0 | 57.3 |
| emlc (V-selected) | 2.724 | 4.229 | 70.2 | 41.1 | 71.6 | 57.6 | 62.3 | 56.8 |
| ieg (default) | 0.771 | 0.993 | 83.5 | 67.2 | 76.6 | 72.1 | 70.7 | 57.0 |
| ieg (default, relabel) | 0.527 | 2.080 | 88.1 | 59.1 | 76.1 | 64.5 | 70.0 | 56.7 |
| ieg (V-selected) | 0.591 | 2.282 | 88.8 | 59.8 | 75.7 | 60.4 | 68.3 | 56.4 |

Every entry × (c, R) × metric: `report.md`; one table per V × R: `by_v_r.md`.

## Findings

1. **By TabArena's metric, no method beats doing nothing.**
   - The top four, EMLC (default) 1020, A3 1013, IEG (default) 1000 and the NaN gate 989, are within noise of
     poisoned + V. Every gap interval contains 0, and no entry has one above 0 under either bootstrap.
   - A rating above 1000 does not mean winning head-to-head against no cleaning: A3 loses to it on 33 of 54
     instances (21-33). Its rating comes from beating the other cleaning methods more often.
   - Significantly below no cleaning over the task bootstrap: all three greedy schedules, KNN-Shapley
     (V-selected), IEG (relabel; just, -205 to -2), label spreading (default), A3 soft + V, IEG (V-selected),
     V only, and the relabel and V-selected variants of EMLC and label spreading.
   - Resampling whole datasets, only label spreading (default), V only and the relabel/V-selected
     EMLC and label-spreading variants stay significantly below. That bootstrap has only six units.
2. **Greedy is the best detector and the most accurate method, but its gains are concentrated on few
   datasets.**
   - It has the highest precision, 0.49-0.53. Random removal of k rows would average 0.20, and the next best
     entries reach 0.34-0.37 (KNN-Shapley, M1 and IEG's relabel variants).
   - It has the highest mean accuracy: 76.0-77.5 against 73.5.
   - On gas-drift it nearly closes the gap to the oracle (1% per step: accuracy 96.5 against 80.5 and 99.3,
     log loss 0.28 against 0.91 and 0.02, precision 0.94). On heloc it adds 5.3 AUC points (77.0 against
     71.7; oracle 80.2).
   - It loses badly elsewhere:
     - **Imbalanced binary data.** On coil2000 (7% positives), 92.5% of the poisoned rows carry the minority
       label. Greedy 1% removes 47% of the genuine minority rows but only 15% of the genuine majority rows, so
       AUC falls to 52.0 (against 56.9). On good_customer (11% minority) it removes 63% against 7.5%: AUC 58.6
       against 70.3. On balanced heloc it removes the two classes about equally (10% and 13%), and AUC rises.
     - **Calibration.** All three schedules' log loss is worse than no cleaning on every one of the 27 heloc,
       students and good_customer instances, and it worsens with R (heloc R = 30: 1.18-1.66 against 0.68). On
       students, accuracy (+6.6) and AUC (+2.0) rise while log loss rises from 0.71 to 1.03. So the mistakes it
       still makes are confident ones.
     - The likely cause: removing every row that disagrees with the trusted set also strips the genuinely
       ambiguous clean rows, so the context looks more separable than the data is.
   - TabArena's metric is AUC on 4 of the 6 datasets and log loss on the other 2, so both failure modes count
     fully. By accuracy on the multi-class datasets, greedy's Elo rises to 980-1012, about level with no
     cleaning. The schedule study ranked schedules by accuracy only, which is why it looked like the winner
     there.
3. **The attention-weight methods:** A3 is the best-calibrated cleaner, and M1 is close to the V-filter.
   - A3 has the lowest log loss of any cleaning method (0.635; no cleaning 0.609), helped by gas-drift (0.675
     against 0.908). Yet it beats no cleaning's log loss on only 10 of 54 instances.
   - M1 sits next to the V-filter it starts from: Elo 955 against 938, precision 0.34 against 0.32, the same
     binary AUC (68.4).
4. **The entries at the top barely move the context.**
   - The NaN gate (0.23) and A3 (0.27) remove poisoned rows at close to the random rate of 0.20.
   - The NaN gate starts from random scores and gets 20 updates. A3 starts from the V-filter ranking and drifts
     below it (0.27 against the V-filter's 0.32).
   - EMLC (0.26) and IEG (0.28) are not much better.
   - Their Elo is close to 1000 most likely because removing k rows at a near-random rate changes little, not
     because they find the poison. A random-removal control would show this directly; it is not in the benchmark.
5. **There is little to recover on most datasets, and none on ecommerce.**
   - The oracle's gain is large on gas-drift (log loss 0.91 → 0.02) and coil2000 (AUC 56.9 → 72.4). It is
     moderate on students, good_customer and heloc (+7.2 to +9.5 AUC points).
   - On ecommerce it is zero (75.5 → 75.4): hub-flip-local does not hurt TabPFN there, so every removal can only
     lose.
   - Without ecommerce, EMLC rises to 1091, A3 to 1040 and IEG to 1032; greedy stays at 913-945.
6. **Gains appear only at R = 30.**
   - In binary AUC at R = 30: IEG +2.2 points, EMLC +1.6, M1 +1.1, V-filter +0.9, NaN gate +0.6 over no
     cleaning. The oracle gains +12.5 there.
   - At R = 10 nothing gains more than 0.1 point (oracle +4.5).
7. **Relabelling and V-selection hurt every baseline.**
   - V-selection picked relabelling on 52/54 instances for EMLC, 54/54 for IEG and 51/54 for label spreading.
     Labels copied from V's model look best on V itself.
   - It cost EMLC 373 Elo, IEG 151, label spreading 311 and KNN-Shapley (no relabel mode) 30.
   - The defaults in removal mode are the only admissible baseline settings worth keeping.
8. **Discarding the poisoned rows is far worse than keeping them.** V only is at 712. Even at R = 30, the
   poisoned set carries most of the signal.

## Caveats

- **One draw per instance**, and the instances of one dataset share its data, so the dataset bootstrap is the
  honest interval, and it is wide.

## Next steps

- **A random-removal control** (k random rows; 54 TabPFN fits, a few GPU minutes) to calibrate finding 4.
- **For greedy**, the two failure modes suggest two changes:
  - a per-class removal cap (or protecting minority rows its trusted set cannot judge);
  - a calibration fix: soft down-weighting instead of hard removal, or a removal budget below k, since half of
    its k removals are clean rows.

# The V + P benchmark (2026-10-26)

**Question:** given a tiny clean set V and a label-poisoned train set P, plus the poisoning rate R, which method makes
TabPFN v2 score best on a held-out test set? A method may use a training context of any size. What matters is the test
ROC-AUC, balanced accuracy and log loss. Removal precision and recall are reported where a method keeps a subset of P.

**Code:**
- scripts/bench_vp.py: the benchmark;
- scripts/plan_vp.py: the 10 scripts in scripts/vp_jobs/, and the 10 in scripts/vp_jobs_rerun/ for the jobs the first
  run left;
- scripts/vp_tune.py and scripts/vp_tune_report.py: the tuning on development draws (results/tables/vp_tune/findings.md);
- scripts/vp_tune_repair.py: the repair methods' (12, 13) development study, and its 7 GPU scripts in
  scripts/vp_tune_repair_jobs/;
- scripts/vp_cpu_workers.sh: runs benchmark jobs or that study on this machine's CPU (several workers share a list).

## Instances

180 instances:
- **Datasets:** the 6 of bench_tinyv, each with a fixed train set (its context file, about 5,000 rows) and a fixed test
  set.
- **V:** c = 1, 3 or 5 clean rows per class, drawn from the train set (bench_small_v.draw_per_class). For one seed, a
  smaller V is contained in a larger one.
- **Seeds:** 0-4. Each seed gives its own V and therefore its own P.
- **Poisonings of P** (the rest of the train set):
  - **rand45:** label-flip-bernoulli at R = 45%. Every row independently, with probability R, moves to a uniformly
    random other class. The flips follow the seed: 44.9% of P on average (43.1-46.3%).
  - **hub30:** hub-flip-local at R = 30%. The 30% densest rows move to their locally closest other class. This is
    deterministic given P, so the five seeds differ mainly in V.
- **k = round(R% of |P|):** what a method that uses R removes.
- **Draws:** results/bench_vp/draws/<dataset>_<poisoning>_c<c>_s<seed>.npz, built once by `bench_vp.py draws`. The
  development draws (seeds 100-101, in results/vp_tune/) are separate.

## Methods

Each method is scored by TabPFN on the test set with the context it builds. A method that keeps a subset of P is also
scored by the precision and recall of the rows it removes and the poisoned share of the rows it keeps. As a train-side
measure, the kept subset also gets V's log loss and accuracy with the kept rows of P alone as the context (v_ce,
v_acc).

| # | row | job | definition |
|---|---|---|---|
| 1 | v_only | refs | TabPFN with V alone as the context |
| 2 | vfilter_R | vfilter | p = TabPFN(V)'s probability of each row's given label; the k lowest removed |
| 3 | vfilter_thr | vfilter | the rows with p < 0.5 / K removed (half of chance), no minimum removal |
| 4 | greedy_R | greedy | greedy at 1% per step, stopped at (1 - R)% of P |
| 5 | greedy_R10 | greedy | the same run, 10 steps further |
| 6 | sigmoid | sigmoid | attention weights w = sigmoid(t) trained on V's log loss from the hard V-filter, deployed soft + V at iteration 10 |
| 7 | m1_last, m1_sel | m1 | M1, top-k attention scores trained on V's log loss, deployed hard + V |
| 8 | knn_shapley, ieg | baselines | KNN-Shapley with its own rule, IEG with its top R% |
| 9 | v_plus_p | refs | V + all of P |
| 10 | oracle | refs | V + P's clean rows |
| 11 | greedy_thr | greedy_thr | greedy at 1% per step, taking only rows with p(given label) > 0.5; stops when none qualifies |
| 12 | vrepair | vrepair | P's labels unused: TabPFN(V) labels P, and the rows it gives a max probability > 0.9 join V |
| 12 (0.5) | vrepair_05 | vrepair | the same at > 0.5 (the user's variant); on binary data this labels all of P |
| 13 | grepair | grepair | the iterative form of 12: the most confident 1% joins the trusted set per step, labelled; stops at 5% of P |

**Details:**
- **greedy (methods 4 and 5)** reproduces bench_tinyv's greedy_pct1, and its (1 - R)% set was checked identical to
  schedule_study's fixed:pct1.
  - The trusted set starts as V. Every step, TabPFN with V + the accepted rows (their given labels) as the context
    scores the rest. The ceil(1% of |P|) rows whose given label it finds most probable join.
  - The batch that reaches (1 - R)% is cut to land on it exactly.
  - Method 5 then takes 10 more full steps: about (1 - R + 10)% of P kept, about R - 10% removed. On the development
    draws this was the best stop: an uncertainty or agreement stop lands near (1 - R)% and is worse.
- **sigmoid (method 6)** is the sigmoid study's form (scripts/sigmoid_study.py).
  - **Training:** all of V as the queries and P weighted as the context.
  - **Start:** the hard V-filter, exactly method 2's split: t = -30 for the k rows whose given label TabPFN(V) finds
    least probable, 4 (w = 0.98) for the rest.
  - **Optimiser:** Adam at lr 0.01, 70 iterations.
  - **Deployed:** soft + V (P at its weights, V at weight 1) at a fixed iteration, 10 (`sigmoid`).
    - The development draws chose it: training at this lr changed neither the test ROC-AUC nor the balanced accuracy,
      and slowly raised the log loss (results/tables/vp_tune/findings.md).
    - A held-out choice did worst there and was dropped, so method 6 holds nothing out.
  - **Also recorded:** the last iteration (`sigmoid_last`) and the start (`sigmoid_start`).
- **m1 (method 7)** is bench_tinyv's attn_topk.
  - **Start:** the scores begin at the fitted V's V-filter log p, so the start's split is the V-filter's.
  - **Updates:** the gradient is taken at the relaxed point (probe 0.05); PGD-L-inf at lr 0.2, 400 iterations.
  - **Deployed:** hard + V (the k lowest-scored rows removed), at the last iteration (`m1_last`) and at the held-out
    choice (`m1_sel`). One run gives both.
- **The held-out part of V** (method 7 only):
  - **c = 3 and 5:** one row per class (gate_context.split_holdout). It is excluded from the start and from training,
    and added back at deployment.
  - **c = 1:** no row can be held out without losing a class, so m1_sel is the last iteration there.
- **baselines (method 8):** bench_tinyv's default configurations, trained on CPU as in the tuning.
  - **knn_shapley:** its own rule, removing the rows of negative KNN-Shapley value.
  - **ieg:** its top R% by score.
  - **Also recorded:** knn_shapley_R and ieg_own (the rows IEG relabels, removed).

## Methods 11-13 (added 2026-10-27, at the user's request)

**Method 11, greedy_thr (specified by the user):**
- As greedy (methods 4 and 5), but each step takes min(ceil(1% of |P|), the number of pool rows whose given label TabPFN
  with V + the accepted rows finds more probable than 0.5), most probable first.
- It stops at the first step where no row qualifies. R is not used.
- **Expected:** it stops near the agreement collapse that vp_tune found at about 50% (rand45) and 57% (hub30) of P kept.
  As a stopping rule, that point was the worst there (log loss 3.83).

**Methods 12 and 13, the repair methods:**
- **They never look at P's given labels.** P is treated as unlabelled, and TabPFN labels it from V.
- **12, vrepair (one shot):** TabPFN with V alone predicts every row of P. A subset chosen by confidence (max probability)
  joins V with the predicted labels.
- **13, grepair (iterative):** the trusted set starts as V. Every step, TabPFN with V + the rows labelled so far predicts
  the rest of P. The ceil(1% of |P|) rows it is most sure of then join with their predicted labels, which stay fixed.
- **Identical under both poisonings:** V and P's rows are the same under rand45 and hub30. The second instance of a pair
  copies the first's result (bench_vp.repair_twin) and recomputes only the poisoning-dependent statistics.
- **Free choices, settled by the development study below** (bench_vp.REPAIR; results/tables/vp_tune/findings_repair.md):
  - **12:** the rows with max probability > 0.9, the best of 15 rules. It is conservative: on the development draws it
    labelled no row in 25 of 36, and at most 20 outside gas-drift and students at c = 3 and 5.
  - **13:** global selection, stopped at 5% of P, the best of 40 stops x selections. The earliest stop recorded was
    best: every further step lowered ROC-AUC and raised log loss.
  - **12 at > 0.5 (vrepair_05), added at the user's request (2026-10-28):** recorded by the same job from the same
    predictions.
    - **Binary data:** a top probability is at least 0.5, so it labels all of P and equals vrepair_all on the 4 binary
      datasets.
    - **Multiclass data:** it labels a subset that swings with V. On students at c = 3 the five benchmark seeds give
      2-98% of P.
  - **Reference rows, recorded but not methods:** vrepair_top10 (the top 10% of P by confidence), vrepair_all (all of
    P labelled by TabPFN(V)), grepair_all (13 run until P is used up).
- **What the study found:** no repair rule beat V alone (ROC-AUC 0.673, log loss 0.80). Every row added raised log loss,
  because a context labelled by TabPFN(V) agrees with itself. TabPFN(V) is too unsure to label much of P (mean max
  probability 0.38-0.72), and along 13's path the batches' confidence reaches 1.0 within a few steps, so a threshold
  would not stop it.

**The development study (scripts/vp_tune_repair.py):**
- **Draws:** vp_tune's 36 development draws, (dataset, c, seed 100/101).
- **Units:** 72 = each draw x two selections for 13 (global, or split over the predicted classes in proportion). The
  global unit also scores 12's candidates:
  - the top 10-100% of P by confidence;
  - the top 10-70% of each predicted class;
  - confidence above 0.6-0.9.
- **13's records:** its path runs until P is used up, with test metrics every 5% of P, so that every stop can be scored.
- **Running:** scripts/vp_tune_repair_jobs/gpu_00-06.sh, one per GPU, about 11 min each on a free H100.
  - Copy over: results/vp_tune/draws, scripts/vp_tune_repair.py, scripts/vp_tune_repair_jobs/ and the changed
    scripts/bench_vp.py.
  - Results go to results/vp_tune/repair/. merge.sh brings them back.
  - `python scripts/vp_tune_repair.py report` writes results/tables/vp_tune/report_repair.md.
- **Status:** run on the user's GPUs (ecen-a86553 and -54), all 72 units, 57 GPU-minutes. CPU runs of the study and of
  method 11 had been started here first and were stopped before anything finished; their partial files were removed.

**The benchmark runs of methods 11-13 (2026-10-27):**
- **Scripts:** scripts/vp_jobs_11_13/gpu_00-11.sh, from `plan_vp.py --scripts 12 --out scripts/vp_jobs_11_13`.
  - 540 jobs (the 180 instances x greedy_thr, vrepair, grepair), about 15 min per script on a free H100.
  - Inference only: about 4 GiB of free GPU memory per script (gas-drift peaks at 2.6 GiB).
- **Twins:** a repair job and its other-poisoning twin are in the same script, rand45 first, so the second copies the
  first.
- **To copy over:** scripts/bench_vp.py and scripts/vp_jobs_11_13/. The rest is unchanged since the last runs.
- **Logs and results:** W&B group vp_bench; logs in logs/vp_jobs_11_13/; scripts/vp_jobs_11_13/merge.sh brings the
  results back.

## Intermediate records

These go to the result JSON and to W&B: one run per job, named `<job>_<instance>`, in project TFM_experiments, group
vp_bench, with the result files as an artifact.

- **greedy, every step:**
  - the next batch's statistics: TabPFN's probability of the given label, its agreement with the prediction, margin,
    max-probability and entropy;
  - the same statistics for the whole pool;
  - the true poisoned shares of the batch, of the pool and of what has been kept.
- **greedy, every 2 steps and at both stops:** the test metrics and the train-side V-score.
- **sigmoid and m1, every iteration up to 10, then every 5:**
  - the fitted V's log loss (the training objective; M1's at the relaxed point);
  - the held-out V's log loss, accuracy and ROC-AUC (m1 at c = 3 and 5);
  - the deployment's test metrics;
  - the weights' AUROC against the poison and the poisoned share of the weight. M1 also records its removal precision
    and recall and its swaps since the start.
- **The one-shot jobs** (refs, vfilter, baselines): their rows only.

## Running

```
bash scripts/vp_jobs/gpu_NN.sh [GPU]            # NN = 00..09, one script per GPU (a gas-drift attention job peaks at 46 GiB)
bash scripts/vp_jobs/merge.sh <host>:<repo path> # bring a machine's results back (locks and temporary files left out)
python scripts/bench_vp.py status
```

**Rerun (2026-10-27):** the first run left 84 jobs, all sigmoid or m1.
- 82 failed in both passes: out of GPU memory, on GPUs shared with other users' jobs. 2 were stopped by hand.
- `plan_vp.py --scripts 10 --out scripts/vp_jobs_rerun` plans only the jobs without a result. scripts/vp_jobs_rerun/gpu_00-09.sh
  hold these 84, about 2.7 H100-hours each; logs go to logs/vp_jobs_rerun/.
- Its inputs.sha256 is the first run's.

**Before running:**
- Copy over: results/bench_vp/draws, datasets/, and the preflight inputs (results/attn_form/preflight_reference,
  results/bench_largev/draws).
- Put WANDB_API_KEY in .env.

**What each script does:**
- It checks the machine first (scripts/attn_form_jobs/check_remote.py against inputs.sha256: the datasets, the 180
  draws and the code). It stops unless that check passes and W&B accepts the key.
- It then runs its list twice; the second pass retries failures.
- Finished jobs are skipped, so rerunning resumes.

**Cost:**
- 1,080 jobs, about 61 H100-hours, 6.1 per script (108 jobs each).
- The attention jobs dominate, m1 most. On gas-drift, sigmoid takes about 8 min and m1 about 40 min; on coil2000,
  about 5 and 28 min. Greedy takes under a minute and the one-shot jobs seconds.

**Results:** results/bench_vp/runs/<instance>/<job>.json (+ .npz: the V-filter's probabilities, greedy's acceptance
order, the baselines' scores, the attention weights at the start, at the deployed or selected iteration and at the last).

**Checked** (smoke runs on good_customer, gas-drift and coil2000, W&B offline):
- vfilter's removals recompute from its saved probabilities;
- greedy_R's kept set equals schedule_study's fixed:pct1;
- m1's start is the fitted V's V-filter log p, and at c = 1 its hard split equals vfilter_R;
- sigmoid's start removes exactly vfilter_R's rows;
- m1's held-out part is one row per class;
- records come at the planned iterations, and the sigmoid row is the record at iteration 10;
- each W&B run carries its per-step history, summary and artifact.

# Sifting with the clean V fixed on the train side (tiny-V benchmark)

Question (2026-10-23): with V = 1, 3 or 5 clean rows per class and R = 30% of the rest poisoned (hub-flip-local), how
do Meta-Sift and a TabPFN version of it do, if V is always trained on and never held out? Compared with the
benchmark's existing methods and baselines on the same 18 instances (6 datasets x 3 V sizes, seed 0; one run each).

Code: scripts/tinyv_sift.py (the methods), scripts/bench_tinyv.py (methods `metasift`, `tabsift` and `tabsift_m1`,
run_sift), scripts/plan_sift.py (the 11 scripts in scripts/sift_jobs/; the 5 tabsift_m1 scripts in
scripts/sift_m1_jobs/).

## The two methods

Both learn a weight per poisoned row from the same soft split: a row is trained on with strength w and held out with
strength 1 - w, and the split is the one under which the held-out rows are predicted worst (a model trained on clean
rows predicts flipped labels badly). V has weight 1 on the train side and is never held out.

**metasift**: Meta-Sift (Zeng et al., USENIX Security '23).
- **Their code:** the weighting network, gated meta-gradients, MetaSGD and per-batch weight normalisation are
  imported unedited from Meta-Sift/.
- **Backbone and dilution:** those of the earlier Meta-Sift baseline (scripts/run_meta_sift_baseline.py): a 4-layer
  MLP, Gaussian noise sigma 0.1 on z-scored features.
- **V in every batch:** their training loop is rewritten with each batch = 128 poisoned rows plus all of V. V has
  weight 1 (before their normalisation) in the virtual and the actual update, and is left out of the held-out loss
  the weighting network is trained on.
- **Scoring fix:** their test_sifter writes the last, smaller batch's weights to the wrong rows, so the scoring loop
  is rewritten too.
- **Learning rates:** the MLP by AdamW at 1e-3 (betas 0.9/0.95, weight decay 0.05), after one warmup epoch; the
  weighting network by SGD at 0.01; 5 epochs per sifter. The virtual (look-ahead) step also runs at 1e-3: their
  code loads the AdamW state into it, which replaces the nominal meta lr 0.1.
- **Ensemble:** 5 sifters, their weights averaged.

**tabsift**: the same split with TabPFN, through the row-weight mechanism of the attention methods
(tabfm_experiments/row_weights.py). Every poisoned row is in the context with weight w_j and is a query with loss
weight 1 - w_j; V is in the context at weight 1. Ascent on L(w) = mean_j (1 - w_j) CE_j over 100 full-gradient
steps, with the sigmoid study's form (results/tables/sigmoid_study):

- **Weights:** w = sigmoid(t), t unbounded (never clipped).
- **Start:** the hard V-filter split. The k rows the V-filter (TabPFN with V as the context) finds least plausible
  start at t = -30 (w about 1e-13), the rest at t = 4 (w = 0.982).
- **Update:** Adam at lr 5.
- **Earlier form:** param "w" (w in [0, 1] itself, clipped, PGD-L-inf at lr 0.05, start all 1) is still available
  (tinyv_sift.TABSIFT_W).
- **What the first steps do.** Adam's first steps move every t by about the learning rate, according to its
  gradient's sign. The ascent gradient is negative for most rows: raising a row's query weight 1 - w adds its loss.
  So one step at lr 5 drops many of the kept rows towards 0. On a development draw (good_customer, 1 row per class),
  rows below w = 0.5 went from 419 to 934 after step 1. After that the split settled at about 630 of 1,398 rows near
  0 and stayed there through step 100.

- **Self-mask only.** Each query is masked from its own copy in the context and nothing else is excluded. Without
  the mask TabPFN reads a row's label straight off the identical context row: flipping a row's own label moved its
  prediction by about 0.2 in probability unmasked, and by 0.003-0.024 masked (other rows still pass it on, like the
  one SGD step through which a sample influences its own held-out loss in Meta-Sift). `holdout="none"` drops the
  mask; `holdout="chunk"` holds whole chunks of queries out of the context.
- **No fixed total.** The total weight is free (`mass=True` would project onto sum(w) = (1 - R) n).
- **Queries in one pass.** All poisoned rows go through as queries in a single refit of the context; the chunk is
  halved only on running out of GPU memory, which leaves the objective unchanged. Per step this is several times
  cheaper than 512-row chunks (heloc: 7 against 45 seconds on a shared GPU).
- **Checked:** the gradient in t and in w against finite differences, within about 1%. The hard start is exactly
  the V-filter's k least plausible rows.

**tabsift_m1** (added 2026-10-24): the same split and objective in M1's form, the top-k attention weights of the
benchmark (tinyv_learned.ATTN_CONFIGS["topk"], attn_learn.py --param topk).

- **Scores:** a free score per poisoned row, started at the V-filter's log p(given label | V). The split is hard: the
  k = R% lowest-scored rows are held out (w = 0 in the context, query weight 1) and the rest are kept (w = 1, query
  weight 0). Step 0 is the V-filter's split exactly (checked on all 18 instances).
- **Gradient:** taken at M1's relaxed point: held-out rows at w = 0.05, kept rows at 0.95, query weights 1 - w. The
  scores move like log-weights: dL/dlog w_j = (the context part) - w_j CE_j / n.
- **Update:** PGD-L-inf ascent at M1's lr, s += 0.2 g / max|g|, for 100 steps. The scores are the deployed weights.
- **Differences from the sigmoid form.** Only the ranking matters, so a push shared by every score changes nothing,
  and a held-out row can come back. In the sigmoid form the held-out rows started at t = -30; they were not frozen (96%
  moved, mostly further down; corrected 2026-10-25), but only 0.6% got back above w = 0.5 within 100 steps.
- **Checked:** the gradient against central finite differences in log w:
  - kept rows: within 0.3%;
  - held-out rows: the objective is strongly curved around w = 0.05, and the differences converge to the gradient
    only at small steps (within 0.1% at eps = 0.002). At eps = 0.2 one row's difference even has the opposite sign,
    the probe misalignment the M1 study found.
- **Also recorded:**
  - the objective at the hard split itself: the held-out rows' loss, with only the kept rows and V as the context;
  - the number of rows that have left the start's held-out set.

**tabsift_adam_lr\<x\> and tabsift_sgd_lr\<x\>** (added 2026-10-25): an optimizer x learning-rate sweep of the sigmoid
form (hard V-filter start, 100 steps, records every 2). The default `tabsift` uses Adam at lr 5, which was set by hand,
never tuned for the sift objective. The sweep replaces the Adam-only `tabsift_lr<x>` plan, which never ran.
- **Grid:**
  - Adam at lr 0.01, 0.1, 0.5, 1, 5;
  - plain SGD (t += lr dL/dt, no momentum) at lr 1e3, 1e4, 5e4, 1e5, 5e5.
- **Why SGD's rates are large:** L is a mean over the n poisoned rows, so dL/dt is small. Measured at the hard start of
  the seed-0 draws, V = 1 and 5 per class, on every dataset but gas-drift (its backward pass needs about 36 GB, and the
  CPU run was cut off):
  - a kept row's median |dL/dt| is 1.4e-5 to 5.4e-5, except coil2000 at 5 per class: 4.9e-6. The 90th percentile is
    about 4 times the median, and the max is 4e-4 to 4e-3;
  - a held-out row's (t = -30) median is about 1e-14, with a few rows up to 1e-2.

  So SGD at Adam's rates would move a typical kept row by at most about 0.03 in 100 steps and change no deployment.
  SGD at 1e3 to 5e5 moves a typical kept row by about 0.014–0.05 (at 1e3) up to 7–27 (at 5e5) per step; on coil2000
  at 5 per class, 0.005 to 2.5. That is the range of Adam's steps, about lr per row per step. Matching Adam's median
  step takes SGD at about lr_adam / median |dL/dt|: 180–730 for Adam's 0.01 (2e3 on coil2000 at 5 per class), and
  9e4–3.7e5 for its 5 (1e6).
- **How the two differ:**
  - Adam's step is about lr for every row, whatever the gradient's size. The first steps are sign-like, so rows with
    gradients near 0 move as much as the others (the run-to-run noise below).
  - SGD's step is proportional to the gradient. Most held-out rows barely move, and the rows with the largest
    gradients jump: at 5e5 on a development draw, 774 rows were below w = 0.5 after the first step (417 at the start),
    and t reached ±5e4 by step 100.
- **Ranking and precision:** ranked by t (rank "t"). t is kept in float64 (t_float64), and checkpoints are saved in
  float64 (float64 weights are no longer cast to float32 on saving).
  - In float32, SGD's steps on most held-out rows (lr times about 1e-14) are below the resolution at t = -30 (about
    2e-6) and round away. On the development draw, 126–161 rows stayed tied at exactly -30 at lr 5e5. The top-70% cut
    fell among them, so file order decided which were removed.
  - The model's log w and query weights are float32 either way.
- **Runs:** `bash scripts/sift_lr_jobs/gpu_NN.sh [GPU]` (NN = 00..05), W&B group `tinyv_tabsift_opt_lr`.
  - 180 jobs, about 25.8 H100-hours, 4.3 per script.
  - One script per GPU: gas-drift's single pass needs about 45 GiB.
- **Built-in limit of Adam at small rates:** the deployment is a ranking, with no threshold. But the hard start ranks in
  two blocks: kept rows at t = 4, held-out rows at -30.
  - Small moves reorder rows within a block, and the top 70% is exactly the upper block. The top-70% set changes only
    once the lowest kept t and the highest held-out t meet.
  - The gap closes from both ends. Measured on the 18 instances (corrected 2026-10-26), it closes at about 2.4 x lr
    per step at lr 0.1 and below, and slower than lr per step above that:
    - lr 0.01: 31.2 at step 100;
    - lr 0.1: 9.6 at step 100, closed on none of the 18 instances (it should meet around step 140);
    - lr 0.5: met by about step 50;
    - lr 1: met by about step 25.
  - Rows swap only slowly after the blocks meet: by step 100, 2.7 rows per instance at lr 0.5 and 8.7 at lr 1.
  - So at lr 0.01 and 0.1 the top-70% deployment stays the V-filter's split for 100 steps. The per-class and 2%
    deployments, which cut inside a block, change from the first step.
- **Run-to-run noise at lr 5:** a rerun on another machine diverges from step 2 on (good_customer, 5 per class): 49 rows
  end more than 0.5 apart in w, and the removed sets share 367 of 417 rows. Adam's sign-like first steps flip rows whose
  gradient is near 0.
- **Results (2026-10-26):** results/tables/bench_tinyv/sift_opt_lr_findings.md, with the mean curves in
  results/plots/bench_tinyv/sift_opt_lr_curves.png.
  - SGD 1e4 is the best and steadiest rate. On the test metrics it ties Adam 5, the best Adam rate.
  - No rate improves detection over the V-filter start.

**tabsift_zero** (added 2026-10-25): the sigmoid form (Adam lr 5, 100 steps, records every 2) started with every
poisoned row near w = 0.
- **Start:** t = -30 (w about 1e-13) plus a N(0, 0.01) jitter (float64), so the step-0 split is random rather than in
  file order. V stays at weight 1. Every poisoned row starts as a query (query weight about 1): the whole poisoned set
  is held out, and the training side is V.
- **Why -30:** a development-draw check (good_customer and heloc at 5 per class, seed 1):
  - the gradients flow at every start from +4 to -30 (median |dL/dt| 6e-6 or more, far above Adam's eps);
  - starts from -4 to -20 leave TabPFN's predictions as far from a V-only context as w = 1 does (mean |dp| 0.24-0.33);
  - -30 is the closest (0.12-0.13).
- **Ranking:** the weights are recorded and deployed as t (rank "t"): the same order as w = sigmoid(t), without the
  ties float32 underflow creates (w is exactly 0 below t of about -104). The records' w_mean is the mean t.
- **V-selection:** step 0 is excluded, as for the all-ones start.
- **Runs:** none. The 100-step plan in `scripts/sift_zero_jobs/` never ran. The 300-step runs below replace it: Adam
  has no schedule, so their first 100 steps at lr 5 are this method. sift_zero_jobs' manifest predates later code
  changes, so its check would now stop it.

**tabsift_zero_lr0.01_it300, tabsift_zero_lr0.1_it300, tabsift_zero_lr5_it300, tabsift_adam_lr0.01_it300,
tabsift_adam_lr0.1_it300** (added 2026-10-26; lr 0.01 added the same day): 300 steps, recorded every 2.
- **Runs:** the zero start above at Adam lr 0.01, 0.1 and 5, and the hard V-filter start at Adam lr 0.01 and 0.1 (t
  in float64, ranked by t).
- **Checkpoints:** steps 0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250 and 300. The default row is step 300.
- **The objective at every step:** tabsift's `trace` keeps every step's objective, which is computed at every step
  anyway. It goes to the result JSON's `objective_trace` and to W&B `objective/L`. The test metrics, precision and
  purity are still recorded every 2 steps. This applies to every tabsift method from now on.
- **Hard start:** the gap of 34 between the blocks closes from both ends. Measured in the 100-step sweep:
  - at lr 0.1, about 0.24 per step (9.6 left at step 100), so the blocks should meet around step 140 and the top-70%
    set starts changing after that, slowly at first;
  - at lr 0.01, about 0.028 per step, about 8 in 300 steps. The top-70% set stays the V-filter's split; on the
    development draw it did.

  The per-class and 2% deployments change from the first step at both rates.
- **Zero start at lr 0.01:** the rows start within a 0.01 jitter of each other, so the ranking changes from step 1.
  The weights stay near 0, though: t moves by at most a few units in 300 steps, so w stays below about 1e-9, and the
  context stays essentially V alone.
- **Runs:** `bash scripts/sift_it300_jobs/gpu_NN.sh [GPU]` (NN = 00..04), W&B group `tinyv_tabsift_it300`.
  - 90 jobs (5 methods x 18 instances), about 35.4 H100-hours, 7.1 per script. Each script runs one method on all 18
    instances (3 gas-drift and 3 coil2000 jobs each).
  - One script per GPU. A gas-drift job takes about an hour on a free H100.
- **Smoke test,** on a development draw (good_customer, 5 per class, seed 1), all 5 methods: about 65 s per job, 151
  records, and a 301-point objective trace equal to the records' objective. W&B had objective/L at all 301 steps.

**tabsift_m1_rand_s0..s4 and tabsift_m1_rand_ens** (added 2026-10-24): tabsift_m1 from a random start.
- **Start:** random N(0, 1) scores drawn with seed i (attn_learn's random topk start) instead of the V-filter's log p.
  V stays in the context at weight 1.
- **Training:** 300 steps, each seed recorded every 5 steps (its own trajectory, like tabsift_m1's).
- **V-selection candidates:** steps 0, 10, 20, 50, 100, 150, 200, 250 and 300 (their scores are saved).
- **The ensemble** (`tabsift_m1_rand_ens`; bench_tinyv.run_sift_ensemble) runs once after its 5 seeds finish. Each
  row's weight is its hard weight (1 kept, 0 held out) averaged over the seeds' chosen checkpoints; ties are broken by
  the row's mean score rank. It has no trajectory.
  - The default rows vote over the seeds' last steps.
  - The `_vsel` rows vote over each seed's V-selected checkpoint for that deployment.
  - Each row also gives the share of rows on which all 5 seeds agree.
- **Run-to-run noise:** two runs of the same seed are bit-identical for about 185 steps. Then GPU rounding flips a
  near-tied swap, and by step 300 their held-out sets overlap on about 70% of the rows (good_customer development
  draw: ROC-AUC 0.590 vs 0.607).
- **Runs:** `bash scripts/sift_m1_rand_jobs/gpu_NN.sh [GPU]` (NN = 00..06, one script per GPU), then `merge.sh`. Each
  instance's 5 seeds and their vote are in the same script, so the scripts need no shared disk.
- **W&B:** group `tinyv_tabsift_m1_rand_v2`. Every run has its result files as an artifact.
  The stopped first version (the vote at every step; seeds 1-4 recorded every 25) left runs in group
  `tinyv_tabsift_m1_rand`. Its result files are in results/bench_tinyv/superseded/sift_m1_rand_v1/.
- **Cost:** 108 jobs, about 33.6 H100-hours, 4.6-5.0 per script (7 scripts).

Every sift trajectory now also records the AUROC of the weights against the poisoned rows (a recording, never used).

**W&B** (`--wandb-mode online`; the tabsift_m1 scripts use it): each sift job is one run in project TFM_experiments.
- **Naming:** named `<method>_<instance>` (e.g. `tabsift_m1_heloc_R30_c5_s0`), group `tinyv_<method>`, tags method,
  dataset, c and R. The config holds the instance and the configuration used.
- **Per record** (x = step):
  - `accuracy/`, `balanced_accuracy/`, `roc_auc/`, `log_loss/`, `precision/`, `purity/` and `n_removed/`, each with
    `full`, `full_cb` and `small`;
  - `objective/L` and `objective/L_hard`;
  - `ranking/auroc_poison` and `ranking/swapped_since_start`;
  - `weights/mean`.
- **At the end:**
  - the summary: `final/<row>/<metric>` for the default and V-selected rows, and `refs/<poisoned_plus_v | oracle_plus_v |
    v_only>/<metric>`;
  - charts overlaying the three deployments with the references as flat lines: `overlay/<metric>`,
    `overlay/precision` and `overlay/objective`;
  - the tables `tables/rows` and `tables/v_selection`.
- **Local files:** the JSON and NPZ results are written as before.

## Deployments, rows and recording

Each set of weights is deployed three ways, V always added:

| deployment | row suffix | kept rows |
|---|---|---|
| top (1 - R)% overall | (none) | the k lowest-weighted rows removed, as V-filter and M1 do |
| top (1 - R)% class by class | `_cb` | each given class keeps its share of the poisoned rows (Meta-Sift's rule, get_sifter_result) |
| small clean subset | `_small` | the top 2% by that rule |

Rows in results/bench_tinyv/runs/<instance>/{metasift,tabsift}.json, for each deployment:

| row | configuration |
|---|---|
| metasift | the earlier Meta-Sift baseline's settings (above) |
| metasift_vsel | V-selected among those settings and 3 variants (model lr 3e-4, 10 epochs, v_lr 0.05) |
| tabsift | sigmoid, hard V-filter start, Adam lr 5, 100 steps (the default) |
| tabsift_vsel | V-selected among its steps 0 (the hard V-filter split itself), 1, 2, 5, 10, 20, 30, 50, 75 and 100 |
| tabsift_m1 | M1's form: hard top-k split, gradient at the relaxed point 0.05, PGD-L-inf lr 0.2, 100 steps |
| tabsift_m1_vsel | V-selected among the same steps (step 0 is the V-filter's split) |

V-selection is the benchmark's rule for baselines: the candidate whose kept rows alone, as TabPFN's context, give V
the lowest log loss. It uses only V and the poisoned rows.

**Trajectory** (the JSON's `trajectory`), recorded while training:

- tabsift and tabsift_m1: every 2 steps;
- metasift: every 10 meta-iterations of each sifter (and after its warmup), default settings, with the sifter's
  current weights; plus the final ensemble.

Each record holds the objective (metasift: the held-out loss it raises, and its training loss), the mean weight and,
per deployment, the test metrics, the precision of the removed rows and the purity of the kept ones. These are
recordings only: no choice is made on them.

Development (seed-1 draws, never the benchmark's seed-0 instances) chose nothing with the test set. The earlier
no-V sift (results/sift) ranked the poison best after one step and decayed toward chance by step 20; the trajectories
show whether 100 steps with V behave the same.

## Running

```
bash scripts/sift_jobs/gpu_NN.sh [GPU]                # NN = 00..10; checks the machine first, resumes when rerun
bash scripts/sift_jobs/merge.sh <host>:<repo path>    # bring the runs back
python scripts/plot_bench_tinyv.py                    # -> results/plots/bench_tinyv/R30_V*.png, results/tables/bench_tinyv/R30_by_v.md
```

The other machines need these copied over:
- results/bench_tinyv/draws;
- results/attn_form/preflight_reference and results/bench_largev/draws, for the check;
- datasets/;
- the Meta-Sift/ folder. It is not a working git submodule: a clone or pull leaves it empty.

**Cost**: 36 jobs, about 4.3 H100-hours, about 0.4 hours on each of the 11 scripts. The longest jobs are tabsift on
gas-drift: about 20 minutes each, with about 45 GiB of GPU memory or more for the single pass (it falls back to
smaller chunks). On a GPU that another busy job shares, expect two to four times longer.

**tabsift_m1**:
- **Run:** `bash scripts/sift_m1_jobs/gpu_NN.sh [GPU]` (NN = 00..04), each logging to W&B online, then
  `bash scripts/sift_m1_jobs/merge.sh <host>:<repo path>`.
- **Cost:** 18 jobs, about 2.9 H100-hours, at most 0.6 h per script.
- **Memory:** two gas-drift jobs on one GPU run out of memory. Give each its own GPU.

Results (2026-10-24):
- results/tables/bench_tinyv/sift_findings.md (Meta-Sift and the sigmoid form), sift_m1_findings.md (M1's form),
  sift_m1_rand_findings.md (M1's form from random starts, and the 5-seed vote) and sift_opt_lr_findings.md (the
  sigmoid form's optimizer x learning-rate sweep, 2026-10-26);
- R30_by_v.md (per dataset);
- results/plots/bench_tinyv/R30_V{1,3,5}.png.

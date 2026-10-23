# Sifting with the clean V fixed on the train side (tiny-V benchmark)

Question (2026-10-23): with V = 1, 3 or 5 clean rows per class and R = 30% of the rest poisoned (hub-flip-local), how
do Meta-Sift and a TabPFN version of it do, if V is always trained on and never held out? Compared with the
benchmark's existing methods and baselines on the same 18 instances (6 datasets x 3 V sizes, seed 0; one run each).

Code: scripts/tinyv_sift.py (the methods), scripts/bench_tinyv.py (methods `metasift` and `tabsift`, run_sift),
scripts/plan_sift.py (the 9 scripts in scripts/sift_jobs/).

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
weight 1 - w_j; V is in the context at weight 1. PGD-L-inf ascent on L(w) = mean_j (1 - w_j) CE_j, w in [0, 1]
(clipped), lr 0.05, 100 full-gradient steps. Two starts: all 1 (the default, the Meta-Sift analogue: trust
everything first) and the soft V-filter weights p(given label | V) / max.

- **Self-mask only.** Each query is masked from its own copy in the context and nothing else is excluded. Without
  the mask TabPFN reads a row's label straight off the identical context row: flipping a row's own label moved its
  prediction by about 0.2 in probability unmasked, and by 0.003-0.024 masked (other rows still pass it on, like the
  one SGD step through which a sample influences its own held-out loss in Meta-Sift). `holdout="none"` drops the
  mask; `holdout="chunk"` holds whole chunks of queries out of the context.
- **No fixed total.** The total weight is free (`mass=True` would project onto sum(w) = (1 - R) n).
- **Queries in one pass.** All poisoned rows go through as queries in a single refit of the context; the chunk is
  halved only on running out of GPU memory, which leaves the objective unchanged. Per step this is several times
  cheaper than 512-row chunks (heloc: 7 against 45 seconds on a shared GPU).
- **Checked:** the gradient against finite differences, within about 1%.

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
| tabsift | all-1 start, 100 steps (the default) |
| tabsift_vfinit | V-filter start, 100 steps |
| tabsift_vsel | V-selected among both starts at steps 1, 2, 5, 10, 20, 30, 50, 75, 100 (and the V-filter start itself) |

V-selection is the benchmark's rule for baselines: the candidate whose kept rows alone, as TabPFN's context, give V
the lowest log loss. It uses only V and the poisoned rows.

**Trajectory** (the JSON's `trajectory`), recorded while training:

- tabsift: every 2 steps, both starts;
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
bash scripts/sift_jobs/gpu_0N.sh [GPU]                # N = 0..8; checks the machine first, resumes when rerun
bash scripts/sift_jobs/merge.sh <host>:<repo path>    # bring the runs back
python scripts/plot_bench_tinyv.py                    # -> results/plots/bench_tinyv/R30_V*.png, results/tables/bench_tinyv/R30_by_v.md
```

The other machines need these copied over:
- results/bench_tinyv/draws;
- results/attn_form/preflight_reference and results/bench_largev/draws, for the check;
- datasets/;
- the Meta-Sift/ folder. It is not a working git submodule: a clone or pull leaves it empty.

**Cost**: 36 jobs, about 6.6 H100-hours, about 0.75 hours on each of the 9 scripts. The longest jobs are tabsift on
gas-drift: about 40 minutes each, with about 45 GiB of GPU memory or more for the single pass (it falls back to
smaller chunks). On a GPU that another busy job shares, expect two to four times longer.

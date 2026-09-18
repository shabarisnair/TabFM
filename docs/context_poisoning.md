# TabPFNv2 context poisoning

This document describes the context-poisoning experiments on TabPFNv2: the threat model, how the
model encodes its prompt, the two attacks (X_train CAPGD and influence-ranked Y_train label flips),
the validation-selected contexts, the output files, and copy-paste commands.

Code: `scripts/tabfm_experiments/` (package), thin CLIs `scripts/infer.py`,
`scripts/attack_context.py`, `scripts/select_contexts.py`. Tests: `tests/tabfm_experiments/`.

Run everything in the `tabfm` conda env (`conda run -n tabfm ...`). `pytest` and `requests` (needed
by TabularBench's dataset modules) were installed into it. TabularBench is used from the local
checkout; `config.ensure_tabularbench_importable()` puts it on `sys.path` and restores the
`np.float_` alias that NumPy 2 removed.

---

## 1. Threat model: transductive white-box upper bound

The attacker has white-box access to TabPFNv2 and can read `X_test`, `y_test` from
`test_attack_1000.csv`. They may modify a budgeted subset of the in-context training set (the
prompt): features of R% of the context rows (`x-capgd`) or labels of R% of the rows
(`label-flip`). They are **scored on the same test rows they optimised against**.

This is a **transductive upper bound**, not held-out generalisation. Numbers from these runs
must not be read as "a poisoned context degrades unseen queries by X".

Djilani et al. (arXiv:2506.02978, SaTML 2026) perturb the **test** input with the context held
clean. We attack the **context**. The datasets and the CAPGD optimiser family are the same; we
work on the opposite side of the prompt.

## 2. How TabPFNv2 encodes the prompt

Features (train and test rows together) go through:

```
X -> NaN/Inf flags -> impute (train means) -> StandardScaler (train stats)
  -> constant-feature removal / feature-group normalisation -> Linear(feature_group_embedder)
```

Train labels go through:

```
Y_train -> NaN-pad test rows -> impute -> _flatten_multiclass_targets -> Linear(target_embedder)
```

`_flatten_multiclass_targets` computes `(y > unique_ys).sum()`. The derivative of `>` is zero
everywhere, so **∂L/∂Y_train ≡ 0**, even though `target_embedder` is linear. That is why there is
no PGD on Y for v2 (checked in `test_runtime_cpu.py::test_grad_wrt_x_nonzero_and_y_has_no_useful_grad`).
Test labels are not an input: test target cells are NaN, and the output head reads the
target-column hidden state of the test rows.

**Categorical X is still PGD-able.** In these CSVs, `cat` features are already integer codes, and
TabularBench's CAPGD does not one-hot them either. During the steps they are treated as continuous
in scaled space. Types are repaired **once, at the end** (`fix_types`: int features truncate the
*delta* with `torch.fix`, cat features round the *value*). There is no per-step rounding; rounding
every step snaps sub-unit steps back and stalls the attack. The fixed differentiable pipeline
declares every column numerical (`categorical_features_indices` stays empty), which is required
by `reject_categoricals_for_differentiable_input`.

### Shared evaluation path (no stat freezing)

`runtime.evaluate_context` is the only way a context gets scored: clean inference, every CAPGD
step, influence scoring, the final poisoned metrics and context selection all go through it. Each
call runs `fit_with_differentiable_input` on the **current full context** (poisoned rows
included), then `forward(X_test, use_inference_mode=True)`. Scaler, imputation means and constant
masks are therefore those of the poisoned prompt. The old `StatFreezer` design in the previous
`attack_context.py` is gone.

Model construction is fixed (`runtime.build_tabpfn_v2`): TabPFNv2 via
`create_default_for_version(ModelVersion.V2)`, `n_estimators=1`, float32, no fingerprint feature,
no feature/class shifts, no polynomial features, no outlier removal, no row subsampling,
`PreprocessorConfig("none", categorical_name="numeric")`, and `n_classes_ = 2`.

**Test chunking (`--test-batch-size`) is exact in almost all cases, but not always.** Test rows do
not attend to each other, and imputation means and standard-scaler stats are fitted on the context
only. But TabPFNv2 computes its **constant-feature masks over all rows of the call** (context plus
the test chunk). If a feature is constant across the context and a test row has a different value,
the mask depends on which test rows share the chunk. `runtime.chunking_is_exact(X_ctx, X_test)`
detects this. `evaluate_context` warns when chunking is inexact, and the CLIs log
`chunking_exact`. When it holds, chunked and unchunked losses agree to float precision; gradients
agree to float32 noise (≈1e-5), because every chunk re-runs the forward and backward.

**One classifier per feature width.** `fit_with_differentiable_input` caches
`inferred_feature_schema_` and `ensemble_configs_` on its first call and silently reuses them. A
wider table then crashes deep inside preprocessing; a narrower one runs on a stale schema (its
outputs happened to be identical in our checks, but this is not guaranteed).
`runtime.check_feature_width` raises instead, and `select_contexts.py` builds one classifier per
dataset.

**Memory.** With `differentiable_input=True`, TabPFN never enters inference mode, and its engine
then hard-codes `save_peak_mem = False`. Peak memory on WiDS (108 features) grew about linearly
with context size: 26 GiB at 2k rows, 64 GiB at 5k, OOM on a 93 GiB card at 10k, regardless of
validation chunking or `memory_saving_mode`. For **gradient-free** calls, `runtime._forward`
therefore sets `differentiable_input = False` for the forward only. The fitted executor,
preprocessing and model are the same, but TabPFN's memory-saving inference path is used.
Probabilities are identical (max |Δp| = 0.0 at WiDS 2k/250 and in
`test_nograd_memory_saving_forward_matches_differentiable_forward`), and a 10k WiDS context scores
in ~4 GiB. **Gradient calls are unchanged and still need the full memory:** a WiDS context of 1000
rows with 16 test rows peaked at 13 GiB. X-CAPGD on large WiDS contexts needs a small
`--test-batch-size`, and the largest contexts may not fit on one card.

## 3. Determinism

**best-effort (default).** Fixed seeds, `n_estimators=1`, the fixed inference config above,
float32, `cudnn.benchmark=False`. FlashAttention and other fast kernels are allowed. Every run logs
`clean_max_repeat_abs_dprob`, the max |Δp| over 3 identical clean forwards. (It was 0.0 in the
url_unique smoke runs on this machine.)

**strict (`--deterministic-mode strict`).** Also sets `torch.use_deterministic_algorithms(True)`
(raises on ops without a deterministic kernel), `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`cudnn.deterministic=True`, and disables the flash and memory-efficient SDP backends.

Neither mode is bitwise reproducible across GPU models, drivers or library versions.

## 4. Attack 1: X_train CAPGD (`--attack x-capgd`)

The optimiser is `tabfm_experiments/capgd.py::capgd`, a port of TabularBench
`CAPGD.attack_single_run`. `scripts/check_capgd_equiv.py` checks it iterate-for-iterate against
the reference file on a non-linear toy objective for Linf and L2 at several step counts and eps
values (`ALL MATCH`). `tests/tabfm_experiments/test_capgd.py` re-runs part of that comparison.

**Objective.** Maximise the mean test CE (`F.nll_loss(log(probs.clamp_min(1e-12)), y)`) over the
scaled features of the attacked context rows. The attacked rows' labels stay in the context and
are not part of the loss.

**Space and budget.** Min-max scaling to [0, 1] uses metadata `min`/`max` (`--scaler metadata`).
Constant features get range 1. `--scaler train` uses the current context's min/max instead, which
couples the box to the clean batch. The budget is **per attacked row** in scaled units:
`eps' = eps · (1 − eps_margin)`. Untouched rows are immutable for the trial.

**Constraint modes (`--constraints`):**

| mode | in the loop | at the end |
|---|---|---|
| `none` | eps-ball only; every feature mutable | nothing |
| `box` | eps-ball, [0,1] box, metadata mutable mask | restore immutable cells |
| `full` (default) | box, mutable mask, relation penalty, in-loop equality fixing (`--fix-equality-iter`, default on) | `fix_types → fix_immutable → fix_equality_constraints`, then **budget re-projection** (default) |

**Budget-aware end repair (default: `--budget-repair`).** The standard end repair
(`fix_types → fix_immutable → fix_equality_constraints`) runs after the last eps-ball projection and is
not budget-aware, so rounding a categorical or recomputing an equality-linked feature (LCLD
`installment`, ratios) can push a row back outside the ε-ball. By default, `full` mode now follows the
standard repair with `repair_end_budgeted`: it radially shrinks any over-budget row's scaled
perturbation back onto the ball and re-rounds, iterating a few times, ending on a type/immutable
repair. The reported row is therefore integer-valid, immutable-respecting, and within budget
(`within_eps_after_repair` is `True`), at the cost of exact equality (budget and equality cannot both
hold with integer features). `--no-budget-repair` reproduces the TabularBench behaviour (repair may
leave the ball). Each trial also records the plain repair under `extra["standard_repair"]` for
comparison. Measured CE cost of enforcing the budget (ΔCE_standard − ΔCE_budgeted, same iterate):
url_unique 0.000 (never left the ball); WiDS ≤ 0.001; LCLD 0.000–0.019 at 100 iters, up to ~0.165 at
400 iters — i.e. much of LCLD's high-iteration gain comes from over-budget rows.

In `full` mode the objective is `mean_test_CE − λ · mean_over_attacked_rows(violation)`, where
`λ = --constraint-penalty` (default 1.0). `violation` is the TabularBench `ConstraintsExecutor`
output for `AndConstraint(relations)` on the unscaled rows (per row, 0 = satisfied, larger = more
violated). This matches `loss_indiv − executor.execute(inverse_transform(x_adv))` in the reference.
NaN or inf penalty values and gradients (e.g. divisions by zero) are zeroed.

> **Observed: the raw-unit penalty can dominate.** Violations are in raw units (character counts,
> dollars), while the mean test CE and its per-cell gradients are small. On the old url_unique
> `splits/context_1000.csv` (64 test rows, k=50, 10 steps), `box` raised test CE by +0.091, but
> `full` with λ=1 halved its step size back to the clean point (ΔCE 0). This is faithful to the
> reference loss. A single-trial λ sweep on `selected/natural/context_1000.csv` (64 test rows, k=50,
> 10 steps) gave:
>
> | λ | ΔCE | relation violation after repair (mean) | violated rows |
> |---|---|---|---|
> | 1 | +0.012 | 0.00 | 0 |
> | 0.1 | +0.059 | 0.28 | 1 |
> | 0.01 | +0.050 | 0.00 | 0 |
> | 0.001 | +0.068 | 0.78 | 2 |
> | 0 | +0.064 | 0.84 | 2 |
>
> These are single noisy trials, so treat the numbers as indicative. A WiDS smoke at λ=1 also stayed
> at the clean point. Pick λ deliberately before running grids. The per-trial JSON records
> `relation_violation_clean_mean`, `relation_violation_final_mean` and
> `relation_violated_rows_final`.

> **Observed: without budget re-projection, repair leaves the eps-ball.** With `--no-budget-repair`,
> equality fixing recomputes dependent features (LCLD `installment`, ratios) and categorical rounding
> can add up to 0.5 raw units per feature, so repaired LCLD rows reached 1.5–3.5 scaled-L2 from clean
> against eps' = 0.475 (`within_eps_after_repair` was `False` in 9 of 10 LCLD `full` runs). The default
> `--budget-repair` fixes this by re-projecting (see above); `within_eps_after_repair` is then `True`.
> Each trial records `within_eps_after_repair`, `max_l2_delta_scaled`, `max_linf_delta_scaled`.

**Restarts and seeds.** Run 0 starts at the clean point. Runs r > 0, restarts j > 0, or
`--random-start` use the reference random start inside the eps-ball, drawn from
`torch.Generator(seed = (attack_seed + r) · 1000 + j)`. Each restart is repaired and re-scored, and
the restart with the highest repaired CE is kept.

**Defaults vs the paper:**

| arg | default | source |
|---|---|---|
| `--norm` | `l2` | TabularBench benchmark example (the paper uses both) |
| `--eps` | 0.5 | scaled units |
| `--eps-margin` | 0.05 | inherited from TabularBench code; **not stated in the Djilani text** |
| `--n-iter` | 40 | this project, from the n_iter sweep below (TabularBench uses 10) |
| `--momentum` / `--rho` | 0.75 / 0.75 | APGD |
| `--n-restarts` | 1 | TabularBench |
| `--eot-iter` | 1 | TabularBench |
| `--n-runs` | 5 | paper: five attack runs (clean start, then random starts) |
| `--n-row-subsamples` | 5 | this project |
| `--row-percent` | 5 | this project |

### n_iter ablation (why the default is 40)

**Setup (2026-09-15).**
- Context `selected/natural/context_5000.csv`, test `test_attack_1000.csv` (all 1000 rows).
- `--row-percent 30` (k = 1500 rows). One trial: run 0 (clean start), subsample 0, `row_seed` 1.
- Every other argument at its default (L2, eps 0.5, λ = 1, equality fixing on).
- Each n_iter is a separate run, because CAPGD's step-halving schedule scales with n_iter.
- LCLD `full` was repeated on a second row subset (`--row-seed 2`).
- Outputs, logs and the sweep script are in `results/ablation_niter/`.

ΔCE (poisoned − clean test CE) after repair:

| dataset / mode | 10 | 20 | 30 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|
| url_unique, `full` (default) | +0.000 | +0.024 | +0.020 | +0.028 | +0.029 | +0.033 |
| url_unique, `box` | +0.045 | +0.117 | +0.205 | +0.402 | +0.655 | +5.972 |
| lcld_v2, `full` (default), row_seed 1 | +0.016 | +0.028 | +0.024 | +0.023 | +0.046 | +0.071 |
| lcld_v2, `full` (default), row_seed 2 | +0.009 | +0.044 | – | +0.056 | – | +0.144 |
| lcld_v2, `box` | +0.019 | +0.125 | +0.077 | +0.157 | +2.309 | +1.367 |

What the sweep showed:
- **No saturation by 100 in the default mode on LCLD.** ΔCE keeps rising on both row subsets. The `box` runs keep rising too.
- **url_unique `full` is roughly flat from 20 iterations (+0.020 to +0.033) because of repair.** Its only repair is `fix_types`, and int truncation removes most of what CAPGD finds. At 100 iterations the objective before repair was 3.79; test CE after repair is 0.126.
- **Gains arrive late and single trials are noisy.** Within most runs the best objective stays flat for most of the steps and then jumps in the last 10–25. Separate runs are not monotone in n_iter (LCLD `box`: +2.31 at 75 but +1.37 at 100).
- **`box` numbers are type-invalid.** On LCLD, every one of the 1500 attacked rows ended with fractional int values (~40% of int cells) and fractional cat values (~40% of cat cells). `full` had none. Don't report `box` as a realistic attack.
- **LCLD `full` rows often leave the budget.** `within_eps_after_repair` was `False` in 9 of 10 LCLD `full` runs (max scaled L2 up to 3.45), due to equality and type repair.

(The table above is the first, exploratory sweep: single trial per point, `box` included, and
run *before* budget-aware repair was the default. It is superseded by the sweep below.)

#### Final sweep (`results/niter_sweep_final/`)

Setup: 3 datasets, `selected/natural/context_5000.csv` vs `test_attack_1000.csv` (all 1000 rows),
`--row-percent 30`, **3 row seeds per point** (mean reported), best-effort, `--constraints full`
with budget-aware repair on, iteration points 10/25/50/75/100/150/200.

| dataset | metric | 10 | 25 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| url_unique | ΔCE | +0.003 | +0.022 | +0.036 | +0.042 | +0.035 | +0.043 | +0.050 |
| url_unique | Δacc | +0.002 | −0.000 | −0.006 | −0.008 | −0.004 | −0.010 | −0.007 |
| lcld_v2 | ΔCE | +0.020 | +0.021 | +0.027 | +0.075 | +0.072 | +0.284 | +0.348 |
| lcld_v2 | Δacc | −0.002 | −0.002 | −0.003 | −0.014 | −0.011 | −0.020 | −0.015 |
| wids | ΔCE | +0.041 | +0.089 | +0.163 | +0.321 | +0.317 | +0.491 | +1.544 |
| wids | Δacc | −0.012 | −0.020 | −0.033 | −0.081 | −0.099 | −0.113 | −0.319 |

What it shows:
- **Accuracy is far more robust than CE.** url_unique CE rises 1.54× for only 0.7 accuracy points;
  lcld_v2 1.77× for 1.5 points. The attack mostly reduces confidence on rows still classified
  correctly. Only WiDS shows real accuracy damage.
- **Vulnerability ranking is stable at every iteration count: wids ≫ lcld_v2 > url_unique.**
- **No saturation by 200** on lcld_v2 or wids; url_unique plateaus around 50–75 at a low level
  (repair-bound — int rounding erases most of what CAPGD finds).
- **Variance is large and grows in absolute terms.** wids at 200 is `[+0.240, +0.220, +4.173]` —
  the 8.4× headline is one outlier seed. Relative spread (spread/mean) is ~0.3–1.4× at *every*
  count from 10 to 75, so more iterations do not buy precision.
- Caveat: the wids 10/25/50/75 cells were produced before the budget-repair convergence fix and
  include a few over-budget rows (5 of 63 runs), so they are mildly optimistic. wids 100/150/200
  were fully within budget.

**Decision: `--n-iter 40`.** Interpolating the sweep, 40 gives ΔCE ≈ +0.030 on url_unique (10× the
value at 10 iterations) and ≈ +0.133 on wids (3.2×), i.e. most of the reachable signal, at **2.5×
less cost than 100**. It does *not* reduce relative variance — that is inherent to best-effort plus
CAPGD's chaotic step-halving. **Caveat:** lcld_v2 is flat from 10 to 50 (+0.020 → +0.027) and only
jumps at 75, so lcld_v2 sits near its floor at 40; raise `--n-iter` if lcld_v2 is the focus.

**Cost.** Per-iteration cost scales almost linearly with the number of feature groups
(features / 2), measured at context_5000 + 1000 test rows, best-effort:

| dataset | features | feature groups | s/iteration | s/trial @ 40 iters |
|---|---|---|---|---|
| lcld_v2 | 28 | 15 | 1.29 | 52 |
| url_unique | 63 | 32 | 2.31 | 92 |
| wids | 108 | 55 | 4.88 | 195 |

WiDS is ~3.8× lcld_v2 per iteration for 3.7× the feature groups — it is slow simply because the
table is wide, and its figure includes the ~25% `--recompute-layers` overhead (3.9 s/iter without).
Cost is **independent of `--row-percent`**: every iteration is a full forward+backward over the
entire context + test set regardless of how many rows are attacked. The default x-capgd grid is
5 runs × 3 subsamples = 15 trials per configuration.

Adaptive step-size checkpoints are `steps_2 = max(int(0.22·n),1)`, `steps_min = max(int(0.06·n),1)`
and `size_decr = max(int(0.03·n),1)`. Linf and L2 differ in three ways: the step is
`sign(grad)` vs the row-normalised grad; the mutable mask applies to the momentum terms only under
L2; and L2 has the reference's 1e-12 pad asymmetry between its two projections.

### Constraints per dataset

The relation factories are imported from TabularBench, not re-written:

| dataset | factory | feature references |
|---|---|---|
| url_unique | `tabularbench.datasets.samples.url.get_relation_constraints()` (14 relations) | `Feature(i)` = 0-based column index of X **after dropping the target**, metadata order (same schema as `url`) |
| lcld_v2 | `samples.lcld.get_relation_constraints()` (9 relations) | feature **names** (`installment`, `open_acc <= total_acc`, ratio equalities, …); `issue_d` is dropped and no constraint references it |
| wids | `samples.wids.get_relation_constraints(metadata)` (31 relations) | `Feature(i+1) <= Feature(i)` for i in 33, 35, …, 93 (the `*_max` / `*_min` pairs) on X after dropping `hospital_death` |

Index-based constraints only make sense if the CSV column order equals the metadata order.
`data.read_metadata` raises if it does not. Only LCLD has fixable equality constraints (an
`EqualConstraint` with a `Feature` on the left), so in-loop and end-of-attack equality fixing are
no-ops for url_unique and wids.

`fix_types` reproduces the reference's early return: if a dataset had no `int` features, `cat`
features would not be rounded. All three datasets have int features, so cats are rounded.

## 5. Attack 2: influence-ranked label flips (`--attack label-flip`)

This is not CAPGD or PGD.

1. The eligible rows are all rows, or rows with `y == --attack-class`. `k = max(1, round(n·R/100))`
   (half-up rounding); if fewer rows are eligible, all of them are used.
2. One forward and backward of the mean test CE w.r.t. the **target embeddings** of the train rows.
   `runtime.target_embedding_influence_scores` wraps `TabPFNV2._embed_targets` on the loaded model
   instance and adds a zero tensor `delta` to the train rows' embeddings, so `g_i = ∂L/∂delta_i`.
   The override is removed afterwards.
3. For binary labels densified to {0, 1}, flipping row i moves its embedding by
   `(1 − 2·yflat_i) · W[:, 0]` (`W = target_embedder.weight`; the NaN-indicator input is 0 for
   observed labels). The score is `s_i = ⟨g_i, (1 − 2·yflat_i) · W[:, 0]⟩`, the first-order change
   in CE.
4. Flip the k eligible rows with the largest `s_i`, keeping X unchanged, and re-score the integer
   flipped context on the normal path.

The ranking is deterministic and covers every eligible row, so repeated trials flip the same set.
The defaults are therefore `--n-runs 1 --n-row-subsamples 1`. Larger values only re-measure GPU
noise.

> **Limitation (observed in the unit-test toys):** the embedding gradient is correct (it matches
> central finite differences, `test_influence_scores_match_finite_differences`), but **first-order
> influence is a weak predictor of the effect of a discrete flip on TabPFNv2**. On small 2-D toys,
> Spearman correlation between `s_i` and the true CE change after flipping row i ranged from −0.16
> to 0.28. On a toy with one isolated row whose flip raises CE by +1.4 to +2.1, that row ranked 1st
> for one seed and last for two others, because CE is non-monotone along the flip direction (it dips
> for a small embedding shift, then jumps). On the unit-test toy (`_lone_row_toy`) that row ranked
> 3rd, 18th and 1st of 31 for seeds 0, 1 and 2. The unit test pins seed 0. These numbers were
> re-measured with a fresh classifier per feature width. On real data, a smoke on
> `lcld_v2/selected/balanced/context_1000.csv` (k=50 class-1 flips, 64 test rows) *lowered* test CE
> by 0.053. On a saturated toy (CE ≈ 7e-4) the gradient is ~1e-4
> while the actual flip adds +5.8 CE. Treat `label-flip` results as "first-order influence", not as
> a near-optimal k-flip attack. The per-trial JSON records `flipped_scores` and
> `first_order_ce_gain_sum` next to the realised Δ.

## 6. Why CAA (and Y-PGD) are omitted

CAA runs CAPGD, then MOEVA on the failures. MOEVA is a genetic search over the joint input, which
is not feasible over an R% block of a 1k–10k row context. PGD on Y is impossible on v2 (§2).
`--method caa` or `--method pgd` exits with this rationale; there is no stub.

## 7. Selected contexts

`scripts/select_contexts.py` runs for each of {url_unique, lcld_v2, wids} × {natural, balanced}:

1. Load the raw CSV and drop `issue_d` for LCLD (the raw file keeps it). Order columns like
   `val_2000.csv`.
2. Deduplicate by row fingerprint: SHA-256 of `repr(tuple(canonical str of every cell, including
   the target))`. Integers and integral floats encode identically and NaN is normalised; the
   DataFrame index is ignored.
3. Exclude every row whose fingerprint appears in `val_2000.csv` or `test_attack_1000.csv`. What
   remains is the pool.
4. Draw 10 candidates with `numpy.random.default_rng(seed)`, seeds 0..9. Natural candidates are a
   uniform draw without replacement of 8000 rows (url_unique) or 10000 rows (lcld_v2, wids), capped
   at the pool size. Balanced candidates have `min(10000, 2·minority)` rows, 50/50.
5. Score each candidate on `val_2000.csv` with the shared runtime. The winner has the highest
   accuracy (ties go to the lower seed).
6. Take nested children of the winner without replacement: `context_5000 ⊂ winner` and
   `context_1000 ⊂ context_5000`. Seed: `10000 + winning_seed`. Balanced children stay 50/50. A
   child is skipped (recorded in the manifest) if the winner is not larger than it.
7. Write `datasets/<name>/splits/selected/<mode>/context_<N>.csv`, `context_5000.csv`,
   `context_1000.csv` and `manifest.json`.

`manifest.json` holds:
- dataset, mode, target, dropped columns and column order;
- raw row count, duplicate count, reserved fingerprint count, raw rows matching reserved, pool size;
- candidate seeds with accuracy, full val metrics and class counts; the winning seed; the child seed;
- per-file row count, class counts and sha256;
- sha256 of the raw CSV, val and test_attack;
- checks: unique rows per file, `overlap_with_reserved` (0 per file), `nesting`, `issue_d_absent`,
  `columns_match_val`;
- model and version info.

### Selection results

Run of 2026-09-14 (`python scripts/select_contexts.py --gpu 3`, model_seed 0, best-effort; log in
`results/select_contexts/select_contexts.log`). No raw duplicates were found in any dataset, and
exactly the 3000 reserved rows were removed from each.

| dataset | mode | pool | files (class counts 0/1) | winning seed | val acc (range over 10) |
|---|---|---|---|---|---|
| url_unique | natural | 8,182 | 8000 (4054/3946), 5000, 1000 | 2 | 0.9670 |
| url_unique | balanced | 8,182 | 8054 (4027/4027), 5000, 1000 | 7 | 0.9675 |
| lcld_v2 | natural | 1,217,092 | 10000 (8092/1908), 5000, 1000 | 5 | 0.8075 |
| lcld_v2 | balanced | 1,217,092 | 10000 (5000/5000), 5000, 1000 | 0 | 0.6520 |
| wids | natural | 31,776 | 10000 (9081/919), 5000, 1000 | 2 | 0.9290 |
| wids | balanced | 31,776 | 5700 (2850/2850), 5000, 1000 | 8 | 0.7995 |

The exact class counts of every file and the full candidate accuracies are in each `manifest.json`.
All six outputs were re-validated from the written CSVs: unique rows, zero overlap with
val/test_attack, `1000 ⊂ 5000 ⊂ winner`, exact 50/50 in balanced mode, no `issue_d`, and file and
source hashes matching the manifest.

**The existing `datasets/*/splits/context_*.csv` files are not the selected contexts.**
Experiments should use `splits/selected/{natural,balanced}/...`. The old files are left untouched.

## 8. Output schema

`scripts/attack_context.py --out DIR` writes:

```
DIR/args.json              resolved CLI args
DIR/ensemble_config.json   resolved TabPFN ensemble_configs_
DIR/versions.json          python/torch/cuda/numpy/sklearn/tabpfn, checkpoint path + sha256, GPU name
DIR/summary.json           clean metrics, clean_max_repeat_abs_dprob, chunking_exact, seeds, k,
                           aggregation, per_run_best (winning trial per run, in full),
                           trials[] (without histories), aggregate_delta, aggregate_poisoned
DIR/run.log
DIR/trials/run{r}_sub{s}.json
    attack, run_id, subsample_id, attacked_indices, k_requested, k_actual,
    clean{ce,roc_auc,accuracy,f1,tn,fp,fn,tp}, poisoned{...}, delta{...}, seconds, history[]
    x-capgd:    constraints_mode, norm, eps, eps_effective, best_restart, restarts[],
                budget_repair, n_cells_changed, max_l2_delta_scaled, max_linf_delta_scaled,
                within_eps_after_repair, standard_repair (plain-repair metrics, when budget_repair on),
                relation_violation_{clean,final}_mean, relation_violated_rows_final (full mode)
    label-flip: flipped_indices, flipped_scores, first_order_ce_gain_sum, n_eligible,
                score_stats, class_counts_clean, class_counts_poisoned
DIR/trials/run{r}_sub{s}_delta.npz
    x-capgd:    row_indices, cell_delta_raw [k, d], feature_names
    label-flip: flipped_indices, y_clean, y_poisoned
DIR/context_poisoned.csv   only with --save-full-context: the single strongest trial
                           (largest CE increase); the winner is named in summary.json
                           under best_context_trial
```

**Aggregation rule.** Runs capture CAPGD randomness; row subsamples capture which rows were drawn.
For each run we keep the **best row subsample by `delta.ce`** (`aggregate.best_per_run`), and the
headline `aggregate_delta` / `aggregate_poisoned` are computed **across runs only**. This models an
attacker who tries several row subsets and keeps the strongest. The winners are saved in full under
`per_run_best`, and the flat mean over every trial is still available as
`aggregate_delta_all_trials` / `aggregate_poisoned_all_trials`.

Because the per-run figure is a **maximum** over subsamples, it is biased upward relative to the
flat all-trial mean; compare like with like when quoting numbers.

`aggregate.summarize_trials` reports `overall` (mean/std/min/max), `by_run`, `by_subsample`, and
`across_runs` / `across_subsamples` (stats of the group means). Standard deviations use ddof=0.

`scripts/infer.py --out DIR` writes `args.json`, `metrics.json` (metrics, `max_repeat_abs_dprob`,
`chunking_exact`), `predictions.csv`, `ensemble_config.json`, `versions.json` and `run.log`.

## 9. Commands

```bash
# tests (CPU) and optimiser equivalence
conda run -n tabfm python -m pytest tests/tabfm_experiments -q
conda run -n tabfm python scripts/check_capgd_equiv.py
TABFM_GPU=1 conda run -n tabfm python -m pytest tests/tabfm_experiments/test_gpu_smoke.py -q

# selected contexts (all six) or one
conda run -n tabfm python scripts/select_contexts.py --gpu 1
conda run -n tabfm python scripts/select_contexts.py --gpu 1 --dataset url_unique --mode natural

# clean inference on the shared runtime
conda run -n tabfm python scripts/infer.py --model tabpfnv2 --gpu 1 \
  --train datasets/url_unique/splits/selected/natural/context_1000.csv \
  --test  datasets/url_unique/splits/test_attack_1000.csv \
  --out   results/infer/url_unique_nat1000 --deterministic-mode best-effort

# X_train CAPGD, one trial (a small start; the default grid is 5 runs x 5 subsamples)
conda run -n tabfm python scripts/attack_context.py --gpu 1 --attack x-capgd \
  --train datasets/url_unique/splits/selected/natural/context_1000.csv \
  --test  datasets/url_unique/splits/test_attack_1000.csv \
  --out   results/attacks/url_unique_nat1000_x \
  --row-percent 5 --norm l2 --eps 0.5 --n-iter 40 --constraints full --constraint-penalty 1.0 \
  --n-runs 1 --n-row-subsamples 1

# WiDS (109 features): chunk the test set if memory is tight
conda run -n tabfm python scripts/attack_context.py --gpu 1 --attack x-capgd \
  --train datasets/wids/splits/selected/natural/context_1000.csv \
  --test  datasets/wids/splits/test_attack_1000.csv \
  --out   results/attacks/wids_nat1000_x --test-batch-size 128 --n-runs 1 --n-row-subsamples 1

# influence-ranked label flips, class-1 rows only
conda run -n tabfm python scripts/attack_context.py --gpu 1 --attack label-flip \
  --train datasets/lcld_v2/splits/selected/balanced/context_1000.csv \
  --test  datasets/lcld_v2/splits/test_attack_1000.csv \
  --out   results/attacks/lcld_bal1000_flip --row-percent 5 --attack-class 1
```

---

## 10. Transfer to XGBoost (`scripts/xgb_transfer.py`)

The poison is optimised white-box against TabPFNv2's in-context forward pass. This
experiment asks whether it still hurts a model that never saw that gradient: the poisoned
context is handed to XGBoost as an ordinary **training set**, XGBoost is tuned and trained
on it from scratch, and scored on the same `test_attack_1000.csv`.

### Two validation protocols

A victim that retrains on poisoned data also *validates* on something. Where that
validation set comes from changes the result, so both are reported:

| | training set | validation set (HPO + early stopping) |
|---|---|---|
| **A** | 80% of the given context | the other 20% of the **same** context -- poisoned when the context is |
| **B** | 100% of the given context | `datasets/<ds>/splits/val_2000.csv`, always **clean** |

A is the attacker who owns the whole pipeline input; B is a defender holding a trusted
validation set. `--val-ratio` changes A's split (default 0.2).

### What is compared against what

Hyperparameters are re-tuned for **every** context (optuna TPE, objective = validation
logloss, `--hpo-trials`, default 30). That is deliberate: a victim retraining on poisoned
data re-tunes too, so the hyperparameters are part of what the poison gets to move. It
also means some of the measured damage is HPO landing somewhere different, not only the
trees being worse.

Aggregation matches the TabPFNv2 side exactly. `--attack-dir` rebuilds the **per-run-best**
poisoned contexts -- the same `run<r>_sub<s>` winners `summary.json` recorded -- from
`trials/*_delta.npz`, so both models are aggregated over the identical five contexts.
Rebuilding is exact: `apply_trial_delta` reproduces the attack's own `context_poisoned.csv`
to 1e-13 (CSV float round-trip) for x-capgd and bit-exactly for label flips.

Each poisoned context `i` is paired against a clean fit with the same seed `i`, so the
reported delta is per-run and the clean baseline carries matched variability (seed drives
both the A split and the TPE stream). Clean baselines depend only on
(train, test, val, protocol, HPO settings), so `--clean-cache-dir` reuses them across all
eight runs of a dataset.

### Metrics

CE (= test logloss), accuracy, balanced accuracy, MCC, ROC-AUC, F1 -- the same set as the
TabPFNv2 tables, with balanced accuracy and MCC derived from the confusion counts by
`metrics.with_derived` (which also back-fills them for results written before those two
metrics existed). The printed table adds:

* `rel%` -- `100 * delta / |clean|`
* `transfer%` -- `100 * XGBoost delta / TabPFNv2 delta`; 100 means the poison costs both
  models the same, 0 means it does not transfer at all
* `~` on either -- the denominator is near zero or inside its own across-run spread, so the
  ratio is noise. coil2000's clean model is degenerate (MCC 0, balanced accuracy 0.5000),
  so its percentage columns are `~` and only the absolute columns mean anything there.

### Commands

```bash
# one attack run, both protocols, compared against its TabPFNv2 deltas
conda run -n tabfm python scripts/xgb_transfer.py \
  --attack-dir results/main_experiments/lcld_v2/label-flip_random_r016 \
  --out results/xgb_transfer/lcld_v2/label-flip_random_r016

# explicit files (no attack dir); --poisoned-train takes one or more CSVs
conda run -n tabfm python scripts/xgb_transfer.py --model xgboost \
  --train datasets/lcld_v2/splits/selected/natural/context_5000.csv \
  --test  datasets/lcld_v2/splits/test_attack_1000.csv \
  --val   datasets/lcld_v2/splits/val_2000.csv \
  --poisoned-train results/.../context_poisoned.csv \
  --out results/xgb_transfer/one_off

# the whole 32-command grid (CPU only, safe to run alongside the GPU attacks)
bash scripts/run_xgb_transfer.sh          # DRY_RUN=1 to preview
```

`--model` currently accepts `xgboost` only. Needs `xgboost` and `optuna` in the `tabfm`
env. Output is `<out>/summary.json` with per-fit records, per-run paired deltas, the
aggregates for both protocols, and the TabPFNv2 side for comparison.

---

## 11. Test-agnostic random label flips (`--attack label-flip-random`)

Every attack above is **transductive**: it reads `test_attack_1000.csv` and optimises the
poison against it, so it measures an upper bound on damage to *that* test set. This one
never looks at the test set. For a given R it flips the labels of a uniformly random R% of
the context rows, and each run draws a **fresh** random subset:

    rows_run = sample_row_indices(y_ctx, k, rng=random_flip_rng(row_seed, run, 0))

The test set is used only to *score* the poisoned context. `test_cli_is_test_agnostic`
checks this directly: the same context and seeds with two unrelated test files flip
exactly the same rows.

**Aggregation.** One draw per run (`--n-row-subsamples` is fixed at 1 and larger values
are rejected). The usual rule keeps the best subsample per run *by test ΔCE*, which would
bring the test set back in through the selection. The headline is therefore a plain mean
over `--n-runs` independent draws (default 5). `summary.json`'s `aggregation` field says so.

`--save-full-context` still writes the single highest-ΔCE draw as `context_poisoned.csv`.
That choice of *which file to save* uses the test set; no reported number does. The
per-trial `trials/*_delta.npz` (with `y_poisoned`) is written for every draw, so the
XGBoost transfer (§10) rebuilds all of them and is unaffected.

**Knobs.** `--attack-class 0|1` restricts the draw to one class. `--row-seed` sets the
draw stream. Nothing else applies: there is no optimiser and no GA.

**Cost.** One forward per trial (~0.3 s on lcld_v2 5000 × 28), versus ~600 GA evaluations
for `label-flip`.

**Evaluating on a general test set.** Because the attack is independent of the test set,
it can be scored on any held-out file — e.g. the full `splits/test.csv` rather than the
1000-row attack set — without changing what was attacked. Scores on `test_attack_1000.csv`
remain directly comparable with the tables in the earlier sections.

```bash
conda run -n tabfm python scripts/attack_context.py --gpu 0 --attack label-flip-random \
  --row-percent 16 \
  --train datasets/lcld_v2/splits/selected/natural/context_5000.csv \
  --test  datasets/lcld_v2/splits/test_attack_1000.csv \
  --out   results/main_experiments/lcld_v2/label-flip-random_random_r016
```

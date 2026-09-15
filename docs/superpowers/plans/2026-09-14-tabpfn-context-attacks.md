# TabPFNv2 Context-Poisoning Implementation Plan

> **For Claude Code / any implementer:** this file is the full spec. Do not rely on prior chat. Implement task-by-task. Do **not** `git commit` unless the user explicitly asks. This workspace may not be a git repo.

**Goal:** Build a shared TabPFNv2 runtime and two context-poisoning attacks (X_train CAPGD; Y_train influence-ranked label flips), plus validation-selected context CSVs, without running the full attack grid.

**Architecture:** One package `scripts/tabfm_experiments/` used by thin CLIs. Clean inference, CAPGD inner loop, and final scoring all call the same `fit_with_differentiable_input` + `forward` path. Poisoned-context normalization is **not** frozen (unlike the current `StatFreezer` in `scripts/attack_context.py`).

**Tech stack:** Python 3.11, conda env `tabfm`, PyTorch, local TabPFN 8.5.0 (`ModelVersion.V2`), TabularBench constraints, sklearn metrics, pytest.

**Related short plan (Cursor UI):** `/home/ssn899/.cursor/plans/tabpfn-context-attacks_e6148acf.plan.md` — this document supersedes it.

---

## Handoff: do / do not

**Do**
- Use `conda run -n tabfm python ...` (or `conda activate tabfm`) for every command.
- Load **TabPFNv2 only**: `TabPFNClassifier.create_default_for_version(ModelVersion.V2, ...)`.
- Keep `n_estimators=1`. Disable fingerprint, feature/class shifts, polynomial features, outlier removal, row subsample.
- Use `fit_with_differentiable_input` + `clf.forward(X_test, use_inference_mode=True)` for baseline, gradients, and reported metrics.
- Optimize and report on `test_attack_1000.csv` (transductive white-box upper bound).
- Implement X_train CAPGD and one Y_train method (influence-ranked discrete flips).
- Build selected contexts for all six dataset/mode combinations.
- Write `docs/context_poisoning.md`.
- Run unit tests + tiny GPU smokes. Prefer GPU 1 or 3 (`nvidia-smi` first).

**Do not**
- Do not switch to TabPFNv2.5 / v3 / TabICL.
- Do not implement CAA, MOEVA, PGD-on-Y, random label flips, greedy label flips, or iterative influence.
- Do not freeze TabPFN preprocessing stats (`StatFreezer` in `scripts/attack_context.py` is the old design; replace it).
- Do not run the full `n_runs × n_row_subsamples` attack grid.
- Do not overwrite existing files under `datasets/*/splits/` except the new `splits/selected/` tree.
- Do not drop `issue_d` from `datasets/lcld_v2/lcld_v2.csv` (raw provenance). Drop it only in **selected** split outputs, as `make_splits.py` already did for current splits.
- Do not treat native sklearn categorical preprocessing as compatible with this gradient path.
- Do not promise bitwise GPU reproducibility.

---

## Global constraints

- Conda env: `tabfm`. Python 3.11. TabPFN installed from `/home/ssn899/Desktop/TabFM/TabPFN` (8.5.0).
- Datasets: `/home/ssn899/Desktop/TabFM/datasets/{url_unique,lcld_v2,wids}/`.
- Constraints code: `/home/ssn899/Desktop/TabFM/tabularbench/` (add to `PYTHONPATH` if not installed).
- Reference CAPGD: `/home/ssn899/Desktop/TabFM/tabularbench/tabularbench/attacks/capgd/capgd.py`.
- Type repair: `/home/ssn899/Desktop/TabFM/tabularbench/tabularbench/attacks/utils.py` (`fix_types`, `fix_immutable`, `fix_equality_constraints`).
- Existing draft scripts (reference only, then thin-CLI rewrite): `scripts/infer.py`, `scripts/attack_context.py`, `scripts/make_splits.py`, `scripts/check_capgd_equiv.py`.
- All binary classification. Targets: `is_phishing` (url_unique), `charged_off` (lcld_v2), `hospital_death` (wids).
- Model seed is independent of row-sampling seeds and CAPGD restart seeds.
- Determinism default: **best-effort** (FlashAttention allowed). Strict is opt-in.

---

## Scientific protocol (must follow)

### Threat model

White-box **transductive context poisoning**. The attacker may read `X_test`, `y_test` from `test_attack_1000.csv`, may modify a budgeted subset of the ICL context (`X_train` and/or `Y_train`), and is scored on that same test set. This is an upper bound, not held-out generalization. Document that in `docs/context_poisoning.md`.

Djilani et al. (arXiv:2506.02978, SaTML 2026) attacked **test X** with context held clean. We attack the **context**. Same datasets, same CAPGD optimiser family, opposite side of the prompt.

### Why CAA is omitted

CAA = CAPGD then MOEVA on failures. MOEVA is a genetic search over the joint input. Joint search over an R% block of a 1k–10k context is not feasible. `--method caa` / `pgd` must raise a clear error. Do not stub fake CAA.

### TabPFNv2 input path (why X PGD works and Y PGD does not)

The model is a table transformer. Train+test **features** go through:

`X → NaN flags → impute → StandardScaler (train stats) → group-norm → Linear(feature_group_embedder) → embeddings`

Train **labels** go through:

`Y_train → NaN-pad test rows → impute → _flatten_multiclass_targets → Linear(target_embedder) → embeddings`

Flatten (classification only) is `(y > unique_ys).sum()` per batch. Autograd of `>` is 0, so `∂L/∂Y_train` is identically 0 even though `target_embedder` is a linear layer. That is why we do **not** PGD on Y for v2.

Test Y is **not** an input; test target cells are NaN. The MLP head reads the target-column hidden state on test rows and outputs class logits. Use probabilities from `forward(..., use_inference_mode=True)` unless you explicitly request logits.

“Categorical” **X** in these CSVs is already an integer code (or real). TabularBench CAPGD does **not** one-hot them. Treat cat/int as continuous in scaled space during steps; repair types **once at the end**. Do not insert per-step `round`.

### Shared evaluation (replaces StatFreezer)

Every forward, including CAPGD steps, must `fit_with_differentiable_input` on the **current full context** (poisoned rows included) so scaler / constant-mask / group-norm are those of the poisoned prompt. Final metrics use the same function. Test-row chunking is still exact: test rows do not attend to each other; train stats are fitted on context only.

### Determinism

**Best-effort (default):**
- `random_state` / torch / numpy seeds fixed
- `n_estimators=1`
- inference flags listed below
- `inference_precision=torch.float32`
- do **not** set `torch.use_deterministic_algorithms(True)`
- record max `|Δp|` over 2–3 repeated clean forwards in the run log

**Strict (opt-in `--deterministic-mode strict`):**
- `torch.use_deterministic_algorithms(True)` (and warn/abort if a kernel is forbidden)
- may disable flash SDP
- still not bitwise across GPU models/drivers

### Model construction (copy exactly)

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.preprocessing.configs import PreprocessorConfig
import torch

inference_config = {
    "PREPROCESS_TRANSFORMS": [
        PreprocessorConfig("none", categorical_name="numeric")
    ],
    "FINGERPRINT_FEATURE": False,
    "POLYNOMIAL_FEATURES": "no",
    "FEATURE_SHIFT_METHOD": None,
    "CLASS_SHIFT_METHOD": None,
    "OUTLIER_REMOVAL_STD": None,
    "SUBSAMPLE_SAMPLES": None,
}

clf = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2,
    n_estimators=1,
    device=device,  # "cpu" or "cuda:{id}"
    random_state=model_seed,
    inference_precision=torch.float32,
    differentiable_input=True,
    ignore_pretraining_limits=True,
    n_preprocessing_jobs=1,
    inference_config=inference_config,
)
# fit_with_differentiable_input cannot infer this from a float y tensor
clf.n_classes_ = 2
```

Pass `X` and `y` as `float32` tensors. Categorical feature indices must stay empty/`None` or `reject_categoricals_for_differentiable_input` raises. That is intended: cats are numeric codes.

### Metrics (all binary)

Always compute on `test_attack_1000` unless a smoke flag shrinks it:

- mean CE: `F.nll_loss(log(probs.clamp_min(1e-12)), y, reduction="mean")`
- ROC-AUC on `probs[:, 1]`
- accuracy, F1 (binary, default sklearn), confusion counts TN/FP/FN/TP
- for attacks: clean, poisoned, and delta for every metric

Do not use MCC as the primary reported metric (Djilani used MCC for some tables; this project asked for the list above). Accuracy is used to **select** context candidates.

---

## Datasets and files

| Dataset dir | Raw CSV | Metadata | Target | Feature cols (excl. target) |
|---|---|---|---|---|
| `datasets/url_unique/` | `url_unique.csv` | `url_unique_metadata.csv` | `is_phishing` | 63 |
| `datasets/lcld_v2/` | `lcld_v2.csv` | `lcld_v2_metadata.csv` | `charged_off` | 28 in splits (raw has `issue_d` extra) |
| `datasets/wids/` | `wids.csv` | `wids_metadata.csv` | `hospital_death` | 108 |

Existing splits (do not overwrite): `train.csv`, `test.csv`, `val_2000.csv`, `test_attack_1000.csv`, `context_*.csv`, `split_info.json`.

Metadata columns: `feature, type, mutable, min, max` (column order varies; LCLD is `feature,min,max,mutable,type`). Types: `real`, `int`, `cat` (LCLD also has `date` on `issue_d` only). Mutable parsed as true if string in `{true,1,True}`.

**URL types:** almost all `int`/`real`; target is `cat`. Binary indicators are typed `int`, so end-of-attack repair uses `torch.fix` on the **delta**, not `round` on the value.

**LCLD:** drop `issue_d` from any tensor that enters TabPFN. Constraints use **feature names**, not positions.

**WiDS:** target is first CSV column; metadata starts at `age`. After dropping target, X column order must match metadata `feature` list excluding `hospital_death`.

Row fingerprint for exclusion/dedup: SHA-256 of a canonical encoding of all columns including target, e.g. `repr` of a tuple of Python `str` of each cell with `nan` normalized. Do not use DataFrame index.

---

## Constraints (X_train CAPGD only)

Build a TabularBench `Constraints` object via `get_constraints_from_metadata` after filtering out the target (and `issue_d` for LCLD).

Relation constraint factories (import, do not rewrite by hand):

- URL / url_unique: `tabularbench.datasets.samples.url.get_relation_constraints()` — `Feature(i)` is **0-based X column index** in metadata order without target. Same schema as `url`.
- LCLD: `tabularbench.datasets.samples.lcld.get_relation_constraints()` — named features (`installment`, ratios, …). No `issue_d` constraint remains.
- WiDS: `tabularbench.datasets.samples.wids.get_relation_constraints(metadata)` — `Feature(i+1) <= Feature(i)` for `i in range(33, 94, 2)` on **X after dropping target**.

During CAPGD, subtract a **normalized** relation-penalty from mean test CE (same sign as TabularBench: they do `loss_indiv - executor.execute(unscaled)` while maximizing CE, i.e. they **reward** constraint satisfaction? Check `capgd.py` lines 283–289: CE is maximized via the attack’s ascent on `loss_indiv`; they subtract the executor output. Read `ConstraintsExecutor.execute`: it returns a **violation** tensor (higher = more violated) in some backends and a boolean-ish satisfaction in others. **Match TabularBench `CAPGD.attack_single_run`:** `loss_indiv = ce - executor.execute(inverse_transform(x_adv))` when relations exist. Use `AndConstraint(relation_constraints)` as they do. Pass `feature_names` for LCLD names.

If the executor returns a per-row tensor, reduce with mean over the **attacked rows only**. Expose `--constraint-penalty` (default 1.0) multiplying that term.

**Inside the loop:** scaled-space eps-ball + `[0,1]` box + mutable mask. Optional `--fix-equality-iter` (TabularBench default True): unscale → `fix_equality_constraints` → rescale. **No type rounding inside the loop.**

**After all steps (and when scoring a restart):** `inverse_transform` → `fix_types` → `fix_immutable` → `fix_equality_constraints`.

`fix_types` (copy the reference, including the quirky early-return):

```python
# if no int columns, return without rounding cats (reference behavior)
int: x_adv = x_clean + torch.fix(x_adv - x_clean)
cat: x_adv = torch.round(x_adv)
```

All three datasets have int columns, so cats still get rounded.

Scaler: min-max to `[0,1]` with metadata `min`/`max` by default (`--scaler metadata`). Constant features (`max<=min`): range 1, immutable in practice. `--scaler train` uses the current context min/max (document that this couples the box to the poisoned batch).

**Budget:** **per attacked row**, in scaled space. Default `norm=L2`, `eps=0.5`, `eps_margin=0.05` so effective `eps' = eps * (1 - eps_margin)`. Untouched context rows have mask 0.

---

## CAPGD hyperparameters

Port the loop from `scripts/attack_context.py::capgd` / TabularBench `attack_single_run`. Keep iterate-for-iterate agreement on a toy objective (`scripts/check_capgd_equiv.py` already exists; keep it passing or re-point it at the new module).

Defaults (Djilani / TabularBench unless noted):

| Arg | Default | Notes |
|---|---|---|
| `--norm` | `l2` | Paper benchmark example uses L2 |
| `--eps` | `0.5` | scaled units |
| `--eps-margin` | `0.05` | inherited; paper does not state it |
| `--n-iter` / steps | `10` | |
| `--momentum` | `0.75` | APGD `a` |
| `--rho` | `0.75` | step-halving |
| `--n-restarts` | `1` plus clean start documented as “clean + random” via `--n-runs` | Paper: five attack runs, clean plus random start. Map: `--n-runs 5` is independent attack seeds; restart 0 = clean start, later runs / `--random-start` = random. |
| `--eot-iter` | `1` | |
| `--loss` | `ce` | mean test CE |
| `--row-percent` | `5` | fraction of context rows attacked |
| `--attack-class` | unset | if 0 or 1, sample only from that label |
| `--n-row-subsamples` | `5` | independent row-subset draws |
| `--n-runs` | `5` | independent CAPGD seeds |
| `--deterministic-mode` | `best-effort` | |
| `--save-full-context` | off | compact deltas always on |

Adaptive checkpoints: `steps_2 = max(int(0.22*steps),1)`, `steps_min = max(int(0.06*steps),1)`, `size_decr = max(int(0.03*steps),1)` as in the reference.

Linf vs L2 differences (must copy): step is `sign(grad)` vs normalized grad; mutable mask on momentum terms only for L2; L2 radial projection has the `1e-12` pad asymmetry between the two projections (`pad` flag in current `capgd()`).

Objective: **maximize** mean test CE minus constraint term. Labels of poisoned rows stay in the context and are **not** in the loss.

---

## Row sampling

For a context of `n` rows, `k = max(1, round(n * row_percent / 100))`.

Eligible indices: all rows, or rows with `y == attack_class`. If `len(eligible) < k`, use all eligible and record `k_actual`.

Each subsample `s` uses `numpy.random.Generator(seed_row + s)` without replacement. Different subsamples may overlap. Attacked rows are mutable; others are immutable for that trial.

Seeds (independent streams):

- `model_seed` (default 0)
- `row_seed` (default 1)
- `attack_seed` (default 2)  # CAPGD random start / restarts
- `flip_seed` unused for influence (ranking is determined); still accept `--seed` as model seed

Trials: all pairs `(run_id in 0..n_runs-1, subsample_id in 0..n_row_subsamples-1)`. Aggregate mean/std over the **25** trials **and** grouped-by-subsample / grouped-by-run, because they are hierarchical, not 25 iid draws.

---

## Y_train attack (one method)

**Name:** `influence`. **Not** CAPGD/PGD.

1. Eligible set as above; `k` as above. Rank **all eligible rows** (do not randomly subsample which labels to consider — ranking *is* the row choice). `--n-row-subsamples` for this method is still executed as repeated evals with the same ranking if GPU noise is being measured, but the flipped index set is identical; document that. Default `--n-row-subsamples 1` for `--attack label-flip` if you want to avoid wasted repeats; the CLI may keep 5 for API uniformity — if so, results will be near-duplicates.
2. One forward + backward of mean test CE w.r.t. **target embeddings of train rows** (hook `TabPFNV2._embed_targets` output, or the tensor fed as the target column). `Y` stays integer; do not backprop through `_flatten_multiclass_targets`.
3. Let `e_i` be row `i`’s target embedding, `g_i = ∂L/∂e_i`. For binary labels densified to `{0,1}`, `e(1)-e(0)` equals the first column of `target_embedder.weight` because `Linear` sees `[y_flat, nan_indicator]` and finite labels have nan_indicator 0. Score of flipping `i`:

   `s_i = ⟨g_i, (1 - 2 yflat_i) * W[:, 0]⟩`

   Equivalent: embed a synthetic all-0 and all-1 train column with the same nan flags and unique_ys, then `s_i = ⟨g_i, e_alt_i - e_i⟩`.
4. Flip the `k` eligible rows with largest `s_i` (`y ← 1-y`). X unchanged.
5. Re-fit / forward on the integer flipped context for reported metrics (no STE in the final path).

If hooking is fragile across TabPFN versions, the synthetic `e(0)/e(1)` method is preferred: call `_embed_targets` twice under `torch.no_grad` for constant 0/1, and once with grad for the true Y.

Save flipped `Y_train` (and indices) when `--save-full-context`.

---

## Selected contexts

New CLI: `scripts/select_contexts.py`.

For each of `{url_unique, lcld_v2, wids}` × `{natural, balanced}`:

1. Load raw CSV. For LCLD, drop `issue_d` for sampling/scoring (keep raw file).
2. Deduplicate by fingerprint.
3. Exclude rows whose fingerprint appears in existing `val_2000.csv` **or** `test_attack_1000.csv`.
4. Remaining pool `P`.
5. Draw **10** candidate subsets:
   - Natural sizes: url_unique **8000**, lcld_v2 **10000**, wids **10000**. If `len(P) < size`, use `len(P)` and record it.
   - Balanced size: `min(10000, 2 * minority_count_in_P)`, 50/50 (or as even as possible if odd). WiDS is heavily imbalanced; the cap will bind.
   - Candidates may overlap each other; each candidate has unique rows internally.
   - Seeds: `0..9` with `numpy.random.Generator`.
6. Score each candidate with the **same** TabPFN runtime on `val_2000.csv`, metric **accuracy**. Winner = highest accuracy (tie: lower seed).
7. From the winner, nested children **without replacement from the winner**:
   - `5000` if `len(winner) >= 5000` else skip that file and record why
   - `1000` if `len(winner) >= 1000`
   - Balanced children stay 50/50 nested subsets of the winner.
8. Write:

```
datasets/<name>/splits/selected/{natural,balanced}/
  context_<N>.csv          # winner
  context_5000.csv         # if applicable
  context_1000.csv
  manifest.json
```

`manifest.json` must include: dataset, mode, target, dropped columns, pool size, duplicate count, reserved fingerprints count, candidate seeds and scores, winning seed, class counts at each size, source file hashes (sha256 of raw csv + val + test_attack), nesting checks (`set(1000) ⊂ set(5000) ⊂ set(winner)`), overlap-with-reserved = 0, `issue_d` absent for LCLD.

---

## File map

Create:

```
scripts/tabfm_experiments/__init__.py
scripts/tabfm_experiments/config.py          # seeds, inference_config, paths
scripts/tabfm_experiments/data.py            # load csv, fingerprint, metadata, splits
scripts/tabfm_experiments/metrics.py         # binary metrics + deltas
scripts/tabfm_experiments/runtime.py         # TabPFNv2 wrapper, determinism, eval
scripts/tabfm_experiments/constraints.py     # metadata + relation constraints + repair
scripts/tabfm_experiments/capgd.py           # optimiser (no TabPFN inside)
scripts/tabfm_experiments/attacks.py         # context CAPGD + influence flip
scripts/tabfm_experiments/sampling.py        # row/class sampling, seed streams
scripts/tabfm_experiments/aggregate.py       # trial summaries
scripts/tabfm_experiments/io.py              # artifacts, jsonable, hashes
scripts/select_contexts.py
tests/tabfm_experiments/test_metrics.py
tests/tabfm_experiments/test_sampling.py
tests/tabfm_experiments/test_aggregate.py
tests/tabfm_experiments/test_constraints.py
tests/tabfm_experiments/test_capgd.py
tests/tabfm_experiments/test_select_contexts.py
tests/tabfm_experiments/test_runtime_cpu.py  # tiny synthetic, skip if no torch
docs/context_poisoning.md
```

Modify (keep as thin CLIs that import the package):

- `scripts/infer.py`
- `scripts/attack_context.py`

Do not delete the old `capgd()` in `attack_context.py` until the new module is covered by `check_capgd_equiv.py`. Then replace the body with a wrapper.

Optional: `scripts/check_capgd_equiv.py` imports `scripts.tabfm_experiments.capgd`.

---

## CLI

### `scripts/infer.py`

```
conda run -n tabfm python scripts/infer.py \
  --model tabpfnv2 --gpu 1 \
  --train datasets/url_unique/splits/selected/natural/context_1000.csv \
  --test datasets/url_unique/splits/test_attack_1000.csv \
  --out results/infer/url_unique_nat1000 \
  --deterministic-mode best-effort
```

Must use the shared runtime (differentiable path), not sklearn `fit`/`predict_proba`, so numbers match attacks. `--model tabiclv2` may remain as a hard error or the old path; do not spend time on TabICL.

### `scripts/attack_context.py`

Required args:

```
--train PATH --test PATH --out PATH
--attack {x-capgd,label-flip}     # default x-capgd
--gpu 1
--row-percent 5
--attack-class {0,1}              # optional
--n-row-subsamples 5
--n-runs 5
--norm l2 --eps 0.5 --eps-margin 0.05 --n-iter 10
--momentum 0.75 --rho 0.75 --n-restarts 1
--scaler {metadata,train}
--constraints {none,box,full}     # default full
--constraint-penalty 1.0
--fix-equality-iter / --no-fix-equality-iter
--deterministic-mode {best-effort,strict}
--model-seed 0 --row-seed 1 --attack-seed 2
--test-batch-size N               # optional chunking
--max-test N                      # smoke only
--save-full-context
```

`--method caa|pgd` → `SystemExit` with the CAA/Y-PGD rationale.

Outputs under `--out`:

- `args.json`
- `ensemble_config.json` (resolved `ensemble_configs_`)
- `versions.json` (`tabpfn`, `torch`, `cuda`, checkpoint path/hash if available)
- `summary.json` (all trials + aggregates)
- `trials/run{r}_sub{s}.json` (history, attacked indices, metrics clean/poisoned/delta)
- `trials/run{r}_sub{s}_delta.npz` (row indices + cell deltas; Y flips as index list)
- optional full `context_poisoned.csv`
- `run.log`

### `scripts/select_contexts.py`

```
conda run -n tabfm python scripts/select_contexts.py --gpu 1
# or --dataset url_unique --mode natural
```

---

## Interfaces (lock these names)

```python
# runtime.py
def build_tabpfn_v2(device: str, model_seed: int, deterministic_mode: str) -> TabPFNClassifier: ...

def evaluate_context(
    clf,
    X_ctx: torch.Tensor,      # [n_ctx, d] float32
    y_ctx: torch.Tensor,      # [n_ctx] float32 class ids
    X_test: torch.Tensor,     # [n_test, d]
    y_test: torch.Tensor,     # [n_test] long
    *,
    need_grad: bool,
    test_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Returns (mean_ce, grad_wrt_X_ctx or None, probs [n_test, 2]).
    Always refits on the provided context. Chunks only X_test.
    """

# sampling.py
def k_from_percent(n: int, row_percent: float) -> int: ...
def sample_row_indices(y: np.ndarray, k: int, *, attack_class: int | None, rng: np.random.Generator) -> np.ndarray: ...

# capgd.py
def capgd(evaluate, x0, *, eps, n_iter, rho, momentum, repair, mut, better, box, norm="L2", on_iter=None):
    """evaluate(x_scaled, need_grad) -> (loss, grad). Returns x_best, loss_best, hist."""

# attacks.py
def run_x_capgd(...) -> TrialResult: ...
def run_label_flip_influence(...) -> TrialResult: ...

# metrics.py
@dataclass
class BinaryMetrics:
    ce: float
    roc_auc: float
    accuracy: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

def binary_metrics(probs: np.ndarray, y: np.ndarray) -> BinaryMetrics: ...
def deltas(before: BinaryMetrics, after: BinaryMetrics) -> dict[str, float]: ...
```

`evaluate_context` for label-flip ranking must also expose target-embedding gradients (separate helper `target_embedding_influence_scores(clf, X_ctx, y_ctx, X_test, y_test, eligible) -> np.ndarray` of shape `[n_ctx]`, non-eligible filled with `-inf`).

---

## Tests (TDD: write failing tests first)

Run: `conda run -n tabfm python -m pytest tests/tabfm_experiments -q`

Minimum cases:

1. **metrics:** hand-made probs `[[0.9,0.1],[0.2,0.8]]`, y `[0,1]` → exact CE, AUC=1, acc=1, F1=1, CM.
2. **sampling:** `k_from_percent(100, 5)==5`; with `attack_class=1` all sampled indices have y=1; two seeds differ; without-replacement unique.
3. **aggregate:** 2×2 fake trials; overall mean vs per-run mean.
4. **fingerprints:** duplicate rows share a fingerprint; val/test exclusion removes them from a toy pool.
5. **nesting:** 1000 ⊂ 5000 ⊂ winner.
6. **LCLD:** selected-style frame has no `issue_d`.
7. **fix_types:** int delta 3.7 → 3.0; cat 1.4 → 1.0; immutable restored; no-int early-return leaves cats unrounded (document; unit-test the helper).
8. **scaler round-trip:** metadata min/max, `to_raw(to_scaled(x)) ≈ x`.
9. **relations:** one URL constraint (`Feature(1) <= Feature(0)`) violated vs satisfied; WiDS min≤max pair; one LCLD named `open_acc <= total_acc`.
10. **capgd toy:** maximize `sum(x)` on `[0,1]^d` with Linf eps — iterate matches a frozen snapshot from `check_capgd_equiv.py` or a few hand-checked steps (sign grad, clamp). Restart with same seed → same `x_best`.
11. **runtime CPU smoke (tiny):** 32×4 random X, 8 test, `n_estimators=1`; two `evaluate_context` calls with `need_grad=False` stay close (`max |Δp| < 1e-4` on CPU). `need_grad=True` yields nonzero `grad` w.r.t. X and **zero/None** useful grad w.r.t. integer Y if you expose it.
12. **influence:** on a 2-class toy where flipping a known row clearly raises CE, that row ranks in the top-k (CPU, small n).

GPU integration (not in default pytest, a script or pytest mark `gpu`): 20-row context, 16 test, 2 CAPGD steps, `--constraints box`, finishes; label-flip k=1 finishes.

---

## Task list

### Task 1: Package skeleton + metrics + io + sampling

**Files:** create `scripts/tabfm_experiments/{__init__,config,metrics,io,sampling,data}.py` and tests listed above for those units.

- [ ] Write failing tests for metrics, sampling, fingerprints, jsonable.
- [ ] Implement until `conda run -n tabfm python -m pytest tests/tabfm_experiments/test_metrics.py tests/tabfm_experiments/test_sampling.py -q` passes.

`data.py` must load CSV, split X/y using `split_info.json` target if `--target` omitted, read metadata, compute fingerprints.

### Task 2: Constraints + type repair

**Files:** `scripts/tabfm_experiments/constraints.py`, `tests/tabfm_experiments/test_constraints.py`

- [ ] Failing tests for fix_types / immutable / scaler / one constraint per dataset.
- [ ] Wrap TabularBench: `build_constraints(dataset_name, metadata_df, feature_columns) -> Constraints`.
- [ ] `repair_end(x_clean, x_adv, constraints)` = fix_types + fix_immutable + fix_equality_constraints.
- [ ] PYTHONPATH includes `/home/ssn899/Desktop/TabFM/tabularbench` if `import tabularbench` fails.

### Task 3: CAPGD optimiser

**Files:** `scripts/tabfm_experiments/capgd.py`, `tests/tabfm_experiments/test_capgd.py`

- [ ] Move/adapt `capgd()` from `scripts/attack_context.py` (already verified vs TabularBench).
- [ ] Point `scripts/check_capgd_equiv.py` at the new function; run it.
- [ ] Command: `conda run -n tabfm python scripts/check_capgd_equiv.py`

### Task 4: Runtime

**Files:** `scripts/tabfm_experiments/runtime.py`, `tests/tabfm_experiments/test_runtime_cpu.py`

- [ ] `build_tabpfn_v2` as specified.
- [ ] `evaluate_context` refits every call; optional test chunking; `need_grad` uses `torch.autograd.grad` on **X_ctx** (full tensor; caller can mask).
- [ ] Best-effort vs strict switches.
- [ ] Tiny CPU test. Skip GPU here.

### Task 5: X_train attack runner + CLI

**Files:** `attacks.py`, `aggregate.py`, rewrite `scripts/attack_context.py`, `scripts/infer.py`

- [ ] One trial: sample rows, mask, CAPGD in scaled space, end repair, metrics, artifacts.
- [ ] Loop n_runs × n_row_subsamples.
- [ ] Infer CLI uses `evaluate_context`.
- [ ] GPU smoke (manual): `--max-test 32 --n-iter 2 --n-runs 1 --n-row-subsamples 1 --row-percent 100` on `context_1000` if memory allows, else a 64-row slice written under `/tmp`. Check GPU 1/3 first:

```bash
nvidia-smi
conda run -n tabfm python scripts/attack_context.py --gpu 1 \
  --attack x-capgd --train ... --test ... --out /tmp/tabfm_smoke_x \
  --n-iter 2 --n-runs 1 --n-row-subsamples 1 --row-percent 5 --max-test 32
```

### Task 6: Influence label-flip

**Files:** `attacks.py` helper + CLI `--attack label-flip`

- [ ] Test on CPU toy that ranking prefers a high-influence row.
- [ ] Implement embedding-space scores; discrete flip; re-eval.
- [ ] Tiny GPU smoke analog.

### Task 7: Context selector

**Files:** `scripts/select_contexts.py`, `tests/tabfm_experiments/test_select_contexts.py`

- [ ] Tests with tiny fake frames (no TabPFN): exclusion, dedup, nesting, balanced counts, LCLD column drop.
- [ ] Real run uses TabPFN on `val_2000` — GPU. After unit tests pass, run:

```bash
conda run -n tabfm python scripts/select_contexts.py --gpu 1
```

- [ ] Validate manifests (unique rows, reserved exclusion, class counts, nesting, no `issue_d` on LCLD).

If a 10k score OOMs, chunk **validation** rows the same way as test chunking (exact).

### Task 8: Docs

**File:** `docs/context_poisoning.md`

Must include:

- Paper vs code defaults (ε-margin 0.05 is inherited, not in Djilani text)
- X vs Y encoding and why Y PGD is impossible on v2
- Why categorical X is still PGD-able (numeric codes; round at end)
- Residual stochasticity (best-effort vs strict)
- Constraints per dataset and Feature index vs name
- Why CAA is omitted
- Transductive caveat
- Output schema
- Copy-paste CLI examples for infer, x-capgd, label-flip, select_contexts
- Note that existing `splits/context_*.csv` are **not** the selected ones; experiments should use `splits/selected/...`

---

## Existing code you may copy

- CAPGD loop and `fix_types_raw`: `scripts/attack_context.py`
- Equivalence harness: `scripts/check_capgd_equiv.py`
- Split construction (do not rerun unless needed): `scripts/make_splits.py`
- TabularBench CAPGD: `tabularbench/tabularbench/attacks/capgd/capgd.py`

Do **not** copy `StatFreezer` into the new runtime.

---

## Environment notes

- Default system `python` may lack torch. Always `conda run -n tabfm`.
- `tabpfn` is the local checkout. Do not pip-install a different TabPFN.
- Sandbox/CI without GPU: CPU tests only.
- WiDS 109-col backward can OOM; `--test-batch-size` (e.g. 64 or 128) is required on smokes if 93GB is not free. Chunking must not change the loss vs unchunked (assert in a unit test with a stub evaluate or a tiny model).

---

## Success criteria

1. `pytest tests/tabfm_experiments` passes in `tabfm`.
2. `check_capgd_equiv.py` still agrees with TabularBench on the toy objective.
3. `splits/selected/{natural,balanced}/` exist for all three datasets with valid manifests.
4. `docs/context_poisoning.md` written.
5. Infer / x-capgd / label-flip CLIs run a **tiny** GPU smoke each.
6. No full 5×5 attack grid in the logs.
7. No TabPFNv2.5/v3/TabICL in the default path.

#!/usr/bin/env python
"""Context gating: learn one gate g_i in [0, 1] per context row, on its FEATURES only.

TabPFNv2 turns every row into feature-group inputs u_i = [values | missing indicators]
(groups x 4 on gas-drift) and feeds them to a linear encoder, feature_group_embedder,
before any attention layer. While the gate is active, context row i's input becomes

    u_i(g) = g_i * u_i + (1 - g_i) * u_null

u_null is the input TabPFN builds for a row whose features are ALL missing (values 0,
indicators -2 on gas-drift). It is captured from TabPFN itself, not hard-coded.

- g_i = 1 leaves the row unchanged.
- g_i = 0 makes its features entirely missing while its label stays in the context.
  This is the "feat-gate" oracle of nan_oracle_experiment.py, which recovered all of
  the loss from hub-flip poisoning.
- Labels and query rows are never touched.
- The encoder is linear, so the gate interpolates the row's embedding.
- Every statistic TabPFN fits (imputation means, scaler, constant-feature masks) is
  computed before the encoder, from the ungated rows, so none of them moves with g.

The validation set is a separate clean file (--val) by default. With --val-from-context
N it is instead a class-stratified random draw of N rows from the CLEAN context (seed
--val-seed); the remaining rows are the context, and only they are poisoned and gated.
One seed gives nested draws: the N = 200 draw contains the N = 100 one.

The gates are learned on the whole clean validation set. Every gate starts at 1 (no
row is known to be poisoned), and the objective is

    min_{g in [0,1]^n}  CE_val(g)

solved with one of

    pgd    g <- clip(g - eta * grad, 0, 1)
           --pgd-scale linf: eta = lr / max|grad_i| over the rows that can move (not
           pinned at 0 or 1 by the clip), so the largest change of any gate is lr.
           --pgd-scale none: eta = lr.
    adam   torch.optim.Adam on g, then clip(g, 0, 1) after every step.

No limit on how many rows may be gated (the R% condition) is enforced during learning.

--gate topk replaces the soft gates by exactly k = remove-percent (default R) of the rows
switched off. Each row has a free score S_i (N(0, --score-init-std^2) at the start, seed
--init-seed; one seed is one restart). Every iteration the gate is the hard mask
x = topk_mask(S): 0 for the k lowest scores, 1 for the rest. The gradient of the loss with
respect to that 0/1 gate reaches the scores through a straight-through estimator (--ste):
d loss/dS = d loss/dx * surrogate'(S), surrogate = S itself (identity), sigmoid(S), or a
sigmoid centred on the keep/remove boundary. The scores then take the PGD step (no clip):
--pgd-scale linf moves the largest score by lr. log.csv adds score_std and mask_flips (rows
that entered or left the mask with the update); detection AUROC is computed from -S.
--select-on ce ranks iterations by CE on the validation(-test) rows instead of accuracy.

Test-set curves, to see where over-fitting starts. The test set is scored at iteration
0, every --test-every iterations, and at the last iteration, in two ways:

    soft   the current soft gates, exactly as in training.
    hard   the P% of context rows with the smallest gates are discarded, and TabPFN runs
           normally on the rest with no gate (P = --remove-percent, default R).
           Ties among equal gates are broken by a fixed random order (--tie-seed). So
           at iteration 0, where every gate is 1, this is a random P% removal.

The test set never influences the gates or the choice of the best iteration.

The top-k gate vectors by validation accuracy are kept (ties -> lower validation CE,
then the earlier iteration). With --val-holdout FRAC, a class-stratified FRAC of the
validation rows is held out as a validation-test set instead: the gates are fitted on
the rest, the held-out rows are scored on the test iterations, and the top-k are the test
iterations with the highest validation-test accuracy (ties -> lower validation-test CE,
then earlier). selection_rules in test_metrics.json also records which iteration other
stopping rules would pick, with the test scores logged there. After training, the best
gates are scored on the test set, soft and hard, next to these references:
- the clean context;
- the poisoned context, ungated;
- a random P% removal;
- two oracles that know the poisoned rows: g = 0 on exactly those rows, and those rows
  removed;
- the clean validation set alone as the context (val_as_context): gating only earns its
  keep if it beats simply using the clean rows;
- the poisoned context with the validation rows added (poisoned_plus_val).
Both use every validation row, held-out ones included.
vs_val_as_context in test_metrics.json is (best soft gates) - (val_as_context) per metric.

The poisoned-row mask is used only for these references and for the gate diagnostics,
never for learning.

Everything is logged to local files and to Weights & Biases (--wandb-project, optional
--wandb-group). The API key is read from the WANDB_API_KEY environment variable. W&B
sections, x-axis = iteration:
    <metric>/val                  every iteration (the gates' training objective)
    <metric>/test_soft, <metric>/test_hard   every --test-every iterations
    detection/precision_at_R, recall_at_R, auroc, ap   every iteration
    gates/..., optim/...          every iteration
    overlay/<metric>              at the end: validation and both test curves together,
                                  with val_as_context as a flat line

    python scripts/gate_context.py --gpu 1 --dataset gas-drift --attack hub-flip-local \
        --row-percent 20 --optimizer pgd --lr 0.05 --steps 300 --run-name my_run

Outputs go to results/gate/<dataset>/<attack>_r<R>/<run-name>/:

    config.json        arguments, data sizes, the captured u_null
    run.log            the console log
    log.csv            one row per iteration; iteration t = after t updates (0 = ungated).
                       Columns:
                         val_<metric>   at this iteration's gates (the fitted rows)
                         valtest_<metric>
                                        the held-out rows (--val-holdout), only on
                                        test iterations
                         test_soft_<metric>, test_hard_<metric>
                                        only on test iterations
                         g_mean, g_min, frac_g_below_0.5
                         g_mean_poisoned, g_mean_clean
                         precision_at_R, recall_at_R, auroc_poisoned, ap_poisoned
                                        detection diagnostics, every iteration, only when
                                        R > 0 (see gate_stats). precision_at_R is the
                                        share of the rows the hard evaluation discards
                                        that are poisoned.
                         grad_max_abs   max |d CE / d g| at this iteration's gates
                         max_abs_dg     largest gate change in the update that follows
    plots/<metric>.png each metric vs iteration (ce is the loss): validation, test soft,
                       test hard; dashed = oracle references
    plots/gates.png, plots/detection.png   gate and detection diagnostics vs iteration
    plots/gate_hist_best.png   the best gates, clean vs poisoned rows
    gates/top<k>.npz   the top-k gate vectors (g) with their log row; gates/final.npz
    gates/masks.npz    --gate topk: the masked rows of every iteration (masked [steps+1, n])
    test_metrics.json  final test-set metrics of the best gates and the references
    test_probs.npz     the matching test-set probabilities
    poison.npz         poisoned rows, clean and poisoned labels, the tie-break order
                       (row indices into the context as gated, i.e. after any split)
    split.npz          with --val-from-context: val_rows and context_rows, row indices
                       into the context file; with --val-holdout also val_fit_rows and
                       val_holdout_rows (into the context file, or into the val file);
                       with --fit-subsample also val_fit_subsample_rows, the rows fitted
    wandb/             the local W&B run files
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tabfm_experiments.config import DATASETS_DIR, Seeds  # noqa: E402
from tabfm_experiments.data import load_split  # noqa: E402
from tabfm_experiments.io import jsonable, setup_logger, write_json  # noqa: E402
from tabfm_experiments.metrics import MULTICLASS_METRICS, multiclass_metrics  # noqa: E402
from tabfm_experiments.multiclass_attacks import ATTACKS, poison  # noqa: E402
from tabfm_experiments.runtime import (  # noqa: E402
    _forward, _install_recompute, build_tabpfn_v2, check_all_classes_present, device_from_gpu,
)
from tabfm_experiments.sampling import k_from_percent  # noqa: E402

METRICS = MULTICLASS_METRICS  # ce (the loss), accuracy, balanced_accuracy, mcc, roc_auc


# ---- the gate ------------------------------------------------------------------------

def capture_null_input(clf, Xc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
    """The feature-encoder input TabPFN builds for a row whose features are ALL missing.

    One all-NaN query row is scored against the context and its slice of the encoder
    input is read back, so padding and constant-column handling are TabPFN's own.
    Shape (feature groups, 2 * features per group) = [values | missing indicators].
    """
    seen = []
    with torch.no_grad():
        clf.fit_with_differentiable_input(Xc, yc)
        handle = clf.models_[0].feature_group_embedder.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            _forward(clf, torch.full((1, Xc.shape[1]), float("nan"), device=Xc.device),
                     need_grad=False)
        finally:
            handle.remove()
    if len(seen) != 1 or seen[0].shape[0] != len(Xc) + 1:
        raise RuntimeError(f"expected one encoder call with {len(Xc) + 1} rows, "
                           f"got {[tuple(s.shape) for s in seen]}")
    # The no-grad forward runs in inference mode; cloning outside it gives a normal
    # tensor that the gradient forwards can use.
    return seen[0][-1].clone()


@contextlib.contextmanager
def feature_gate(clf, g: torch.Tensor, null_x: torch.Tensor):
    """While active, context row i's feature-encoder input u_i becomes
    g_i * u_i + (1 - g_i) * null_x. Query rows and the label encoder are untouched.

    The encoder sees all rows of a forward in ONE call, context rows first and in input
    order (test_gate_context.py checks this); the hook refuses any other layout.
    """
    if len(clf.models_) != 1:
        raise RuntimeError("the gate assumes a single loaded model (n_estimators=1)")
    n, calls = len(g), []

    def hook(module, args):
        u = args[0]  # (rows, feature groups, 2 * features per group)
        if u.shape[0] <= n or u.shape[1:] != null_x.shape:
            raise RuntimeError(f"feature-encoder input {tuple(u.shape)} is not {n} context rows "
                               f"+ query rows, each {tuple(null_x.shape)}")
        calls.append(u.shape[0])
        gi = g.to(u.dtype)[:, None, None]
        gated = gi * u[:n] + (1 - gi) * null_x.to(u.dtype)
        return (torch.cat([gated, u[n:]]),) + tuple(args[1:])

    handle = clf.models_[0].feature_group_embedder.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
    if len(calls) != 1:
        raise RuntimeError(f"feature encoder ran {len(calls)} times in one forward; the gate expects 1")


def evaluate(clf, Xc, yc, Xq, yq, g, null_x, *, need_grad: bool, batch_size: int | None):
    """Mean CE of the query rows under the gated context, its gradient d CE / d g (if
    need_grad, else None) and the query probabilities [n_q, K] as numpy. g = None scores
    the context as it is, with no gate.

    Query rows are scored in chunks of batch_size. Each chunk refits on the full context
    and contributes sum CE / n_q, which is exact because query rows never attend to each
    other. Same pattern as runtime.evaluate_context.
    """
    n_q = len(yq)
    bs = min(batch_size or n_q, n_q)
    total, grad, probs = 0.0, (torch.zeros_like(g) if need_grad else None), []
    for st in range(0, n_q, bs):
        with torch.enable_grad() if need_grad else torch.no_grad():
            clf.fit_with_differentiable_input(Xc, yc)
            _install_recompute(clf)
            with feature_gate(clf, g, null_x) if g is not None else contextlib.nullcontext():
                p = _forward(clf, Xq[st:st + bs], need_grad=need_grad)
            loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), yq[st:st + bs].to(p.device),
                              reduction="sum") / n_q
        if need_grad:
            grad += torch.autograd.grad(loss, g)[0]
        total += float(loss.detach())
        probs.append(p.detach().float().cpu())
    return total, grad, torch.cat(probs).numpy()


def lowest_gated(g: np.ndarray, k: int, tie_order: np.ndarray) -> np.ndarray:
    """Sorted indices of the k rows with the smallest gates.

    Ties among equal gates are broken by tie_order, a fixed random permutation, so the
    choice is random but reproducible. At iteration 0, where every gate is 1, these are
    k random rows.
    """
    return np.sort(np.lexsort((tie_order, g))[:k])  # primary key g, then tie_order


# ---- the update ----------------------------------------------------------------------

def pgd_step(g: torch.Tensor, grad: torch.Tensor, lr: float, scale: str) -> None:
    """In place: g <- clip(g - eta * grad, 0, 1). See the module docstring for eta."""
    with torch.no_grad():
        if scale == "linf":
            pinned = ((g <= 0) & (grad > 0)) | ((g >= 1) & (grad < 0))  # the clip undoes their step
            free = grad[~pinned].abs()
            if free.numel() == 0 or float(free.max()) == 0.0:
                return
            eta = lr / free.max()
        else:
            eta = lr
        g.sub_(eta * grad).clamp_(0.0, 1.0)


def adam_step(g: torch.Tensor, grad: torch.Tensor, opt: torch.optim.Optimizer) -> None:
    """In place: one Adam step on g, then clip to [0, 1]."""
    g.grad = grad
    opt.step()
    with torch.no_grad():
        g.clamp_(0.0, 1.0)
    g.grad = None


# ---- the top-k gate ------------------------------------------------------------------

def init_scores(n: int, std: float, seed: int, device) -> torch.Tensor:
    """N(0, std^2) scores, drawn on the CPU from torch.Generator().manual_seed(seed), so a
    restart is the same on every device."""
    return (torch.randn(n, generator=torch.Generator().manual_seed(seed)) * std).to(device)


def topk_mask(scores: torch.Tensor, k_remove: int) -> torch.Tensor:
    """0 for the k_remove rows with the lowest scores, 1 for the rest; ties go to the
    earlier row (stable sort)."""
    mask = torch.ones_like(scores)
    if k_remove > 0:
        mask[torch.argsort(scores, stable=True)[:k_remove]] = 0.0
    return mask


def ste_grad(grad_g: torch.Tensor, scores: torch.Tensor, kind: str, *, k_remove: int,
             temp: float = 1.0) -> torch.Tensor:
    """d loss / d scores by the straight-through estimator: the forward pass used the hard
    mask, the backward pass treats it as surrogate(scores), whose derivative scales the
    gate gradient:
      identity          the scores themselves: d/dS = 1
      sigmoid           sigmoid(S): largest at S = 0
      sigmoid-centered  sigmoid((S - tau) / w), tau midway between the k_remove-th lowest
                        score and the next, w = temp * std(S): largest at the keep/remove
                        boundary
    """
    if kind == "identity":
        return grad_g
    if kind == "sigmoid":
        s = torch.sigmoid(scores)
        return grad_g * s * (1 - s)
    if kind == "sigmoid-centered":
        srt = torch.sort(scores).values
        tau = (srt[k_remove - 1] + srt[k_remove]) / 2
        w = temp * scores.std()
        s = torch.sigmoid((scores - tau) / w)
        return grad_g * s * (1 - s) / w
    raise ValueError(f"unknown straight-through surrogate {kind!r}")


def budget_at(t: int, k: int, warmup: int) -> int:
    """Rows masked at iteration t: a cubic ramp from 0 to k over the first `warmup`
    iterations, k * (1 - (1 - t / warmup)^3), then k (the sparsity schedule of gradual and
    movement pruning). warmup <= 0: k from the start."""
    if warmup <= 0 or t >= warmup:
        return k
    return int(round(k * (1 - (1 - t / warmup) ** 3)))


def lr_at(t: int, steps: int, lr: float, schedule: str, min_frac: float) -> float:
    """Step size of update t: constant, or decayed to min_frac * lr at the last update
    along a cosine or a straight line."""
    if schedule == "constant":
        return lr
    frac = t / steps
    if schedule == "cosine":
        return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
    if schedule == "linear":
        return lr * (1 - (1 - min_frac) * frac)
    raise ValueError(f"unknown step schedule {schedule!r}")


def score_step(scores: torch.Tensor, grad: torch.Tensor, lr: float, scale: str) -> None:
    """In place: scores <- scores - eta * grad, unbounded. linf: eta = lr / max|grad|, so
    the largest score change is lr; none: eta = lr."""
    with torch.no_grad():
        if scale == "linf":
            m = float(grad.abs().max())
            if m == 0.0:
                return
            eta = lr / m
        else:
            eta = lr
        scores.sub_(eta * grad)


# ---- bookkeeping ---------------------------------------------------------------------

def gate_stats(g: np.ndarray, poisoned: np.ndarray, removed: np.ndarray,
               scores: np.ndarray | None = None) -> dict:
    """Gate summary. With poisoned rows present, also how well the gates detect them:
      precision_at_R  share of the removed rows (the hard evaluation's lowest-gated P%)
                      that are poisoned
      recall_at_R     share of the poisoned rows among the removed rows (equal to the
                      precision when as many rows are removed as were poisoned)
      auroc_poisoned  AUROC of 1 - g against the poisoned mask (0.5 = chance); of -scores
                      when scores are given (top-k gates, whose g is only 0 or 1)
      ap_poisoned     average precision of the same (chance = the poisoned share)
    Diagnostics only: the poisoned mask never reaches the optimisation."""
    s = {"g_mean": float(g.mean()), "g_min": float(g.min()),
         "frac_g_below_0.5": float((g < 0.5).mean())}
    if poisoned.any() and not poisoned.all():
        from sklearn.metrics import average_precision_score, roc_auc_score
        hits = int(poisoned[removed].sum())
        rank = -scores if scores is not None else 1 - g  # higher = more likely poisoned
        s.update({"g_mean_poisoned": float(g[poisoned].mean()),
                  "g_mean_clean": float(g[~poisoned].mean()),
                  "precision_at_R": hits / len(removed) if len(removed) else math.nan,
                  "recall_at_R": hits / int(poisoned.sum()),
                  "auroc_poisoned": float(roc_auc_score(poisoned, rank)),
                  "ap_poisoned": float(average_precision_score(poisoned, rank))})
    return s


def prefixed(prefix: str, m: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in m.items()}


DETECTION = {"precision_at_R": "precision_at_R", "recall_at_R": "recall_at_R",
             "auroc_poisoned": "auroc", "ap_poisoned": "ap"}  # log.csv name -> W&B name
OPTIM = ("grad_max_abs", "max_abs_dg", "mask_flips", "time_s")


def wandb_keys(row: dict) -> dict:
    """log.csv names -> W&B names. Each metric gets its own section, so its validation
    curve, its validation-test curve (with --val-holdout) and its two test curves sit side
    by side (accuracy/val, accuracy/valtest, accuracy/test_soft, accuracy/test_hard). The
    other sections are detection/, gates/ and optim/."""
    out = {}
    for key, v in row.items():
        if key == "iter":
            continue
        for split in ("val", "valtest", "test_soft", "test_hard"):
            if key.startswith(split + "_") and key[len(split) + 1:] in METRICS:
                out[f"{key[len(split) + 1:]}/{split}"] = v
                break
        else:
            if key in DETECTION:
                out[f"detection/{DETECTION[key]}"] = v
            elif key in OPTIM:
                out[f"optim/{key}"] = v
            else:
                out[f"gates/{key}"] = v
    return out


def save_gate(path: Path, g: np.ndarray, row: dict, scores: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if scores is None else {"scores": scores.astype(np.float32)}
    np.savez_compressed(path, g=g.astype(np.float32), iteration=row["iter"],
                        log_row=json.dumps(jsonable(row)), **extra)


def plot_curves(rows: list[dict], refs: dict, best_iter: int, plot_dir: Path,
                baseline: tuple[str, dict] | None = None) -> None:
    """One figure per metric (validation every iteration, test soft and hard on the test
    iterations, oracle references dashed, the test score of `baseline` = (label, metrics)
    dash-dotted) plus the gate diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    styles = {  # split: (colour, marker, label)
        "val": ("#2a78d6", None, "validation (optimised)"),
        "valtest": ("#c4378f", "^", "validation-test (held out)"),
        "test_soft": ("#eb6834", "o", "test, soft gates"),
        "test_hard": ("#1baf7a", "s", "test, lowest-gated P% removed"),
    }
    ref_labels = {"val": "validation, oracle gate", "valtest": "validation-test, oracle gate",
                  "test_soft": "test, oracle gate",
                  "test_hard": "test, poisoned rows removed"}
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for split, (colour, marker, label) in styles.items():
            pts = [(r["iter"], r[f"{split}_{m}"]) for r in rows
                   if f"{split}_{m}" in r and not math.isnan(r[f"{split}_{m}"])]
            if pts:
                ax.plot(*zip(*pts), color=colour, marker=marker, ms=3, lw=1.3, label=label)
            if f"{split}_{m}" in refs:
                ax.axhline(refs[f"{split}_{m}"], color=colour, ls="--", lw=1, label=ref_labels[split])
        if baseline is not None:
            ax.axhline(baseline[1][m], color="#4d4d4d", ls="-.", lw=1.2, label=baseline[0])
        ax.axvline(best_iter, color="0.5", ls=":", lw=1, label=f"best (iteration {best_iter})")
        ax.set(xlabel="iteration (gate updates)", ylabel=m,
               title=("cross-entropy (the loss)" if m == "ce" else m) + " vs iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{m}.png", dpi=120)
        plt.close(fig)

    for name, keys, title in (
            ("gates", ("g_mean", "g_min", "frac_g_below_0.5", "g_mean_poisoned", "g_mean_clean"),
             "gates vs iteration"),
            ("detection", tuple(DETECTION), "detection of the poisoned rows vs iteration")):
        keys = [k for k in keys if k in rows[0]]
        if not keys:  # no poisoned rows: nothing to detect
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k in keys:
            ax.plot([r["iter"] for r in rows], [r[k] for r in rows], label=k)
        ax.axvline(best_iter, color="0.5", ls=":", lw=1)
        ax.set(xlabel="iteration (gate updates)", ylim=(-0.02, 1.02), title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=120)
        plt.close(fig)


def plot_gate_hist(g: np.ndarray, poisoned: np.ndarray, path: Path, title: str,
                   xlabel: str = "gate g") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_unit = g.min() >= 0 and g.max() <= 1
    bins = np.linspace(0, 1, 41) if in_unit else np.linspace(g.min(), g.max(), 41)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(g[~poisoned], bins=bins, alpha=0.6, color="#2a78d6", label=f"clean rows ({(~poisoned).sum()})")
    if poisoned.any():
        ax.hist(g[poisoned], bins=bins, alpha=0.6, color="#eb6834", label=f"poisoned rows ({poisoned.sum()})")
    ax.set(xlabel=xlabel, ylabel="rows (log scale)", yscale="log", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@contextlib.contextmanager
def wandb_run(a, run_name: str, out: Path, config: dict):
    """The W&B run for the whole training (None with --wandb-mode disabled). It is
    marked failed if the training raises."""
    if a.wandb_mode == "disabled":
        yield None
        return
    import wandb
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group,
                     name=run_name, mode=a.wandb_mode, dir=str(out), config=jsonable(config))
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()


def wandb_upload(run, out: Path, run_name: str, summary: dict, rows: list[dict],
                 baseline: tuple[str, dict] | None = None) -> None:
    """Final summary, one interactive chart per metric with the validation and both test
    curves on the same axes (and `baseline` = (label, test metrics) as a flat line), the
    plots as images, and every output file as one artifact."""
    import wandb
    run.summary.update(jsonable(summary))
    charts = {}
    for m in METRICS:
        series = []
        for split, label in (("val", "validation (optimised)"), ("valtest", "validation-test (held out)"),
                             ("test_soft", "test, soft gates"), ("test_hard", "test, lowest-gated removed")):
            pts = [(r["iter"], r[f"{split}_{m}"]) for r in rows
                   if f"{split}_{m}" in r and not math.isnan(r[f"{split}_{m}"])]
            if pts:
                series.append((label, pts))
        if baseline is not None:
            series.append((baseline[0], [(rows[0]["iter"], baseline[1][m]), (rows[-1]["iter"], baseline[1][m])]))
        charts[f"overlay/{m}"] = wandb.plot.line_series(
            xs=[[x for x, _ in pts] for _, pts in series], ys=[[y for _, y in pts] for _, pts in series],
            keys=[label for label, _ in series], title=f"{m}: validation vs test", xname="iteration")
    charts |= {f"plots/{p.stem}": wandb.Image(str(p)) for p in sorted((out / "plots").glob("*.png"))}
    run.log(charts)
    art = wandb.Artifact(re.sub(r"[^A-Za-z0-9._-]", "-", f"gate-run-{run_name}"), type="gate-run")
    for p in sorted(out.rglob("*")):
        if p.is_file() and "wandb" not in p.relative_to(out).parts:
            art.add_file(str(p), name=str(p.relative_to(out)))
    run.log_artifact(art)


# ---- data ----------------------------------------------------------------------------

def find_split(dataset: str, prefix: str) -> Path:
    """The dataset's only <prefix>*.csv listed in its split_info.json."""
    d = DATASETS_DIR / dataset / "splits"
    names = [s for s in json.loads((d / "split_info.json").read_text())["splits"] if s.startswith(prefix)]
    if len(names) != 1:
        raise SystemExit(f"expected one {prefix}*.csv in {d}, found {names}; pass it explicitly "
                         "(--context / --val / --test)")
    return d / names[0]


def load(path: Path, device: str) -> tuple[torch.Tensor, np.ndarray]:
    X, y, _ = load_split(path)
    return torch.tensor(X.to_numpy(np.float32), device=device), np.asarray(y).astype(int)


def draw_validation(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted indices of a class-stratified random draw of n rows.

    Class quotas by largest remainder: floor every class's share, then the leftover rows
    go to the largest fractional parts (the rule of prepare_multiclass_splits). Within a
    class the rows follow one random order fixed by the seed and independent of n, so
    for one seed a larger draw contains every smaller one whenever no class's quota
    shrinks as n grows.
    """
    if not 0 < n <= len(y):
        raise ValueError(f"cannot draw {n} rows from {len(y)}")
    classes, counts = np.unique(y, return_counts=True)
    share = n * counts / counts.sum()
    quota = np.floor(share).astype(int)
    quota[np.argsort(-(share - quota), kind="stable")[: n - int(quota.sum())]] += 1
    rng = np.random.default_rng(seed)
    order = [rng.permutation(np.flatnonzero(y == c)) for c in classes]
    return np.sort(np.concatenate([o[:q] for o, q in zip(order, quota)]))


def split_holdout(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(fit, holdout): sorted positions in y of a class-stratified random split that holds
    out round(frac * len(y)) rows (half up), drawn as in draw_validation."""
    n_hold = int(math.floor(frac * len(y) + 0.5))
    if not 0 < n_hold < len(y):
        raise ValueError(f"holding out {frac:g} of {len(y)} rows leaves one side empty")
    hold = draw_validation(y, n_hold, seed)
    return np.setdiff1d(np.arange(len(y)), hold), hold


def selection_rules(rows: list[dict], holdout: bool) -> dict:
    """The iteration each stopping rule picks, with the scores logged there.

    The candidates are the test iterations, the only ones with test (and validation-test)
    scores; rows are log rows, in iteration order.
      valtest_acc      highest validation-test accuracy, ties -> lower validation-test CE,
                       then earlier: how the best gates are chosen with --val-holdout
      valtest_ce       lowest validation-test CE, ties -> earlier
      val_acc          highest accuracy on the fitted validation rows, ties -> lower CE,
                       then earlier: the rule without a holdout, on the test iterations
      val_first_max    the first test iteration at or after the one where accuracy on the
                       fitted rows first reaches its maximum
      last             the last iteration
      test_acc_oracle  highest test accuracy, ties -> lower test CE, then earlier. It looks
                       at the test set: a ceiling to compare against, not a usable rule.
    """
    cand = [r for r in rows if "test_soft_ce" in r]
    best = lambda key: min(cand, key=key)  # noqa: E731
    picked = {}
    if holdout:
        picked["valtest_acc"] = best(lambda r: (-r["valtest_accuracy"], r["valtest_ce"], r["iter"]))
        picked["valtest_ce"] = best(lambda r: (r["valtest_ce"], r["iter"]))
    picked["val_acc"] = best(lambda r: (-r["val_accuracy"], r["val_ce"], r["iter"]))
    top = max(r["val_accuracy"] for r in rows)
    first = next(r["iter"] for r in rows if r["val_accuracy"] == top)
    picked["val_first_max"] = best(lambda r: (r["iter"] < first, r["iter"]))
    picked["last"] = max(cand, key=lambda r: r["iter"])
    picked["test_acc_oracle"] = best(lambda r: (-r["test_soft_accuracy"], r["test_soft_ce"], r["iter"]))
    splits = ("val", "valtest", "test_soft", "test_hard") if holdout else ("val", "test_soft", "test_hard")
    return {rule: {"iteration": r["iter"]} | {s: {m: r[f"{s}_{m}"] for m in METRICS if f"{s}_{m}" in r}
                                             for s in splits}
            for rule, r in picked.items()}


# ---- main ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1", help="GPU index, or 'cpu'")
    # data and poisoning
    ap.add_argument("--dataset", default="gas-drift")
    ap.add_argument("--context", type=Path, default=None, help="default: the dataset's context_*.csv")
    ap.add_argument("--val", type=Path, default=None,
                    help="clean validation set, all of it optimised on; default: the dataset's val_*.csv")
    ap.add_argument("--val-from-context", type=int, default=None, metavar="N",
                    help="instead of --val, draw N class-stratified random rows of the CLEAN context "
                         "as the validation set; the other rows are the context that is poisoned and gated")
    ap.add_argument("--val-seed", type=int, default=0,
                    help="seed of the --val-from-context draw; one seed gives nested draws across N")
    ap.add_argument("--val-holdout", type=float, default=0.0, metavar="FRAC",
                    help="hold out this fraction of the validation rows (class-stratified) as a "
                         "validation-test set: the gates are fitted on the rest, it is scored on the "
                         "test iterations, and the best iteration is the one with its highest accuracy")
    ap.add_argument("--fit-subsample", type=float, default=1.0, metavar="FRAC",
                    help="fit on a class-stratified random FRAC of the fitted validation rows only "
                         "(bagging: restarts with different --fit-subsample-seed see different rows); "
                         "1 = all of them")
    ap.add_argument("--fit-subsample-seed", type=int, default=None,
                    help="seed of the --fit-subsample draw (default: --init-seed)")
    ap.add_argument("--test", type=Path, default=None, help="default: the dataset's test_attack_*.csv")
    ap.add_argument("--n-classes", type=int, default=None,
                    help="default: n_classes in split_info.json, else the classes in the context")
    ap.add_argument("--attack", default="hub-flip-local", choices=ATTACKS)
    ap.add_argument("--row-percent", type=float, default=20.0,
                    help="R, percent of context rows poisoned (0 = clean context)")
    ap.add_argument("--run-id", type=int, default=0, help="poison draw, for the stochastic attacks")
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed)
    ap.add_argument("--hub-metric", default="knn", choices=["knn", "all"])
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--target-k", type=int, default=10)
    # optimisation
    ap.add_argument("--optimizer", default="pgd", choices=["pgd", "adam"])
    ap.add_argument("--lr", type=float, default=None,
                    help="pgd with --pgd-scale linf: largest change of any gate per step; "
                         "pgd with none: plain step size; adam: Adam's learning rate "
                         "(default: 0.05 for pgd, 0.01 for adam)")
    ap.add_argument("--steps", type=int, default=300, help="number of gate updates")
    ap.add_argument("--pgd-scale", default="linf", choices=["linf", "none"])
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--gate", default="soft", choices=["soft", "topk"],
                    help="soft: one gate in [0, 1] per row; topk: one free score per row, the forward "
                         "pass masks exactly the remove-percent rows with the lowest scores (gate 0, "
                         "the rest 1), and the gradient reaches the scores straight through (--ste)")
    ap.add_argument("--ste", default="identity", choices=["identity", "sigmoid", "sigmoid-centered"],
                    help="topk: surrogate of the hard mask in the backward pass (see ste_grad)")
    ap.add_argument("--ste-temp", type=float, default=1.0,
                    help="topk, sigmoid-centered: surrogate width in standard deviations of the scores")
    ap.add_argument("--score-init-std", type=float, default=1.0,
                    help="topk: the scores start as N(0, std^2)")
    ap.add_argument("--init-seed", type=int, default=0,
                    help="topk: seed of the initial scores; one seed = one restart")
    ap.add_argument("--score-init", default="random", choices=["random", "gradient", "file"],
                    help="topk: random = N(0, std^2); gradient = -(d CE / d g at g = 1 on the fitted "
                         "validation rows) scaled to unit spread, plus N(0, std^2) noise, so the first "
                         "mask is the rows whose masking lowers the loss most; file = the per-row values "
                         "of --score-init-path (e.g. soft gates; low = masked first), plus the noise")
    ap.add_argument("--score-init-path", type=Path, default=None,
                    help="topk, --score-init file: an .npz with 'scores' or 'g', one value per context row")
    ap.add_argument("--ste-probe", type=float, default=0.0, metavar="ALPHA",
                    help="topk: take the straight-through gradient at a relaxed point (kept rows at "
                         "1 - ALPHA, masked rows at ALPHA) instead of at the hard mask; the loss and "
                         "every metric still use the hard mask (0 = the plain straight-through gradient)")
    ap.add_argument("--swap-search", type=int, default=0, metavar="PAIRS",
                    help="topk: instead of moving scores, swap PAIRS kept rows with the largest gradient "
                         "for PAIRS masked rows with the most negative one, each sampled from the top "
                         "4 x PAIRS of its side, and keep the swap only if the fitted-row CE drops; the "
                         "pair count halves after a rejection (up to --line-search tries, default 3) and "
                         "doubles back after a first-try success (0 = off)")
    ap.add_argument("--line-search", type=int, default=0, metavar="TRIES",
                    help="topk: keep a step only if the new mask lowers the CE on the fitted rows; "
                         "otherwise halve the step and retry, up to TRIES times (0 = off). The step "
                         "multiplier is remembered, doubling back towards 1 after a first-try success")
    ap.add_argument("--budget-warmup", type=float, default=0.0, metavar="FRAC",
                    help="topk: ramp the number of masked rows from 0 to k over this fraction of the "
                         "updates (cubic); only full-budget iterations can be chosen as the best")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine", "linear"],
                    help="topk: step size over the updates")
    ap.add_argument("--lr-min-frac", type=float, default=0.0,
                    help="topk: the decayed schedules end at this fraction of --lr")
    ap.add_argument("--select-on", default="acc", choices=["acc", "ce"],
                    help="rank iterations on the validation (or validation-test) rows by accuracy "
                         "(ties: lower CE) or by CE (ties: higher accuracy); then the earlier one")
    ap.add_argument("--g-init", type=float, default=1.0,
                    help="initial value of every gate (one value for all rows)")
    ap.add_argument("--top-k", type=int, default=3, help="how many best gate vectors to keep")
    # test-set curves
    ap.add_argument("--test-every", type=int, default=5,
                    help="score the test set (soft and hard) every N iterations; 0 = only at the end")
    ap.add_argument("--remove-percent", type=float, default=None,
                    help="P: the hard evaluation discards the P%% lowest-gated rows (default: R)")
    ap.add_argument("--tie-seed", type=int, default=0,
                    help="seed of the random order that breaks ties among equal gates")
    # runtime
    ap.add_argument("--batch-size", type=int, default=None,
                    help="query rows per forward (each chunk re-runs the full context, so this "
                         "saves little memory); default: all at once")
    ap.add_argument("--model-seed", type=int, default=Seeds.model_seed)
    ap.add_argument("--no-recompute", action="store_true",
                    help="disable activation checkpointing (faster, much more GPU memory)")
    ap.add_argument("--plot-every", type=int, default=10, help="redraw the plots every N iterations")
    ap.add_argument("--out", type=Path, default=Path("results/gate"))
    ap.add_argument("--run-name", default=None,
                    help="output folder and W&B run name; default: "
                         "<dataset>_<attack>_R<R>_<optimizer>[-<pgd-scale>]_lr<lr>_it<steps>")
    ap.add_argument("--force", action="store_true", help="overwrite a finished run")
    # Weights & Biases
    ap.add_argument("--wandb-project", default="TFM_experiments")
    ap.add_argument("--wandb-entity", default=None, help="default: the API key's default entity")
    ap.add_argument("--wandb-group", default=None, help="W&B run group (e.g. one per sweep)")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    a = ap.parse_args(argv)
    if a.lr is None:
        a.lr = 0.05 if a.optimizer == "pgd" else 0.01
    if a.remove_percent is None:
        a.remove_percent = a.row_percent
    if not 0 <= a.g_init <= 1:
        raise SystemExit("--g-init must be in [0, 1]")
    if a.steps < 0 or a.top_k < 1 or a.test_every < 0:
        raise SystemExit("--steps and --test-every must be >= 0, --top-k >= 1")
    if a.val_from_context is not None:
        if a.val is not None:
            raise SystemExit("--val-from-context draws the validation set from the context; "
                             "do not also pass --val")
        if a.val_from_context < 1:
            raise SystemExit("--val-from-context must be >= 1")
    if a.gate == "topk":
        if a.optimizer != "pgd":
            raise SystemExit("--gate topk moves its scores with the PGD step (--pgd-scale linf or none), "
                             "not with Adam")
        if a.remove_percent <= 0:
            raise SystemExit("--gate topk masks the remove-percent rows; it must be > 0")
        if a.score_init_std < 0 or (a.score_init == "random" and a.score_init_std == 0):
            raise SystemExit("--score-init-std must be > 0 (>= 0 with --score-init gradient)")
        if not 0 <= a.budget_warmup < 1:
            raise SystemExit("--budget-warmup must be in [0, 1)")
        if a.score_init == "file" and (a.score_init_path is None or not a.score_init_path.exists()):
            raise SystemExit("--score-init file needs an existing --score-init-path")
        if a.line_search < 0:
            raise SystemExit("--line-search must be >= 0")
        if not 0 <= a.ste_probe < 0.5:
            raise SystemExit("--ste-probe must be in [0, 0.5)")
        if a.swap_search < 0 or (a.swap_search and a.budget_warmup):
            raise SystemExit("--swap-search needs PAIRS >= 0 and no --budget-warmup")
    if not 0 <= a.val_holdout < 1:
        raise SystemExit("--val-holdout must be in [0, 1)")
    if not 0 < a.fit_subsample <= 1:
        raise SystemExit("--fit-subsample must be in (0, 1]")
    if a.val_holdout > 0 and a.test_every == 0:
        raise SystemExit("--val-holdout needs --test-every > 0: the held-out rows are scored, and the "
                         "best iteration chosen, on the test iterations")

    run_name = a.run_name or (f"{a.dataset}_{a.attack}_R{a.row_percent:g}"
                              + (f"_V{a.val_from_context}s{a.val_seed}" if a.val_from_context else "")
                              + (f"_H{round(100 * a.val_holdout)}" if a.val_holdout else "")
                              + (f"_topk-{a.ste}-std{a.score_init_std:g}-init{a.init_seed}"
                                 if a.gate == "topk" else "")
                              + f"_{a.optimizer}" + (f"-{a.pgd_scale}" if a.optimizer == "pgd" else "")
                              + f"_lr{a.lr:g}_it{a.steps}")
    out = a.out / a.dataset / f"{a.attack}_r{int(round(a.row_percent)):03d}" / run_name
    if (out / "test_metrics.json").exists() and not a.force:
        print(f"[skip] {out}/test_metrics.json exists (use --force)")
        return None
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "gates", ignore_errors=True)  # no stale top-k files from an earlier run
    log = setup_logger(out, "gate_context").info
    device = device_from_gpu(a.gpu)

    # -- data: poison the context labels; the features are never changed ------------------
    ctx_p = a.context or find_split(a.dataset, "context_")
    test_p = a.test or find_split(a.dataset, "test_attack_")
    Xc, yc = load(ctx_p, device)
    Xt, yt = load(test_p, device)
    info_p = DATASETS_DIR / a.dataset / "splits" / "split_info.json"
    k_info = json.loads(info_p.read_text()).get("n_classes") if info_p.exists() else None
    K = int(a.n_classes or k_info or len(np.unique(yc)))
    split = {}  # row indices, saved to split.npz
    if a.val_from_context is None:
        val_p = a.val or find_split(a.dataset, "val_")
        Xv, yv = load(val_p, device)
        validation = {"source": "file", "path": val_p}
    else:  # carve the validation rows out of the clean context, before any poisoning
        val_p, n_file = None, len(yc)
        if a.val_from_context >= n_file:
            raise SystemExit(f"--val-from-context {a.val_from_context} leaves no context: "
                             f"{ctx_p.name} has {n_file} rows")
        val_rows = draw_validation(yc, a.val_from_context, a.val_seed)
        if len(np.unique(yc[val_rows])) < K:
            raise SystemExit(f"--val-from-context {a.val_from_context} misses a class (class counts "
                             f"{np.bincount(yc[val_rows], minlength=K).tolist()}); draw more rows")
        ctx_rows = np.setdiff1d(np.arange(n_file), val_rows)
        split |= {"val_rows": val_rows, "context_rows": ctx_rows, "val_seed": a.val_seed}
        Xv, yv = Xc[torch.as_tensor(val_rows, device=device)], yc[val_rows]
        Xc, yc = Xc[torch.as_tensor(ctx_rows, device=device)], yc[ctx_rows]
        validation = {"source": "class-stratified random draw from the clean context",
                      "n": a.val_from_context, "seed": a.val_seed, "n_context_file": n_file,
                      "nested_across_n": True, "rows": "split.npz"}
    Xv_all, yv_all = Xv, yv  # every clean validation row; the baselines use all of them
    holdout = a.val_holdout > 0
    if holdout:  # fit the gates on part of the validation rows, choose the iteration on the rest
        fit_pos, hold_pos = split_holdout(yv, a.val_holdout, a.val_seed)
        for pos, what in ((fit_pos, "fitted"), (hold_pos, "held-out")):
            if len(np.unique(yv[pos])) < K:
                raise SystemExit(f"--val-holdout {a.val_holdout:g}: the {what} validation rows miss a class "
                                 f"({np.bincount(yv[pos], minlength=K).tolist()}); use more validation rows")
        base = split.get("val_rows", np.arange(len(yv)))  # context-file rows, or val-file rows
        split |= {"val_fit_rows": base[fit_pos], "val_holdout_rows": base[hold_pos],
                  "holdout_fraction": a.val_holdout}
        Xh, yh = Xv[torch.as_tensor(hold_pos, device=device)], yv[hold_pos]
        Xv, yv = Xv[torch.as_tensor(fit_pos, device=device)], yv[fit_pos]
        validation |= {"holdout_fraction": a.val_holdout, "n_fit": len(fit_pos),
                       "n_holdout": len(hold_pos), "rows": "split.npz"}
    if a.fit_subsample < 1:  # bagging: fit on a random part of the fitted rows
        sub_seed = a.init_seed if a.fit_subsample_seed is None else a.fit_subsample_seed
        sub = draw_validation(yv, max(1, int(math.floor(a.fit_subsample * len(yv) + 0.5))), sub_seed)
        if len(np.unique(yv[sub])) < K:
            raise SystemExit(f"--fit-subsample {a.fit_subsample:g}: the subsample misses a class "
                             f"({np.bincount(yv[sub], minlength=K).tolist()})")
        base = split.get("val_fit_rows", split.get("val_rows", np.arange(len(yv))))
        split["val_fit_subsample_rows"] = base[sub]
        Xv, yv = Xv[torch.as_tensor(sub, device=device)], yv[sub]
        validation |= {"fit_subsample": a.fit_subsample, "fit_subsample_seed": sub_seed,
                       "n_fit": len(sub), "rows": "split.npz"}
    if split:
        np.savez_compressed(out / "split.npz", **split)
    for y in (yc, yv, yt):
        check_all_classes_present(y, K)
    n = len(yc)

    rows, y_pois = poison(a.attack, Xc, yc, K, a.row_percent, run_id=a.run_id, row_seed=a.row_seed,
                          hub_metric=a.hub_metric, hub_k=a.hub_k, target_k=a.target_k)
    check_all_classes_present(y_pois, K)
    poisoned = np.zeros(n, dtype=bool)
    poisoned[rows] = True
    k_remove = k_from_percent(n, a.remove_percent) if a.remove_percent > 0 else 0
    tie_order = np.random.default_rng(a.tie_seed).permutation(n)
    np.savez_compressed(out / "poison.npz", poisoned_rows=rows, y_clean=yc.astype(np.int16),
                        y_poisoned=y_pois.astype(np.int16), tie_order=tie_order)

    def as_t(y, dtype):
        return torch.as_tensor(y, dtype=dtype, device=device)

    yc_t, yp_t = as_t(yc, torch.float32), as_t(y_pois, torch.float32)
    yv_t, yt_t = as_t(yv, torch.long), as_t(yt, torch.long)
    yh_t = as_t(yh, torch.long) if holdout else None

    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%: {len(rows)} of {n} context rows poisoned; "
        f"{Xc.shape[1]} features, K={K}")
    val_txt = val_p.name if val_p is not None else f"drawn from the clean context, seed {a.val_seed}"
    val_txt += (f"; {len(yv)} fitted, {len(yh)} held out as validation-test, which picks the best "
                "iteration" if holdout else ", all optimised on")
    if a.fit_subsample < 1:
        val_txt += f" (bagging: a {a.fit_subsample:g} subsample of the fitted rows, seed {sub_seed})"
    log(f"  context {ctx_p}\n  validation {len(yv_all)} rows ({val_txt}), "
        f"test {len(yt)} rows ({test_p.name}), scored every {a.test_every} iterations")
    log(f"  {a.optimizer}, lr {a.lr:g}, {a.steps} steps"
        + (f", scale {a.pgd_scale}" if a.optimizer == "pgd" else
           f", betas ({a.adam_beta1:g}, {a.adam_beta2:g}), eps {a.adam_eps:g}")
        + f", every gate starts at {a.g_init:g}; hard evaluation discards {k_remove} rows "
          f"({a.remove_percent:g}%)")
    log(f"  run {run_name}  ->  {out}")
    config = {"args": vars(a), "run_name": run_name, "context": ctx_p, "val": val_p, "test": test_p,
              "n_classes": K, "n_context": n, "n_features": Xc.shape[1], "n_poisoned": len(rows),
              "n_val": len(yv_all), "n_test": len(yt), "k_remove": k_remove, "validation": validation,
              "gate": ({"type": "topk", "ste": a.ste, "score_init_std": a.score_init_std,
                        "init_seed": a.init_seed, "k_remove": k_remove, "select_on": a.select_on}
                       | ({"ste_temp": a.ste_temp} if a.ste == "sigmoid-centered" else {})
                       | ({"score_init": a.score_init} if a.score_init != "random" else {})
                       | ({"score_init_path": str(a.score_init_path)} if a.score_init == "file" else {})
                       | ({"line_search": a.line_search} if a.line_search else {})
                       | ({"ste_probe": a.ste_probe} if a.ste_probe else {})
                       | ({"swap_search": a.swap_search} if a.swap_search else {})
                       | ({"budget_warmup": a.budget_warmup} if a.budget_warmup > 0 else {})
                       | ({"lr_schedule": a.lr_schedule, "lr_min_frac": a.lr_min_frac}
                          if a.lr_schedule != "constant" else {})
                       if a.gate == "topk" else {"type": "soft", "select_on": a.select_on}),
              "gated": "features only (feature_group_embedder input); labels untouched"}

    with wandb_run(a, run_name, out, config) as run:
        # -- model and the all-missing encoder input ----------------------------------------
        clf = build_tabpfn_v2(device, a.model_seed, recompute_layers=not a.no_recompute, n_classes=K)
        null_x = capture_null_input(clf, Xc, yp_t)
        half = null_x.shape[1] // 2
        config["u_null"] = {"shape": list(null_x.shape),
                            "value_part_unique": torch.unique(null_x[:, :half]).tolist(),
                            "indicator_part_unique": torch.unique(null_x[:, half:]).tolist()}
        log(f"  u_null {config['u_null']['shape']}: values {config['u_null']['value_part_unique']}, "
            f"indicators {config['u_null']['indicator_part_unique']}")
        write_json(out / "config.json", config)
        if run is not None:
            run.config.update({"u_null": config["u_null"]})

        def metrics(p, y):
            m = multiclass_metrics(p, y, K)
            return {k: float(m[k]) for k in METRICS}

        def soft(Xq, yq, yq_t, g, y_ctx=yp_t):
            """Scores of the query rows with the (gated) context; g = None: no gate."""
            p = evaluate(clf, Xc, y_ctx, Xq, yq_t, g, null_x, need_grad=False, batch_size=a.batch_size)[2]
            return metrics(p, yq), p

        def as_context(X_ctx, y_ctx):
            """Test scores of plain TabPFN, no gate, on the context (X_ctx, y_ctx)."""
            p = evaluate(clf, X_ctx, as_t(y_ctx, torch.float32), Xt, yt_t, None, null_x,
                         need_grad=False, batch_size=a.batch_size)[2]
            return metrics(p, yt), p

        def hard(drop):
            """Test scores of plain TabPFN on the poisoned context without the rows `drop`."""
            keep = np.setdiff1d(np.arange(n), drop)
            if len(np.unique(y_pois[keep])) < K:  # a class vanished: TabPFN would misalign classes
                log(f"  [warn] removing {len(drop)} rows drops a whole class; hard scores skipped")
                return {k: math.nan for k in METRICS}, None
            return as_context(Xc[keep], y_pois[keep])

        # -- references, all independent of the learned gates --------------------------------
        oracle = as_t(~poisoned, torch.float32)
        final, probs = {}, {}
        final["clean_context"], probs["clean_context"] = soft(Xt, yt, yt_t, None, y_ctx=yc_t)
        final["poisoned_no_gate"], probs["poisoned_no_gate"] = soft(Xt, yt, yt_t, None)
        final["poisoned_random_removal"], probs["poisoned_random_removal"] = hard(
            lowest_gated(np.ones(n, dtype=np.float32), k_remove, tie_order))
        refs = {}
        if poisoned.any():
            final["poisoned_oracle_gate"], probs["poisoned_oracle_gate"] = soft(Xt, yt, yt_t, oracle)
            final["poisoned_oracle_removal"], probs["poisoned_oracle_removal"] = hard(rows)
            refs |= prefixed("val", soft(Xv, yv, yv_t, oracle)[0])
            if holdout:
                refs |= prefixed("valtest", soft(Xh, yh, yh_t, oracle)[0])
            refs |= prefixed("test_soft", final["poisoned_oracle_gate"])
            refs |= prefixed("test_hard", final["poisoned_oracle_removal"])
        # the clean validation rows used directly: alone, and added to the poisoned context
        final["val_as_context"], probs["val_as_context"] = as_context(Xv_all, yv_all)
        final["poisoned_plus_val"], probs["poisoned_plus_val"] = as_context(
            torch.cat([Xc, Xv_all]), np.concatenate([y_pois, yv_all]))
        baseline = ("test, validation set as the context", final["val_as_context"])
        for name, m in final.items():
            log(f"  reference  test {name:<26}" + "  ".join(f"{k} {m[k]:.4f}" for k in METRICS))

        # -- training ------------------------------------------------------------------
        topk = a.gate == "topk"
        if topk:  # free scores; the gate is the hard top-k mask of them, rebuilt every iteration
            if not 0 < k_remove < n:
                raise SystemExit(f"--gate topk needs 0 < rows masked < {n}, got {k_remove}")
            scores = (init_scores(n, a.score_init_std, a.init_seed, device) if a.score_init_std > 0
                      else torch.zeros(n, device=device))
            if a.score_init == "gradient":  # SNIP-like start: rank rows by dCE/dg at the full context
                g1 = torch.ones(n, device=device, requires_grad=True)
                _, g_full, _ = evaluate(clf, Xc, yp_t, Xv, yv_t, g1, null_x, need_grad=True,
                                        batch_size=a.batch_size)
                scores = scores - g_full / g_full.std()
            elif a.score_init == "file":
                z = np.load(a.score_init_path)
                v = z["scores"] if "scores" in z.files else z["g"]
                if len(v) != n:
                    raise SystemExit(f"{a.score_init_path} has {len(v)} values for {n} context rows")
                scores = scores + torch.as_tensor(v, dtype=torch.float32, device=device)
            warmup = int(round(a.budget_warmup * a.steps))
            step_mult = 1.0  # line search: remembered step multiplier
            pairs = a.swap_search  # swap search: current pairs per proposal
            swap_rng = np.random.default_rng(a.init_seed)
            if a.swap_search and torch.unique(scores).numel() < n:  # break ties once, reproducibly
                scores = scores + 1e-6 * init_scores(n, 1.0, a.init_seed + 10_000, device)
            g = topk_mask(scores, budget_at(0, k_remove, warmup))
        else:
            scores = None
            g = torch.full((n,), float(a.g_init), device=device, requires_grad=True)
        opt = (torch.optim.Adam([g], lr=a.lr, betas=(a.adam_beta1, a.adam_beta2), eps=a.adam_eps)
               if a.optimizer == "adam" else None)
        scored = ("val",) + (("valtest",) if holdout else ()) + ("test_soft", "test_hard")
        fields = (["iter"] + [f"{s}_{k}" for s in scored for k in METRICS]
                  + list(gate_stats(np.ones(n), poisoned, np.arange(k_remove)))
                  + (["score_std", "mask_flips"] if topk else [])
                + (["ls_evals", "ls_accept", "step_mult"] if topk and (a.line_search or a.swap_search) else [])
                + (["swap_pairs"] if topk and a.swap_search else [])
                  + ["grad_max_abs", "max_abs_dg", "time_s"])

        sel = "valtest" if holdout else "val"  # the split the best iteration is chosen on

        def rank_key(entry):  # top-k order (--select-on): accuracy then CE, or CE then accuracy
            r = entry[0]
            if a.select_on == "ce":
                return (r[f"{sel}_ce"], -r[f"{sel}_accuracy"], r["iter"])
            return (-r[f"{sel}_accuracy"], r[f"{sel}_ce"], r["iter"])

        log_rows, top = [], []  # top: [(log row, gate vector, scores or None)], best first
        masks_seen = []  # topk: the masked rows of every iteration, saved to gates/masks.npz
        with open(out / "log.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, restval="")
            writer.writeheader()
            for it in range(a.steps + 1):
                t0 = time.time()
                if topk:
                    k_t = budget_at(it, k_remove, warmup)
                    g = topk_mask(scores, k_t).requires_grad_(it < a.steps)
                g_now = g.detach().cpu().numpy().copy()
                s_now = scores.detach().cpu().numpy().copy() if topk else None
                if topk:
                    masks_seen.append(g_now == 0)
                removed = (np.flatnonzero(g_now == 0) if topk  # exactly the masked rows
                           else lowest_gated(g_now, k_remove, tie_order))
                relaxed = topk and (a.ste_probe > 0 or a.swap_search)
                ce, grad, p_val = evaluate(clf, Xc, yp_t, Xv, yv_t, g, null_x,
                                           need_grad=it < a.steps and not relaxed, batch_size=a.batch_size)
                if relaxed and it < a.steps:  # the gradient at the probe point; loss and metrics stay hard
                    hard_g = g.detach()
                    probe = (hard_g * (1 - a.ste_probe) + (1 - hard_g) * a.ste_probe).requires_grad_(True)
                    grad = evaluate(clf, Xc, yp_t, Xv, yv_t, probe, null_x, need_grad=True,
                                    batch_size=a.batch_size)[1]
                row = {"iter": it} | prefixed("val", metrics(p_val, yv))
                if abs(ce - row["val_ce"]) > 1e-4:  # the loss and the metric are independent paths
                    raise AssertionError(f"loss {ce:.6f} != metric CE {row['val_ce']:.6f}")
                if a.test_every and (it % a.test_every == 0 or it == a.steps):
                    row |= prefixed("test_soft", soft(Xt, yt, yt_t, g)[0])
                    row |= prefixed("test_hard", hard(removed)[0])
                    if holdout:
                        row |= prefixed("valtest", soft(Xh, yh, yh_t, g)[0])
                row |= gate_stats(g_now, poisoned, removed, scores=s_now)
                if topk:
                    row["score_std"] = float(s_now.std())

                if grad is not None and topk and a.swap_search:  # verified swaps of gradient-ranked pairs
                    hard_g = g.detach()
                    gr = grad.detach().cpu().numpy()
                    kept, off = np.flatnonzero(g_now == 1), np.flatnonzero(g_now == 0)
                    out_pool = kept[np.argsort(-gr[kept], kind="stable")]  # masking should help most
                    in_pool = off[np.argsort(gr[off], kind="stable")]      # returning should help most
                    row["grad_max_abs"] = float(np.abs(gr).max())
                    evals, accepted, flips = 0, 0, 0
                    for attempt in range(a.line_search or 3):
                        b = min(pairs, len(off), len(kept))
                        go = swap_rng.choice(out_pool[:4 * b], b, replace=False)
                        gi = swap_rng.choice(in_pool[:4 * b], b, replace=False)
                        cand = hard_g.clone()
                        cand[torch.as_tensor(go, device=device)] = 0.0
                        cand[torch.as_tensor(gi, device=device)] = 1.0
                        evals += 1
                        ce_c = evaluate(clf, Xc, yp_t, Xv, yv_t, cand, null_x, need_grad=False,
                                        batch_size=a.batch_size)[0]
                        if ce_c < ce:
                            accepted, flips = 1, 2 * b
                            # scores follow the mask (low = masked), the gradient orders within each side
                            with torch.no_grad():
                                scores.copy_(cand - 0.5 - 1e-3 * grad / grad.abs().max().clamp_min(1e-12))
                            if attempt == 0:
                                pairs = min(2 * pairs, a.swap_search * 8)
                            break
                        pairs = max(1, pairs // 2)
                    row |= {"ls_evals": evals, "ls_accept": accepted, "step_mult": pairs / a.swap_search,
                            "swap_pairs": b, "mask_flips": flips,
                            "max_abs_dg": float(np.abs(scores.detach().cpu().numpy() - s_now).max())}
                elif grad is not None and topk:  # straight through to the scores, then a new mask
                    gs = ste_grad(grad, scores, a.ste, k_remove=k_remove, temp=a.ste_temp)
                    row["grad_max_abs"] = float(gs.abs().max())
                    lr_t = lr_at(it, a.steps, a.lr, a.lr_schedule, a.lr_min_frac)
                    if a.line_search and k_t == k_remove and budget_at(it + 1, k_remove, warmup) == k_remove:
                        # backtracking: keep the step only if the new mask lowers the fitted-row CE
                        evals, accepted = 0, 0
                        for attempt in range(a.line_search):
                            cand = scores.detach().clone()
                            score_step(cand, gs, lr_t * step_mult, a.pgd_scale)
                            m_c = topk_mask(cand, k_remove)
                            if torch.equal(m_c, g.detach()):  # no row moved: same CE, keep the scores
                                accepted = 1
                            else:
                                evals += 1
                                ce_c = evaluate(clf, Xc, yp_t, Xv, yv_t, m_c, null_x, need_grad=False,
                                                batch_size=a.batch_size)[0]
                                accepted = int(ce_c < ce)
                            if accepted:
                                scores.data.copy_(cand)
                                if attempt == 0:
                                    step_mult = min(1.0, 2 * step_mult)
                                break
                            step_mult = max(step_mult / 2, 1e-4)
                        row |= {"ls_evals": evals, "ls_accept": accepted, "step_mult": step_mult}
                    else:
                        score_step(scores, gs, lr_t, a.pgd_scale)
                    row["max_abs_dg"] = float(np.abs(scores.detach().cpu().numpy() - s_now).max())
                    row["mask_flips"] = int((topk_mask(scores, k_t).cpu().numpy() != g_now).sum())
                elif grad is not None:  # every iteration but the last: update the gates
                    row["grad_max_abs"] = float(grad.abs().max())
                    if a.optimizer == "pgd":
                        pgd_step(g, grad, a.lr, a.pgd_scale)
                    else:
                        adam_step(g, grad, opt)
                    row["max_abs_dg"] = float((g.detach().cpu() - torch.from_numpy(g_now)).abs().max())
                row["time_s"] = time.time() - t0
                writer.writerow(row)
                fh.flush()
                log_rows.append(row)
                if run is not None:
                    run.log(wandb_keys(row), step=it)

                full_budget = not topk or k_t == k_remove
                if f"{sel}_ce" in row and full_budget:  # holdout: only test iterations compete
                    top = sorted(top + [(row, g_now, s_now)], key=rank_key)[:a.top_k]
                if any(r is row for r, *_ in top):  # this iteration entered the top-k
                    for k, (r, gk, sk) in enumerate(top, 1):
                        save_gate(out / "gates" / f"top{k}.npz", gk, r, scores=sk)

                test_txt = ((f" | valtest acc {row['valtest_accuracy']:.4f}" if "valtest_ce" in row else "")
                            + f" | test soft acc {row['test_soft_accuracy']:.4f} "
                            f"hard acc {row['test_hard_accuracy']:.4f}" if "test_soft_ce" in row else "")
                det_txt = (f" | poisoned g {row['g_mean_poisoned']:.3f} clean g {row['g_mean_clean']:.3f}"
                           f" auroc {row['auroc_poisoned']:.3f}"
                           + (f" precision {row['precision_at_R']:.3f} flips {row.get('mask_flips', 0)}"
                              if topk else "") if "auroc_poisoned" in row else "")
                log(f"it {it:4d}  val ce {row['val_ce']:.4f} acc {row['val_accuracy']:.4f}{test_txt}"
                    f"{det_txt}  [{row['time_s']:.1f}s]")
                if a.plot_every and (it % a.plot_every == 0 or it == a.steps):
                    plot_curves(log_rows, refs, top[0][0]["iter"] if top else it, out / "plots", baseline)

        # -- the best gates on the test set, soft and hard ----------------------------------
        save_gate(out / "gates" / "final.npz", g.detach().cpu().numpy(), log_rows[-1],
                  scores=scores.detach().cpu().numpy() if topk else None)
        if topk:  # one row per iteration (0..steps): True = masked
            np.savez_compressed(out / "gates" / "masks.npz", masked=np.stack(masks_seen),
                                iters=np.arange(len(masks_seen)))
        best_row, g_best, s_best = top[0]
        final["poisoned_best_gate_soft"], probs["poisoned_best_gate_soft"] = soft(
            Xt, yt, yt_t, as_t(g_best, torch.float32))
        final["poisoned_best_gate_hard"], probs["poisoned_best_gate_hard"] = hard(
            lowest_gated(g_best, k_remove, tie_order))
        log(f"best: iteration {best_row['iter']}  val acc {best_row['val_accuracy']:.4f} "
            f"ce {best_row['val_ce']:.4f}; top-k iterations {[r['iter'] for r, *_ in top]}")
        for name in ("poisoned_best_gate_soft", "poisoned_best_gate_hard"):
            log(f"  test {name:<26}" + "  ".join(f"{k} {final[name][k]:.4f}" for k in METRICS))

        recovery = {}  # share of the poisoned -> oracle gap that the best gates close
        for mode, ref in (("soft", "poisoned_oracle_gate"), ("hard", "poisoned_oracle_removal")):
            if ref in final:
                recovery[mode] = {}
                for k in METRICS:
                    gap = final[ref][k] - final["poisoned_no_gate"][k]
                    got = final[f"poisoned_best_gate_{mode}"][k] - final["poisoned_no_gate"][k]
                    recovery[mode][k] = got / gap if abs(gap) > 1e-12 else None
        # > 0 on accuracy-type metrics (< 0 on ce): the gates beat the clean rows used directly
        vs_val = {k: final["poisoned_best_gate_soft"][k] - final["val_as_context"][k] for k in METRICS}
        summary = {
            "best_iteration": best_row["iter"],
            "ranked_on": (("validation-test (held-out)" if holdout else "validation")
                          + (" CE, ties -> higher accuracy" if a.select_on == "ce" else
                             " accuracy, ties -> lower CE")
                          + ", then earlier" + ("; test iterations only" if holdout else "")),
            "best_val_row": best_row, "top_k_iterations": [r["iter"] for r, *_ in top],
            "best_gate_stats": gate_stats(g_best, poisoned, lowest_gated(g_best, k_remove, tie_order),
                                          scores=s_best),
            "k_removed_in_hard_evaluation": k_remove,
            "metrics": {name: (multiclass_metrics(probs[name], yt, K) if probs[name] is not None
                               else final[name]) for name in final},
            "recovery_vs_oracle": recovery,
            "vs_val_as_context": vs_val,
            "selection_rules": selection_rules(log_rows, holdout),
        }
        write_json(out / "test_metrics.json", summary)
        np.savez_compressed(out / "test_probs.npz", y_test=yt.astype(np.int16),
                            **{k: v.astype(np.float32) for k, v in probs.items() if v is not None})
        plot_curves(log_rows, refs, best_row["iter"], out / "plots", baseline)
        plot_gate_hist(s_best if topk else g_best, poisoned, out / "plots" / "gate_hist_best.png",
                       f"best {'scores' if topk else 'gates'} (iteration {best_row['iter']})",
                       xlabel="score S (the lowest rows are masked)" if topk else "gate g")
        if run is not None:
            wandb_upload(run, out, run_name, {
                "best_iteration": best_row["iter"], "top_k_iterations": summary["top_k_iterations"],
                "final": final, "recovery_vs_oracle": recovery, "vs_val_as_context": vs_val,
                "selection_rules": summary["selection_rules"],
                "best_gate_stats": summary["best_gate_stats"]}, log_rows, baseline)
        log(f"  wrote {out}")
    return out


if __name__ == "__main__":
    main()

"""Sifting: carve a large, hopefully clean subset out of a label-poisoned context.

THE IDEA
    Put the whole poisoned set D on both sides of TabPFNv2 -- as the context and as the
    query set -- and learn a soft split. Each row i carries a weight X[i] = sigmoid(t[i])
    in [0, 1]: it belongs to the context with strength X[i] and to the query set with
    strength 1 - X[i]. Then MAXIMISE

        L(t) = (1/n) * sum_j (1 - X[j]) * CE( row j | context weighted by X )

    The argument: the split that hurts most is the one where the context is clean and
    everything held out is corrupted, because a clean context predicts a flipped label
    about as badly as it is possible to predict anything. Ascending L should therefore
    push the poison out of the context. Afterwards the top (1-R)% of rows by X are kept.

HOW X ENTERS THE MODEL
    Through the row attention (``per_column_attention_between_cells``: rows attending to
    rows), as an additive pre-softmax bias of log X[i] on every key i:

        alpha_ji = softmax_i( s_ji + log X[i] )    <=>    alpha_ji ~ X[i] * exp(s_ji)

    Additive in log space, NOT multiplicative on the logit: multiplying would send a
    zero-weight row's logit to 0, and exp(0) = 1 would give it MORE weight than a row
    with a negative score. log X[i] = logsigmoid(t[i]) is computed stably and is exactly
    0 when X = 1, so a weight of 1 is a true no-op.

    NOTE -- softmax is shift-invariant, so scaling every row down by the same factor
    changes the attention NOT AT ALL. Only the spread of X matters here. The collapse
    "all X -> 0" therefore does not empty the context; it leaves it at full strength and
    merely turns every row into a full-weight query. That is a far weaker attractor than
    it looks, which is why a hard mass constraint is optional (see ``mass_penalty``).

THE SECOND CHANNEL
    Attention is not the only way a context row reaches the test rows. Every test row
    carries a NaN target, and TabPFN imputes it with the MEAN OF THE TRAINING LABELS --
    a statistic no attention bias can touch. X is therefore also applied there, as a
    weighted mean, so a row being weighted out stops contributing to it as well. Both
    channels are installed together by ``weights_installed``.

WHAT IS NOT TOUCHED
    The feature attention (``per_sample_attention_between_features``) is left alone: it
    acts within a row, so a per-row weight has no meaning there.

    The FEATURE statistics (the standard scaler and the feature-group normalisation,
    both fitted on the context) stay unweighted. They carry no label information, which
    is what the poison is in, and weighting them would change the input scaling out from
    under the model at every step.

SELF-ATTENTION
    D appears twice, so query copy j and context copy j are the same row. Left alone,
    that row would attend to itself and read its own label straight off the context --
    the loss would be measuring leakage, not generalisation. The diagonal is therefore
    masked out for the query side.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

# Large finite stand-in for -inf. exp(-1e9) underflows to exactly 0 in float32, so the
# attention weight is zero, while the backward pass stays finite -- a true -inf can
# produce NaN gradients even at positions whose weight is 0.
NEG_INF = -1e9

# A logit low enough to silence a row outright. Suppression is RELATIVE to the attention
# logits, not absolute: -60 (X ~ 1e-26) still leaves a measurable trace on heloc, while
# -1000 removes a row's influence exactly (verified in verify_sift_mechanism.py). Only
# used for hard on/off cases such as tests and ablations -- the optimiser works in the
# soft regime, where what matters is the SPREAD of X, not its level.
HARD_OFF = -1000.0


class RowWeights:
    """The parameter being optimised: one logit per context row, X = sigmoid(logit).

    Initialised high on purpose (``init_logit`` default 4.0 -> X = 0.982): every row
    starts essentially fully in the context, so the search begins from "trust
    everything" and has to be argued down into excluding rows.
    """

    def __init__(self, n: int, device: str, init_logit: float = 4.0):
        self.logit = torch.full((n,), float(init_logit), device=device,
                                dtype=torch.float32, requires_grad=True)

    @property
    def x(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    @property
    def log_x(self) -> torch.Tensor:
        """log sigmoid(logit) -- the attention bias. Stable as the logit goes negative."""
        return F.logsigmoid(self.logit)

    def numpy(self) -> np.ndarray:
        return self.x.detach().float().cpu().numpy()


def row_attention_modules(clf, layers: str | list[int] = "all") -> list:
    """The row-attention module of each selected transformer block.

    ``layers`` is "first", "all", or an explicit list of block indices.

    The classifier must already have been fitted once: ``models_`` is created by the
    first ``fit_with_differentiable_input`` call, and the modules we attach to live
    inside it.
    """
    if not hasattr(clf, "models_"):
        raise RuntimeError("classifier is not fitted yet -- call "
                           "fit_with_differentiable_input(X, y) once before selecting "
                           "the attention modules")
    blocks = clf.models_[0].blocks
    if layers == "all":
        idx = list(range(len(blocks)))
    elif layers == "first":
        idx = [0]
    else:
        idx = list(layers)
    bad = [i for i in idx if not 0 <= i < len(blocks)]
    if bad:
        raise ValueError(f"block index out of range {bad}; model has {len(blocks)} blocks")
    return [blocks[i].per_column_attention_between_cells for i in idx]


def build_masks(log_x: torch.Tensor, query_rows: torch.Tensor | None) -> dict:
    """The two additive masks, in torch SDPA's (B, H, S_q, S_k) layout.

    ``train`` biases the context rows' own attention; it is the same for every query, so
    it broadcasts as [1, 1, 1, N] and costs nothing.

    ``test`` additionally blocks each query from its own copy in the context. It has to
    be materialised at [1, 1, M, N] because that diagonal differs per query -- which is
    cheap only because the query set is chunked (M is the chunk size, not n).

    ``query_rows[t]`` is the context-row index that query t is a copy of, or None when
    the queries are not drawn from the context and no diagonal has to be cut.
    """
    n = log_x.shape[0]
    train = log_x.view(1, 1, 1, n)
    if query_rows is None:
        return {"train": train, "test": train}
    m = query_rows.shape[0]
    test = log_x.view(1, 1, 1, n).expand(1, 1, m, n).clone()
    test[0, 0, torch.arange(m, device=log_x.device), query_rows] = NEG_INF
    return {"train": train, "test": test}


@contextmanager
def weights_installed(arch, modules: list, masks: dict | None,
                      x: torch.Tensor | None = None):
    """Attach both channels X acts through, for the duration of one forward.

    ``masks``  the additive attention bias, on the chosen blocks.
    ``x``      the weights themselves, for the weighted training-label mean that the
               test rows' NaN targets are imputed with. Pass None to leave the stock
               (unweighted) statistic in place.

    Everything is removed on the way out, including on an exception: a bias left behind
    would silently corrupt every later call on the same classifier.
    """
    try:
        for mod in modules:
            mod.row_attn_bias = masks
        if arch is not None:
            arch.row_weights = x
        yield
    finally:
        for mod in modules:
            mod.row_attn_bias = None
        if arch is not None:
            arch.row_weights = None


@contextmanager
def row_bias_installed(modules: list, masks: dict | None):
    """Attention bias only -- kept for the mechanism checks, which isolate that channel."""
    with weights_installed(None, modules, masks, None):
        yield


def weighted_loss(clf, X_ctx: torch.Tensor, y_ctx: torch.Tensor, weights: RowWeights,
                  modules: list, *, chunk: int, y_np: np.ndarray,
                  need_grad: bool = True,
                  weighted_label_mean: bool = True) -> tuple[torch.Tensor, np.ndarray]:
    """L = (1/n) * sum_j (1 - X[j]) * CE(row j | context weighted by X), and per-row CE.

    The query set is D itself, walked in chunks. The loss is a plain sum over rows, so
    chunking is exact and the gradients of the chunks simply accumulate. Dividing by n
    (a constant) keeps the gradient scale sane without changing what is being maximised.
    """
    from .runtime import _forward, _install_recompute

    n = X_ctx.shape[0]
    device = X_ctx.device
    total = torch.zeros((), device=device)
    per_row = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # Rebuilt per chunk on purpose. Each chunk calls backward(), which frees the
        # graph it walked; a sigmoid computed once outside the loop would be freed by
        # the first chunk and unusable by the second. Rebuilding gives every chunk its
        # own path back to the leaf, and the gradients still accumulate on it.
        x = weights.x
        log_x = weights.log_x
        rows = torch.arange(start, end, device=device)
        masks = build_masks(log_x, rows)
        with torch.enable_grad() if need_grad else torch.no_grad():
            clf.fit_with_differentiable_input(X_ctx, y_ctx)
            _install_recompute(clf)
            # The bias must still be installed during backward(). Activation
            # checkpointing re-runs the forward inside the backward pass, and a bias
            # removed too early would make the recomputation disagree with the original
            # forward -- which torch detects and raises on.
            with weights_installed(clf.models_[0], modules, masks,
                                   x if weighted_label_mean else None):
                probs = _forward(clf, X_ctx[start:end], need_grad=need_grad)
                p_given = probs[torch.arange(end - start, device=probs.device),
                                torch.as_tensor(y_np[start:end], device=probs.device)]
                ce = -torch.log(p_given.clamp_min(1e-12))
                loss = ((1.0 - x[start:end]) * ce).sum() / n
                if need_grad:
                    loss.backward()
        total = total + loss.detach()
        per_row[start:end] = ce.detach().float().cpu().numpy()
        del probs, ce, loss, masks
    return total, per_row


def select_clean_subset(x: np.ndarray, y_given: np.ndarray, keep: int,
                        strategy: str = "proportional") -> np.ndarray:
    """The ``keep`` highest-weight rows, taken per class so one class cannot be wiped out.

    ``proportional``  each class contributes in the ratio it has in D. D is poisoned, so
                      that ratio is only approximate -- which is fine, it just has to
                      stop the optimiser from solving the problem by deleting a whole
                      class (a real risk on coil2000, 6% positive).
    ``balanced``      equal count per class.
    ``global``        no class constraint at all; the plain top-``keep``. For comparison.
    """
    y_given = np.asarray(y_given).astype(int)
    order = np.argsort(-np.asarray(x), kind="stable")          # high weight first
    if strategy == "global":
        return np.sort(order[:keep])

    classes = [0, 1]
    counts = np.bincount(y_given, minlength=2)
    if strategy == "proportional":
        quota = {c: int(round(keep * counts[c] / len(y_given))) for c in classes}
    elif strategy == "balanced":
        quota = {c: keep // 2 for c in classes}
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # A class may not hold its quota; the shortfall goes to the other class so that
    # exactly `keep` rows come back.
    quota = {c: min(quota[c], int(counts[c])) for c in classes}
    picked: list[int] = []
    for c in classes:
        want = quota[c]
        picked += [i for i in order if y_given[i] == c][:want]
    if len(picked) < keep:
        chosen = set(picked)
        picked += [i for i in order if i not in chosen][:keep - len(picked)]
    return np.sort(np.asarray(picked[:keep], dtype=np.int64))


def subset_report(kept: np.ndarray, flipped: np.ndarray, n: int) -> dict:
    """How clean the kept subset actually is -- the number the whole method is for."""
    is_flipped = np.zeros(n, dtype=bool)
    is_flipped[np.asarray(flipped, dtype=np.int64)] = True
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[np.asarray(kept, dtype=np.int64)] = True
    poison_kept = int((kept_mask & is_flipped).sum())
    return {
        "n_kept": int(kept_mask.sum()),
        "poison_in_kept": poison_kept,
        "purity": float(1 - poison_kept / max(int(kept_mask.sum()), 1)),
        "poison_total": int(is_flipped.sum()),
        "poison_removed": int(is_flipped.sum()) - poison_kept,
        "poison_recall": float((int(is_flipped.sum()) - poison_kept)
                               / max(int(is_flipped.sum()), 1)),
        "clean_discarded": int((~kept_mask & ~is_flipped).sum()),
    }


# ------------------------------------------------------------------ the search

def optimise(clf, X_ctx: torch.Tensor, y_ctx: torch.Tensor, weights: RowWeights,
             modules: list, *, y_np: np.ndarray, n_iter: int, lr: float, chunk: int,
             keep_frac: float, mass_penalty: float = 0.0, flipped: np.ndarray | None = None,
             strategies: tuple = ("proportional", "balanced", "global"),
             log=print) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Gradient ASCENT on the held-out loss.

    Returns ``(trace, x_history, ce_history)``.

    ``x_history`` is ``[n_iter + 1, n]``: row 0 is the initial X, row t+1 the X after
    step t. ``ce_history`` is ``[n_iter, n]``, the per-row cross-entropy measured during
    step t -- so ``ce_history[t]`` and ``trace[t]["loss"]`` were both computed at
    ``x_history[t]``, BEFORE that step was applied. Keeping the whole trajectory is what
    makes it possible to see afterwards whether the run drifted into a degenerate
    solution or peaked early and then decayed.

    ``mass_penalty`` (default 0, i.e. off) subtracts ``lambda * (mean(X) - keep_frac)^2``,
    a soft pull towards keeping the intended fraction. It is not needed to stop the
    "everything leaves the context" collapse -- softmax is shift-invariant, so a uniform
    X has no effect on attention at all -- but it is here if a run needs steering.

    ``flipped`` is used ONLY to score the ranking as the run proceeds. It never enters
    the loss; the method itself has no access to it.
    """
    opt = torch.optim.Adam([weights.logit], lr=lr, maximize=True)
    n = X_ctx.shape[0]
    keep = int(round(keep_frac * n))
    trace: list[dict] = []
    x_history = [weights.numpy()]
    ce_history = []

    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss, per_row = weighted_loss(clf, X_ctx, y_ctx, weights, modules,
                                      chunk=chunk, y_np=y_np, need_grad=True)
        if mass_penalty:
            pen = mass_penalty * (weights.x.mean() - keep_frac) ** 2
            (-pen).backward()                      # ascending, so the penalty subtracts
        gmax = float(weights.logit.grad.abs().max())
        x_before = x_history[-1]
        opt.step()

        row = {"iter": it, "loss": float(loss), "grad_absmax": gmax,
               "x_mean": float(x_before.mean()), "x_min": float(x_before.min()),
               "x_max": float(x_before.max()), "x_std": float(x_before.std()),
               "ce_mean": float(per_row.mean())}
        # Scored on the weights the loss was measured at, so the columns line up.
        if flipped is not None and len(flipped):
            for strat in strategies:
                picked = select_clean_subset(x_before, y_np, keep, strat)
                row[f"precision_{strat}"] = subset_report(picked, flipped, n)["poison_recall"]
        trace.append(row)
        x_history.append(weights.numpy())
        ce_history.append(per_row.astype(np.float32))

        log(f"    iter {it:>2d}  loss {row['loss']:.6f}  X mean {row['x_mean']:.4f} "
            f"[{row['x_min']:.4f}, {row['x_max']:.4f}] sd {row['x_std']:.4f}  "
            f"|grad|max {gmax:.2e}"
            + "".join(f"  {s[:4]} {row['precision_' + s]:.4f}"
                      for s in strategies if f"precision_{s}" in row))

    return trace, np.asarray(x_history, dtype=np.float32), np.asarray(ce_history, dtype=np.float32)

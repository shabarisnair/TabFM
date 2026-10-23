"""Weight the context rows of TabPFNv2 by w in [0, 1], everywhere a context row reaches another row.

w_j = 1 leaves context row j as it is, w_j = 0 is exactly deleting it, and a small w_j is
close to deleting it. The weights enter every computation of the model and of its
differentiable preprocessing that mixes rows (checked by scripts/test_row_weights.py):

  1. attention between items, in every block: log w_j is added to every logit whose key
     is context row j, so  alpha_ij = w_j exp(s_ij) / sum_k w_k exp(s_ik)  -- for context-row
     and test-row queries alike, every head (test rows use only key/value head 0) and
     every column, the label column included. Additive in log space: multiplying the
     score by w would leave a zero-weight row with exp(0) = 1, not 0.
  2. the label mean that fills each test row's unknown label: sum w_i y_i / sum w_i. Its
     class rank (the multiclass target encoding) is a step function, so it carries no
     gradient.
  3. the z-norms fitted on the context, the external DifferentiableZNormStep and the
     model's internal standard scaler: mean = sum w x / sum w, var = sum w (x - mean)^2 /
     (sum w - 1); likewise the encoder's NaN-imputation means.
  4. the discrete fits on the context -- the features that are constant (the external
     RemoveConstantFeaturesStep, the encoder's constant-feature and feature-group masks)
     and the classes present among the labels -- ignore rows of weight exactly 0. They
     carry no gradient, and while every weight is > 0 they are the stock ones.

Channels 1-2 are those of the older --gate attn (sift.py); 3-4 are what make w = 0 the same
as deleting the row (sift.py only claimed approximate removal).

log w = -inf (w = 0) enters the attention as NEG_INF: exp(NEG_INF + s) underflows to exactly
0 in float32 while the gradients stay finite.

    probs = probs_weighted(clf, X_ctx, y_ctx, X_query, log_w)
    loss, grad_log_w, probs = evaluate_weighted(clf, X_ctx, y_ctx, X_query, y_query, log_w, need_grad=True)
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

NEG_INF = -1e9


def attention_bias(log_w: torch.Tensor, query_ctx_rows=None) -> dict:
    """The additive attention bias, broadcastable to SDPA's (batch, heads, queries, keys):
    one entry per context row (key), the same for the context-row and the test-row queries.
    query_ctx_rows[q] >= 0 marks test-row query q as a copy of that context row: its own key
    is cut out (NEG_INF), so it is predicted leave-one-out."""
    bias = log_w.clamp(min=NEG_INF).reshape(1, 1, 1, -1)
    if query_ctx_rows is None:
        return {"train": bias, "test": bias}
    rows = torch.as_tensor(np.asarray(query_ctx_rows), dtype=torch.long, device=log_w.device)
    test = bias.expand(1, 1, len(rows), bias.shape[-1]).clone()
    own = rows >= 0
    test[0, 0, torch.nonzero(own).flatten(), rows[own]] = NEG_INF
    return {"train": bias, "test": test}


@contextmanager
def model_weights_installed(clf, log_w: torch.Tensor, label_log_w: torch.Tensor | None = None,
                            stats_log_w: torch.Tensor | None = None, query_ctx_rows=None):
    """Channels 1, 2 and the encoder's part of 3-4 on the fitted model, for one forward
    (and its backward: activation checkpointing re-runs the forward inside it). Removed
    on the way out, also on an exception -- a weight left behind would silently change
    every later call on the same classifier. label_log_w / stats_log_w: other log-weights for
    the label mean / the encoder statistics (e.g. detached ones; default: log_w)."""
    from .sift import row_attention_modules

    arch = clf.models_[0]
    modules = row_attention_modules(clf, "all")
    w_label = (log_w if label_log_w is None else label_log_w).exp()
    w_stats = (log_w if stats_log_w is None else stats_log_w).exp()
    try:
        for mod in modules:
            mod.row_attn_bias = attention_bias(log_w, query_ctx_rows)
        arch.row_weights = w_label      # the label mean (channel 2)
        arch.row_weights_stats = w_stats  # the encoder's statistics and masks (3-4)
        yield
    finally:
        for mod in modules:
            mod.row_attn_bias = None
        arch.row_weights = None
        arch.row_weights_stats = None


@contextmanager
def preprocessing_weights_installed(log_w: torch.Tensor):
    """The external preprocessing's part of channels 3-4, for one fit_with_differentiable_input."""
    from tabpfn.preprocessing.steps.context_row_weights import context_row_weights_installed

    with context_row_weights_installed(log_w.exp()):
        yield


def _check(clf, X_ctx: torch.Tensor, log_w: torch.Tensor) -> None:
    if log_w.dim() != 1 or log_w.shape[0] != X_ctx.shape[0]:
        raise ValueError(f"log_w must hold one entry per context row: {tuple(log_w.shape)} for {X_ctx.shape[0]} rows")
    if getattr(clf, "n_preprocessing_jobs", 1) != 1:
        raise ValueError("the preprocessing weights are installed in-process: n_preprocessing_jobs must be 1")
    if torch.isnan(log_w).any() or (log_w > 0).any():
        raise ValueError("log_w must be in [-inf, 0] (w in [0, 1]) and not NaN")


def evaluate_weighted(clf, X_ctx: torch.Tensor, y_ctx: torch.Tensor, X_q: torch.Tensor, y_q: torch.Tensor,
                      log_w: torch.Tensor, *, need_grad: bool, batch_size: int | None = None,
                      grad_channels: str = "all", query_ctx_rows=None, query_weights=None):
    """As gate_context.evaluate, with context row j weighted by w_j = exp(log_w[j]).

    Returns (mean CE of the query rows, d CE / d log_w or None, query probabilities [n_q, K]).
    The query rows are taken in chunks of batch_size; every chunk refits on the whole
    weighted context and adds its sum CE / n_q, which is exact (query rows never attend to
    each other). The gradient is taken with the weights still installed.

    grad_channels: "all" (default), "attn_label" (the statistics weighted but detached: no
    gradient through the z-norms and the other context-fitted statistics) or "attn" (the label
    mean detached as well): the forward pass is the same, only the gradient changes.
    query_ctx_rows: per query row, the context row it is a copy of (-1: none); a copy does not
    attend to its own context row (leave-one-out).
    query_weights: per query row, a constant weight on its CE (default 1; no gradient flows
    through it): the loss is then sum_q weight_q CE_q / n_q."""
    if grad_channels not in ("all", "attn", "attn_label"):
        raise ValueError(f"grad_channels must be all, attn or attn_label, not {grad_channels!r}")
    from .runtime import _forward, _install_recompute

    _check(clf, X_ctx, log_w)
    y_ctx = torch.as_tensor(y_ctx, dtype=torch.float32, device=X_ctx.device)
    y_q = torch.as_tensor(y_q, dtype=torch.long, device=X_ctx.device)
    n_q = len(y_q)
    bs = min(batch_size or n_q, n_q)
    qw = None if query_weights is None else torch.as_tensor(query_weights, dtype=torch.float32, device=X_ctx.device).detach()
    total, grad, chunks = 0.0, (torch.zeros_like(log_w) if need_grad else None), []
    for st in range(0, n_q, bs):
        with torch.enable_grad() if need_grad else torch.no_grad():
            lw = log_w if need_grad else log_w.detach()
            lw_stats = lw if grad_channels == "all" else lw.detach()
            lw_label = lw.detach() if grad_channels == "attn" else lw
            with preprocessing_weights_installed(lw_stats):
                clf.fit_with_differentiable_input(X_ctx, y_ctx)
                _install_recompute(clf)
                qrows = None if query_ctx_rows is None else np.asarray(query_ctx_rows)[st:st + bs]
                with model_weights_installed(clf, lw, label_log_w=lw_label, stats_log_w=lw_stats,
                                             query_ctx_rows=qrows):
                    p = _forward(clf, X_q[st:st + bs], need_grad=need_grad)
                    if qw is None:
                        loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), y_q[st:st + bs].to(p.device),
                                          reduction="sum") / n_q
                    else:
                        loss = (F.nll_loss(torch.log(p.clamp_min(1e-12)), y_q[st:st + bs].to(p.device),
                                           reduction="none") * qw[st:st + bs].to(p.device)).sum() / n_q
                    if need_grad:
                        grad += torch.autograd.grad(loss, log_w)[0]
        total += float(loss.detach())
        chunks.append(p.detach().float().cpu())
    return total, grad, torch.cat(chunks).numpy()


def probs_weighted(clf, X_ctx: torch.Tensor, y_ctx: torch.Tensor, X_q: torch.Tensor,
                   log_w: torch.Tensor) -> np.ndarray:
    """Query probabilities [n_q, K] with context row j weighted by exp(log_w[j]) (no gradient)."""
    dummy = torch.zeros(len(X_q), dtype=torch.long, device=X_ctx.device)
    return evaluate_weighted(clf, X_ctx, y_ctx, X_q, dummy, log_w, need_grad=False)[2]


def log_weights(w: torch.Tensor) -> torch.Tensor:
    """log w with log 0 = -inf, for weights given directly (w in [0, 1])."""
    w = torch.as_tensor(w, dtype=torch.float32)
    return torch.where(w > 0, torch.log(w.clamp_min(torch.finfo(torch.float32).tiny)),
                       torch.full_like(w, float("-inf")))

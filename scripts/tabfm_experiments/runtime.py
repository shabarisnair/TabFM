"""The single TabPFNv2 evaluation path shared by inference, attacks and context selection.

Every call refits on the context it is given (``fit_with_differentiable_input``), so
preprocessing statistics are always those of the current -- possibly poisoned --
prompt. There is deliberately no stat freezing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .config import inference_config

DETERMINISTIC_MODES = ("best-effort", "strict")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(mode: str) -> None:
    """``best-effort``: seeds + fixed config, fast kernels allowed. ``strict``: deterministic algorithms.

    Strict mode raises on any op without a deterministic kernel and disables the
    flash / memory-efficient SDP backends. Neither mode is bitwise reproducible across
    GPU models or driver versions.
    """
    if mode not in DETERMINISTIC_MODES:
        raise ValueError(f"deterministic mode must be one of {DETERMINISTIC_MODES}")
    torch.backends.cudnn.benchmark = False
    if mode == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        torch.use_deterministic_algorithms(False)


def device_from_gpu(gpu: str | int) -> str:
    return "cpu" if str(gpu) == "cpu" else f"cuda:{gpu}"


def build_tabpfn_v2(device: str, model_seed: int, deterministic_mode: str = "best-effort",
                    recompute_layers: bool = False):
    """TabPFNv2 classifier with the fixed, non-random differentiable configuration.

    ``recompute_layers`` turns on activation checkpointing for gradient forwards only
    (TabPFN's ``force_recompute_layer``): same loss and gradient, far less memory
    (WiDS 5k context + 1000 test rows: 73.8 GiB -> 9.6 GiB), ~25% slower per step.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    configure_determinism(deterministic_mode)
    seed_everything(model_seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=1,
        device=device,
        random_state=model_seed,
        inference_precision=torch.float32,
        differentiable_input=True,
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=1,
        inference_config=inference_config(),
    )
    # fit_with_differentiable_input cannot infer this from a float y tensor
    clf.n_classes_ = 2
    clf.tabfm_recompute_layers = bool(recompute_layers)
    return clf


def check_feature_width(clf, n_features: int) -> None:
    """Refuse to reuse a classifier across feature widths.

    ``fit_with_differentiable_input`` caches ``inferred_feature_schema_`` and
    ``ensemble_configs_`` on the first call and silently reuses them later; a wider
    table then fails deep inside preprocessing, a narrower one runs on a stale schema.
    """
    schema = getattr(clf, "inferred_feature_schema_", None)
    if schema is not None and len(schema.features) != n_features:
        raise ValueError(f"classifier was first fitted on {len(schema.features)} features but got "
                         f"{n_features}; build a new classifier (build_tabpfn_v2) per feature width")


def chunking_is_exact(X_ctx: torch.Tensor, X_test: torch.Tensor) -> bool:
    """Whether splitting ``X_test`` into chunks leaves every prediction unchanged.

    Test rows never attend to each other, and imputation / standard scaling are fitted
    on the context only. But TabPFNv2 fits its constant-feature masks over *all* rows
    (context + the test rows of the call). A column constant over the context therefore
    only yields chunk-independent masks if every test row also has that value.
    """
    first = X_ctx[0]
    const = (X_ctx == first).all(0)
    if not bool(const.any()):
        return True
    return bool((X_test[:, const] == first[const]).all())


def _forward(clf, X_test: torch.Tensor, *, need_grad: bool) -> torch.Tensor:
    """``clf.forward(X_test, use_inference_mode=True)`` on the context fitted just before.

    With ``differentiable_input=True`` TabPFN never enters inference mode, and its engine
    then hard-codes ``save_peak_mem = False`` (inference.py ``iter_outputs``). For
    gradient-free calls we lift ``differentiable_input`` for the forward only: same fitted
    executor and preprocessing, but TabPFN's memory-saving inference path. Outputs are
    identical (max |dp| = 0.0 measured on WiDS 2k/250 and in test_runtime_cpu), while
    peak memory for a 10k-row WiDS context drops from >93 GiB (OOM) to ~4 GiB.
    """
    if need_grad:
        with _recompute_active(clf, True):
            return clf.forward(X_test, use_inference_mode=True)
    clf.differentiable_input = False
    try:
        return clf.forward(X_test, use_inference_mode=True)
    finally:
        clf.differentiable_input = True


def evaluate_context(
    clf,
    X_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    need_grad: bool,
    test_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Returns (mean_ce, grad_wrt_X_ctx or None, probs [n_test, 2]).

    Always refits on the provided context. Chunks only X_test; each chunk refits and
    contributes ``sum_ce_chunk / n_test`` to the loss and its gradient.
    """
    check_feature_width(clf, X_ctx.shape[1])
    n_test = X_test.shape[0]
    bs = min(test_batch_size or n_test, n_test)
    if bs < n_test and not chunking_is_exact(X_ctx, X_test):
        warnings.warn("test chunking is not exact for this context: a context-constant feature "
                      "varies on the test rows (see runtime.chunking_is_exact)", stacklevel=2)
    xs = X_ctx.detach().float()
    if need_grad:
        xs = xs.requires_grad_(True)
    y_ctx = y_ctx.detach().float()
    y_test = y_test.detach().long()
    total = torch.zeros((), device=X_test.device)
    grad = torch.zeros_like(xs) if need_grad else None
    chunks = []
    for st in range(0, n_test, bs):
        en = min(st + bs, n_test)
        with torch.enable_grad() if need_grad else torch.no_grad():
            clf.fit_with_differentiable_input(xs, y_ctx)
            _install_recompute(clf)
            probs = _forward(clf, X_test[st:en], need_grad=need_grad)
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y_test[st:en].to(probs.device),
                              reduction="sum") / n_test
        if need_grad:
            (g,) = torch.autograd.grad(loss, xs)
            grad += g
        total = total + loss.detach().to(total.device)
        chunks.append(probs.detach())
        del probs, loss
    return total, grad, torch.cat(chunks)


def max_repeat_prob_delta(clf, X_ctx, y_ctx, X_test, y_test, *, n_repeats: int = 3,
                          test_batch_size: int | None = None) -> float:
    """Max |Δp| over ``n_repeats`` identical clean forwards (residual nondeterminism)."""
    ref = None
    worst = 0.0
    for _ in range(n_repeats):
        _, _, p = evaluate_context(clf, X_ctx, y_ctx, X_test, y_test, need_grad=False,
                                   test_batch_size=test_batch_size)
        if ref is None:
            ref = p
        else:
            worst = max(worst, float((p - ref).abs().max()))
    return worst


def target_embedding_influence_scores(
    clf,
    X_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    eligible,
    *,
    test_batch_size: int | None = None,
) -> np.ndarray:
    """First-order increase in mean test CE from flipping each context label.

    Hooks ``TabPFNV2._embed_targets`` on the loaded model instance and adds a zero
    tensor ``delta`` to the train rows' target embeddings, so ``g_i = dL/de_i`` is
    ``dL/d delta_i``. For binary labels densified to {0, 1}, flipping row i moves its
    embedding by ``(1 - 2 yflat_i) * W[:, 0]`` (``W = target_embedder.weight``; the
    NaN-indicator input is 0 for observed labels), so ``s_i = <g_i, (1-2 yflat_i) W[:,0]>``.

    Returns ``[n_ctx]`` scores; rows not in ``eligible`` (bool mask or index array) are -inf.
    """
    check_feature_width(clf, X_ctx.shape[1])
    n_ctx, n_test = X_ctx.shape[0], X_test.shape[0]
    bs = min(test_batch_size or n_test, n_test)
    xs = X_ctx.detach().float()
    y_f = y_ctx.detach().float()
    y_test = y_test.detach().long()

    with torch.no_grad():  # make sure models_ is loaded
        clf.fit_with_differentiable_input(xs, y_f)
    _install_recompute(clf)
    if len(clf.models_) != 1:
        raise RuntimeError("influence scores assume a single loaded model")
    arch = clf.models_[0]
    orig = arch._embed_targets
    captured: dict = {}

    def hooked(y, *, num_rows, num_train_labels, batch_size):
        emb, means, uniq = orig(y, num_rows=num_rows, num_train_labels=num_train_labels,
                                batch_size=batch_size)
        delta = torch.zeros(emb.shape[0], num_train_labels, emb.shape[2], device=emb.device,
                            dtype=emb.dtype, requires_grad=True)
        pad = torch.zeros(emb.shape[0], emb.shape[1] - num_train_labels, emb.shape[2],
                          device=emb.device, dtype=emb.dtype)
        captured["delta"], captured["unique_ys"], captured["calls"] = delta, uniq, captured.get("calls", 0) + 1
        return emb + torch.cat([delta, pad], dim=1), means, uniq

    g_acc = None
    arch._embed_targets = hooked
    try:
        for st in range(0, n_test, bs):
            en = min(st + bs, n_test)
            with torch.enable_grad():
                clf.fit_with_differentiable_input(xs, y_f)
                probs = _forward(clf, X_test[st:en], need_grad=True)
                loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y_test[st:en].to(probs.device),
                                  reduction="sum") / n_test
            (g,) = torch.autograd.grad(loss, captured["delta"])
            g = g[0].detach()  # batch_size == 1 -> [n_ctx, emsize]
            g_acc = g if g_acc is None else g_acc + g
    finally:
        del arch._embed_targets  # remove the instance override; class method is back
    if captured.get("calls", 0) == 0 or g_acc is None:
        raise RuntimeError("target-embedding hook was never called; TabPFN internals changed")
    if g_acc.shape[0] != n_ctx:
        raise RuntimeError(f"hooked embeddings have {g_acc.shape[0]} train rows, expected {n_ctx}")

    uniq = captured["unique_ys"][0].to(g_acc.device)
    if uniq.numel() < 2:
        warnings.warn("context has a single class; flip direction uses yflat=0 for all rows", stacklevel=2)
    y_dev = y_f.to(g_acc.device)
    yflat = (y_dev.unsqueeze(-1) > uniq).sum(-1).to(g_acc.dtype)
    w0 = arch.target_embedder.weight[:, 0].detach().to(g_acc.dtype)
    scores = ((g_acc * w0).sum(1) * (1.0 - 2.0 * yflat)).double().cpu().numpy()

    mask = np.zeros(n_ctx, dtype=bool)
    elig = np.asarray(eligible)
    if elig.dtype == bool:
        mask[:] = elig
    else:
        mask[elig.astype(int)] = True
    scores[~mask] = -np.inf
    return scores


# ------------------------------------------------------------- memory / batching
def _install_recompute(clf) -> None:
    """If requested, make the loaded model checkpoint its layers while a gradient forward runs."""
    if not getattr(clf, "tabfm_recompute_layers", False):
        return
    for arch in getattr(clf, "models_", []):
        if "get_default_performance_options" in vars(arch):
            continue
        orig = arch.get_default_performance_options

        def patched(orig=orig):
            opts = orig()
            if getattr(clf, "_tabfm_recompute_now", False):
                return dataclasses.replace(opts, force_recompute_layer=True)
            return opts

        arch.get_default_performance_options = patched


@contextlib.contextmanager
def _recompute_active(clf, active: bool):
    prev = getattr(clf, "_tabfm_recompute_now", False)
    clf._tabfm_recompute_now = bool(active and getattr(clf, "tabfm_recompute_layers", False))
    try:
        yield
    finally:
        clf._tabfm_recompute_now = prev


def batched_label_probs(
    clf,
    X_ctx: torch.Tensor,
    Y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    *,
    save_peak_memory_factor: int | None = 8,
) -> torch.Tensor:
    """Class probabilities ``[n_test, B, 2]`` for ``B`` label vectors ``Y_ctx [n_ctx, B]`` on one context.

    One fused forward of the TabPFNv2 architecture with batch dimension ``B`` (no
    gradients). It bypasses the sklearn wrapper, which is valid only for the fixed
    runtime configuration (identity preprocessing, one estimator, no class
    permutation, float32); callers should compare one column against
    ``evaluate_context`` (the label-flip GA does, every trial). Measured agreement:
    max |dp| ~5e-6 on LCLD 5k.
    """
    from tabpfn.architectures.interface import PerformanceOptions

    if not hasattr(clf, "models_"):
        with torch.no_grad():
            clf.fit_with_differentiable_input(X_ctx.detach().float(), Y_ctx[:, 0].detach().float())
    check_feature_width(clf, X_ctx.shape[1])
    if len(clf.models_) != 1 or clf.n_estimators != 1:
        raise RuntimeError("batched_label_probs assumes a single estimator and model")
    arch = clf.models_[0]
    dev = next(arch.parameters()).device
    B = Y_ctx.shape[1]
    # Feed the SAME preprocessed features the normal path feeds. The fitted ensemble
    # pipeline is not the identity: besides standardising (which the architecture's own
    # scaler would cancel), it DROPS degenerate/constant columns -- 1 on url_unique,
    # 2 on wids. Feeding raw X then changes the feature-group packing and the model sees
    # a structurally different input (max |dp| was 0.60 on url_unique). Using the
    # pipeline output makes this exactly equal to evaluate_context on all datasets.
    member = clf.executor_.ensemble_members[0]

    def _to(a):  # the pipeline returns torch tensors or numpy, on either device
        if isinstance(a, torch.Tensor):
            return a.detach().to(dev, torch.float32)
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)

    Xtr_p = _to(member.X_train)
    Xte_p = _to(member.transform_X_test(X_test.detach()))
    if Xtr_p.shape[0] != X_ctx.shape[0]:
        raise RuntimeError("fitted pipeline does not match the given context; refit first")
    x = torch.cat([Xtr_p, Xte_p])
    x = x[:, None, :].expand(-1, B, -1).contiguous()
    with torch.inference_mode():
        out = arch(x, Y_ctx.detach().to(dev, torch.float32), only_return_standard_out=True,
                   categorical_inds=[[] for _ in range(B)],
                   performance_options=PerformanceOptions(save_peak_memory_factor=save_peak_memory_factor))
    temp = getattr(clf, "softmax_temperature_", clf.softmax_temperature)
    return torch.softmax(out.float()[..., : clf.n_classes_] / temp, dim=-1)

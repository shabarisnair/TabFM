"""Context-poisoning attacks on TabPFNv2: X_train CAPGD and influence-ranked Y_train flips.

Both score on the same ``runtime.evaluate_context`` path used for clean inference.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .capgd import capgd, project_ball, random_start_point
from .constraints import (
    FeatureSpec,
    equality_post_step,
    fix_immutable_raw,
    relation_penalty_fn,
    repair_end,
    repair_end_budgeted,
)
from .metrics import BinaryMetrics, binary_metrics, deltas
from .ga import GAConfig, run_ga
from .runtime import batched_label_probs, evaluate_context, target_embedding_influence_scores
from .sampling import eligible_indices

CONSTRAINT_MODES = ("none", "box", "full")


@dataclass
class XCapgdConfig:
    norm: str = "L2"
    eps: float = 0.5
    eps_margin: float = 0.05
    n_iter: int = 40  # TabularBench: 10; see the n_iter ablation in docs/context_poisoning.md
    momentum: float = 0.75
    rho: float = 0.75
    n_restarts: int = 1
    eot_iter: int = 1
    constraints_mode: str = "full"
    constraint_penalty: float = 1.0
    fix_equality_iter: bool = True
    random_start: bool = False
    budget_repair: bool = True  # full mode: keep the repaired row within the eps-budget (re-project after repair)

    @property
    def eps_effective(self) -> float:
        return self.eps * (1.0 - self.eps_margin)

    @property
    def norm_name(self) -> str:
        return {"l2": "L2", "linf": "Linf"}.get(self.norm.lower(), self.norm)


@dataclass
class TrialResult:
    attack: str
    run_id: int
    subsample_id: int
    attacked_indices: np.ndarray
    k_requested: int
    k_actual: int
    clean: BinaryMetrics
    poisoned: BinaryMetrics
    delta: dict
    seconds: float
    history: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    x_delta_raw: np.ndarray | None = None      # [k, d] cell deltas for attacked rows
    y_poisoned: np.ndarray | None = None       # full flipped label vector (label-flip)

    def to_json(self) -> dict:
        return {
            "attack": self.attack, "run_id": self.run_id, "subsample_id": self.subsample_id,
            "attacked_indices": self.attacked_indices, "k_requested": self.k_requested,
            "k_actual": self.k_actual, "clean": self.clean.as_dict(),
            "poisoned": self.poisoned.as_dict(), "delta": self.delta,
            "seconds": round(self.seconds, 2), "history": self.history, **self.extra,
        }


def _metrics(probs: torch.Tensor, y_test: torch.Tensor) -> BinaryMetrics:
    return binary_metrics(probs.float().cpu().numpy(), y_test.cpu().numpy())


# ------------------------------------------------------------------ X_train CAPGD
def run_x_capgd(
    clf,
    X_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    rows: np.ndarray,
    *,
    spec: FeatureSpec,
    constraints,
    cfg: XCapgdConfig,
    run_id: int,
    subsample_id: int,
    attack_seed: int,
    clean_metrics: BinaryMetrics,
    k_requested: int,
    test_batch_size: int | None = None,
    log=None,
) -> TrialResult:
    """CAPGD on the scaled features of context ``rows``; maximises mean test CE.

    ``constraints_mode``: ``none`` = eps-ball only, every feature mutable;
    ``box`` = + [0,1] scaled box, immutable mask; ``full`` = box + relation penalty,
    in-loop equality fixing (``fix_equality_iter``) and end repair
    (``fix_types -> fix_immutable -> fix_equality``). Labels of attacked rows stay in
    the context and never enter the loss.
    """
    if cfg.constraints_mode not in CONSTRAINT_MODES:
        raise ValueError(f"constraints_mode must be one of {CONSTRAINT_MODES}")
    if cfg.constraints_mode == "full" and constraints is None:
        raise ValueError("constraints_mode='full' needs a Constraints object")
    t0 = time.time()
    dev = X_ctx.device
    rows_t = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=dev)
    x_clean_raw = X_ctx[rows_t].clone()
    x0 = spec.to_scaled(x_clean_raw)
    mode = cfg.constraints_mode
    box = mode != "none"
    norm = cfg.norm_name
    eps = cfg.eps_effective
    mut = (spec.mutable.float() if box else torch.ones(x0.shape[1], device=dev)).unsqueeze(0)
    penalty_fn = relation_penalty_fn(constraints) if mode == "full" else None
    post_step = equality_post_step(spec, constraints) if (mode == "full" and cfg.fix_equality_iter) else None
    lam = cfg.constraint_penalty

    def splice(x_raw: torch.Tensor) -> torch.Tensor:
        X = X_ctx.detach().clone()
        X[rows_t] = x_raw.detach().to(X.dtype)
        return X

    def penalty(x_scaled: torch.Tensor, need_grad: bool):
        if penalty_fn is None:
            return 0.0, None
        xl = x_scaled.detach().clone().requires_grad_(need_grad)
        with torch.enable_grad() if need_grad else torch.no_grad():
            pen = torch.nan_to_num(penalty_fn(spec.to_raw(xl)), nan=0.0, posinf=0.0, neginf=0.0).mean()
        g = torch.autograd.grad(pen, xl)[0] if need_grad and pen.requires_grad else None
        return float(pen.detach()), g

    def evaluate(x_scaled: torch.Tensor, need_grad: bool):
        loss_sum, grad_sum = 0.0, None
        for _ in range(cfg.eot_iter):
            ce, g_ctx, _ = evaluate_context(clf, splice(spec.to_raw(x_scaled)), y_ctx, X_test, y_test,
                                            need_grad=need_grad, test_batch_size=test_batch_size)
            pen, g_pen = penalty(x_scaled, need_grad)
            loss_sum += float(ce) - lam * pen
            if need_grad:
                g = g_ctx[rows_t] * spec.rng  # chain rule: d raw / d scaled = range
                if g_pen is not None:
                    g = g - lam * g_pen
                grad_sum = g if grad_sum is None else grad_sum + g
        grad = None
        if need_grad:
            grad = torch.nan_to_num(grad_sum / cfg.eot_iter, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(loss_sum / cfg.eot_iter), grad

    def repair(x1, x_ref, pad):
        x1 = project_ball(x1, x_ref, eps, norm, pad)
        return x1.clamp(0.0, 1.0) if box else x1

    def show(rec, improved):
        if log is not None:
            log.info(f"    iter {rec['iter']:>3}  obj {rec['loss']:.5f}  best {rec['best']:.5f}  "
                     f"step {rec['step_size']:.4f}" + ("  *" if improved else "") + ("  halved" if rec["halved"] else ""))

    # Budget-aware end repair (default for full mode): after fix_types/fix_immutable/
    # fix_equality, re-project any over-budget row back onto the eps-ball and re-round, so
    # the reported row stays within budget. Prioritises the budget over exact equality
    # (they cannot both hold with integer features). Turn off with budget_repair=False to
    # reproduce the TabularBench behaviour (repair can leave the ball).
    use_budget = cfg.budget_repair and mode == "full"

    def repair_primary(raw_pre):
        if mode == "full":
            if use_budget:
                return repair_end_budgeted(x_clean_raw, raw_pre, spec, constraints, eps,
                                           norm=norm, fix_equality=cfg.fix_equality_iter)
            return repair_end(x_clean_raw, raw_pre, constraints, fix_equality=True)
        if mode == "box":
            return fix_immutable_raw(x_clean_raw, raw_pre, spec.mutable)
        return raw_pre

    restarts = []
    best = None
    for restart in range(cfg.n_restarts):
        random = cfg.random_start or run_id > 0 or restart > 0
        x_init = None
        if random:
            gen = torch.Generator().manual_seed(int(attack_seed) * 1000 + restart)
            x_init = random_start_point(x0, eps=eps, norm=norm, mut=mut, generator=gen)
        x_best, obj_best, hist = capgd(
            evaluate, x0, eps=eps, n_iter=cfg.n_iter, rho=cfg.rho, momentum=cfg.momentum,
            repair=repair, mut=mut, better=lambda u, v: u > v, box=box, norm=norm,
            on_iter=show, x_init=x_init, post_step=post_step,
        )
        # Cells CAPGD never moved keep their exact clean value (no scale round-trip noise).
        raw_pre = torch.where(x_best == x0, x_clean_raw, spec.to_raw(x_best))
        raw = repair_primary(raw_pre)
        ce_f, _, probs_f = evaluate_context(clf, splice(raw), y_ctx, X_test, y_test, need_grad=False,
                                            test_batch_size=test_batch_size)
        info = {"restart": restart, "random_start": random, "objective_pre_repair": obj_best,
                "ce_post_repair": float(ce_f), "history": hist}
        restarts.append(info)
        if best is None or float(ce_f) > best[0]:
            best = (float(ce_f), raw, probs_f, info, raw_pre)

    _, raw_final, probs_final, best_info, raw_pre_best = best
    poisoned = _metrics(probs_final, y_test)
    d_raw = (raw_final - x_clean_raw).detach()
    d_scaled = spec.to_scaled(raw_final) - x0
    max_l2 = float(d_scaled.pow(2).sum(1).sqrt().max())
    extra = {
        "constraints_mode": mode, "norm": norm, "eps": cfg.eps, "eps_effective": eps,
        "budget_repair": bool(use_budget), "best_restart": best_info["restart"],
        # Diagnostic: norm of the perturbation BEFORE any end repair. The in-loop
        # projection should hold this at <= eps regardless of n_iter; if it grows with
        # n_iter the projection is not binding.
        "max_l2_delta_scaled_pre_repair": float(
            (spec.to_scaled(raw_pre_best) - x0).pow(2).sum(1).sqrt().max()),
        "restarts": [{k: v for k, v in r.items() if k != "history"} for r in restarts],
        "n_cells_changed": int((d_raw.abs() > 0).sum()),
        "max_l2_delta_scaled": max_l2,
        "max_linf_delta_scaled": float(d_scaled.abs().max()),
        "within_eps_after_repair": bool(
            (max_l2 if norm == "L2" else float(d_scaled.abs().max())) <= eps + 1e-5),
    }
    # For comparison, also record the plain (TabularBench) repair on the same iterate.
    if use_budget:
        raw_std = repair_end(x_clean_raw, raw_pre_best, constraints, fix_equality=True)
        _, _, probs_std = evaluate_context(clf, splice(raw_std), y_ctx, X_test, y_test, need_grad=False,
                                           test_batch_size=test_batch_size)
        m_std = _metrics(probs_std, y_test)
        dstd = spec.to_scaled(raw_std) - x0
        extra["standard_repair"] = {
            "poisoned": m_std.as_dict(), "delta": deltas(clean_metrics, m_std),
            "max_l2_delta_scaled": float(dstd.pow(2).sum(1).sqrt().max()),
            "within_eps": bool(float(dstd.pow(2).sum(1).sqrt().max()) <= eps + 1e-5),
        }

    if penalty_fn is not None:
        with torch.no_grad():
            v_clean = torch.nan_to_num(penalty_fn(x_clean_raw), nan=0.0)
            v_final = torch.nan_to_num(penalty_fn(raw_final), nan=0.0)
        extra["relation_violation_clean_mean"] = float(v_clean.mean())
        extra["relation_violation_final_mean"] = float(v_final.mean())
        extra["relation_violated_rows_final"] = int((v_final > 1e-6).sum())
    return TrialResult(
        attack="x-capgd", run_id=run_id, subsample_id=subsample_id,
        attacked_indices=np.asarray(rows), k_requested=k_requested, k_actual=len(rows),
        clean=clean_metrics, poisoned=poisoned, delta=deltas(clean_metrics, poisoned),
        seconds=time.time() - t0, history=best_info["history"], extra=extra,
        x_delta_raw=d_raw.float().cpu().numpy(),
    )


def apply_x_delta(X_ctx: np.ndarray, rows: np.ndarray, d_raw: np.ndarray) -> np.ndarray:
    X = np.array(X_ctx, dtype=np.float64, copy=True)
    X[rows] = X[rows] + d_raw
    return X


# ------------------------------------------------------------ Y_train label flips
def run_label_flip_influence(
    clf,
    X_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    k: int,
    attack_class: int | None,
    run_id: int,
    subsample_id: int,
    clean_metrics: BinaryMetrics,
    test_batch_size: int | None = None,
    log=None,
) -> TrialResult:
    """Flip the ``k`` eligible labels with the largest first-order CE increase.

    Ranking covers every eligible row (it *is* the row choice), so repeated trials
    flip the identical set and differ only by residual GPU nondeterminism.
    """
    t0 = time.time()
    y_np = y_ctx.detach().cpu().numpy().astype(int)
    elig = eligible_indices(y_np, attack_class)
    if len(elig) == 0:
        raise ValueError(f"no context rows with y == {attack_class}")
    scores = target_embedding_influence_scores(clf, X_ctx, y_ctx, X_test, y_test, elig,
                                               test_batch_size=test_batch_size)
    k_actual = min(k, len(elig))
    order = np.argsort(-scores, kind="stable")[:k_actual]
    flip = np.sort(order)
    y_flip = y_ctx.detach().clone().float()
    flip_t = torch.as_tensor(flip, dtype=torch.long, device=y_flip.device)
    y_flip[flip_t] = 1.0 - y_flip[flip_t]
    y_flip_np = y_flip.cpu().numpy().astype(int)
    if len(np.unique(y_flip_np)) < 2 and log is not None:
        log.info("  WARNING: flipped context has a single class")
    _, _, probs = evaluate_context(clf, X_ctx, y_flip, X_test, y_test, need_grad=False,
                                   test_batch_size=test_batch_size)
    poisoned = _metrics(probs, y_test)
    fin = scores[np.isfinite(scores)]
    extra = {
        "flipped_indices": flip,
        "flipped_scores": scores[flip],
        "first_order_ce_gain_sum": float(scores[flip].sum()),
        "n_eligible": int(len(elig)),
        "score_stats": {"max": float(fin.max()), "min": float(fin.min()), "mean": float(fin.mean()),
                        "n_positive": int((fin > 0).sum())},
        "class_counts_clean": np.bincount(y_np, minlength=2),
        "class_counts_poisoned": np.bincount(y_flip_np, minlength=2),
    }
    return TrialResult(
        attack="label-flip-influence", run_id=run_id, subsample_id=subsample_id,
        attacked_indices=flip, k_requested=k, k_actual=k_actual,
        clean=clean_metrics, poisoned=poisoned, delta=deltas(clean_metrics, poisoned),
        seconds=time.time() - t0, extra=extra, y_poisoned=y_flip_np,
    )


# ------------------------------------------------------ Y_train genetic algorithm
def run_label_flip_ga(
    clf,
    X_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    pool_rows: np.ndarray,
    k: int,
    *,
    cfg: GAConfig,
    run_id: int,
    subsample_id: int,
    ga_seed: int,
    clean_metrics: BinaryMetrics,
    seed_influence: bool = True,
    test_batch_size: int | None = None,
    log=None,
) -> TrialResult:
    """Genetic search for the <=k rows of ``pool_rows`` whose label flips maximise test CE.

    The pool is the whole eligible set (all rows, or one class); the GA searches
    k-subsets of it (budget = k flips anywhere), so it is not boxed into a random
    slice. Generation 0 is seeded with the influence top-k mask (``seed_influence``)
    and random k-subsets. Fitness is exact mean test CE, scored ``cfg.batch_size``
    individuals per fused forward (``runtime.batched_label_probs``), optionally on
    ``cfg.fitness_test_rows`` random test rows. The winner, the influence top-k, and
    "flip all k lowest-loss" baselines are re-scored on all test rows with
    ``evaluate_context``. Randomness: population / operators (``ga_seed``).
    """
    t0 = time.time()
    dev = X_ctx.device
    pool = np.sort(np.asarray(pool_rows))
    npool = len(pool)
    k = int(min(k, npool))
    rng = np.random.default_rng(ga_seed)
    n_test = X_test.shape[0]
    if cfg.fitness_test_rows and cfg.fitness_test_rows < n_test:
        fit_idx = np.sort(rng.choice(n_test, cfg.fitness_test_rows, replace=False))
    else:
        fit_idx = np.arange(n_test)
    fit_idx_t = torch.as_tensor(fit_idx, device=dev)
    Xt_fit, yt_fit = X_test[fit_idx_t], y_test[fit_idx_t].long()
    pool_t = torch.as_tensor(pool, dtype=torch.long, device=dev)
    y_f = y_ctx.detach().float()
    state = {"bs": max(1, cfg.batch_size)}

    # Influence top-k seed and baseline.
    scores = target_embedding_influence_scores(clf, X_ctx, y_ctx, X_test, y_test, pool,
                                               test_batch_size=test_batch_size)
    pool_scores = scores[pool]
    infl_topk = pool[np.argsort(-pool_scores, kind="stable")[:k]]
    seed_masks = None
    if seed_influence:
        m = np.zeros(npool, dtype=bool)
        m[np.argsort(-pool_scores, kind="stable")[:k]] = True
        seed_masks = [m]

    def flipped_labels(masks: np.ndarray) -> torch.Tensor:
        """``[B, npool]`` masks -> ``[n_ctx, B]`` label columns."""
        M = torch.as_tensor(masks, dtype=torch.bool, device=dev)
        F = torch.zeros(y_f.shape[0], M.shape[0], dtype=torch.bool, device=dev)
        F[pool_t] = M.T
        base = y_f[:, None].expand(-1, M.shape[0])
        return torch.where(F, 1.0 - base, base)

    def fitness(masks: np.ndarray) -> np.ndarray:
        out = np.empty(len(masks))
        i = 0
        while i < len(masks):
            bs = state["bs"]
            chunk = masks[i:i + bs]
            try:
                probs = batched_label_probs(clf, X_ctx, flipped_labels(chunk), Xt_fit)
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                state["bs"] = bs // 2
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if log is not None:
                    log.info(f"    GA batch OOM; batch size -> {state['bs']}")
                continue
            idx = yt_fit.to(probs.device).view(-1, 1, 1).expand(-1, probs.shape[1], 1)
            p_true = probs.gather(2, idx).squeeze(2)
            out[i:i + len(chunk)] = (-torch.log(p_true.clamp_min(1e-12))).mean(0).double().cpu().numpy()
            i += len(chunk)
        return out

    # Self-check: fused batch vs the reference path on the influence top-k mask.
    # The two use different memory-chunking paths, so their absolute CE differs by a
    # small, mask-deterministic offset (float accumulation in chunked attention); this
    # does not affect the ranking, and the winner is always re-scored with
    # evaluate_context below. Only a gross gap (wrong class order / temperature) aborts.
    seed0 = seed_masks[0] if seed_masks else (np.arange(npool) < k)
    y_seed = flipped_labels(seed0[None])[:, 0]
    ref_ce, _, _ = evaluate_context(clf, X_ctx, y_seed, Xt_fit, yt_fit, need_grad=False)
    check_gap = abs(float(fitness(seed0[None])[0]) - float(ref_ce))
    if check_gap > 0.05:
        raise RuntimeError(f"batched fitness disagrees with evaluate_context by {check_gap:.2e} "
                           "(expected a small float-path offset; this is a gross mismatch)")
    if check_gap > 2e-3 and log is not None:
        log.info(f"    note: batched-fitness vs evaluate_context CE offset {check_gap:.2e} "
                 "(ranking-only; final metrics use evaluate_context)")

    def show(rec):
        if log is not None:
            log.info(f"    gen {rec['generation']:>3}  best {rec['best']:.5f}  mean {rec['mean']:.5f}  "
                     f"best_so_far {rec['best_so_far']:.5f}  flips {rec['best_size']}  evals {rec['n_evals']}")

    res = run_ga(fitness, npool, cfg, rng, max_active=k, seed_masks=seed_masks, on_generation=show)
    flip = np.sort(pool[res["best_mask"]])
    def flip_and_score(rows):
        yy = y_f.clone()
        rt = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=dev)
        yy[rt] = 1.0 - yy[rt]
        _, _, pr = evaluate_context(clf, X_ctx, yy, X_test, y_test, need_grad=False,
                                    test_batch_size=test_batch_size)
        return yy, _metrics(pr, y_test)

    y_flip, poisoned = flip_and_score(flip)
    _, m_infl = flip_and_score(infl_topk)
    y_np = y_f.cpu().numpy().astype(int)
    y_flip_np = y_flip.cpu().numpy().astype(int)
    extra = {
        "pool_size": int(npool), "budget_k": k, "flipped_indices": flip, "n_flipped": int(len(flip)),
        "ga": asdict(cfg), "ga_seed": ga_seed, "fitness_test_rows": int(len(fit_idx)),
        "seeded_influence": bool(seed_influence),
        "n_fitness_evals": res["n_evals"], "generations_run": res["generations_run"],
        "fitness_best": res["best_fitness"], "fitness_initial_best": res["initial_best_fitness"],
        "batched_check_abs_dce": check_gap, "ga_history": res["history"],
        "influence_topk": {"indices": infl_topk, "poisoned": m_infl.as_dict(),
                           "delta": deltas(clean_metrics, m_infl)},
        "class_counts_clean": np.bincount(y_np, minlength=2),
        "class_counts_poisoned": np.bincount(y_flip_np, minlength=2),
    }
    return TrialResult(
        attack="label-flip-ga", run_id=run_id, subsample_id=subsample_id,
        attacked_indices=flip, k_requested=k, k_actual=int(len(flip)),
        clean=clean_metrics, poisoned=poisoned, delta=deltas(clean_metrics, poisoned),
        seconds=time.time() - t0, extra=extra, y_poisoned=y_flip_np,
    )

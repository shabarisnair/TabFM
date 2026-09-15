"""CAPGD optimiser core, free of TabPFN / poisoning specifics.

Ported from ``tabularbench/attacks/capgd/capgd.py::CAPGD.attack_single_run`` and
checked iterate-for-iterate against it by ``scripts/check_capgd_equiv.py``.
"""

from __future__ import annotations

import torch


def project_ball(x_cand: torch.Tensor, x_ref: torch.Tensor, eps: float, norm: str, pad: bool) -> torch.Tensor:
    """Project onto the per-row eps-ball around ``x_ref``.

    Linf clamps each coordinate. L2 rescales each row's perturbation to norm <= eps;
    ``pad`` reproduces the reference's 1e-12 asymmetry between its two L2 projections.
    """
    if norm == "Linf":
        return torch.min(torch.max(x_cand, x_ref - eps), x_ref + eps)
    if norm != "L2":
        raise ValueError(f"norm must be Linf or L2, got {norm}")
    d = x_cand - x_ref
    dims = tuple(range(1, d.dim()))
    n = d.pow(2).sum(dim=dims, keepdim=True).sqrt()
    return x_ref + d / (n + 1e-12) * torch.min(torch.full_like(d, eps), n + (1e-12 if pad else 0.0))


def random_start_point(
    x: torch.Tensor, *, eps: float, norm: str, mut: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Random start inside the eps-ball, as in the reference (before its [0,1] clamp).

    Linf: ``x + mut * eps * t / max|t|`` with ``t ~ U(-1, 1)``.
    L2:   ``x + mut * eps * t / ||t||`` with ``t ~ N(0, 1)``.
    Draws on CPU from ``generator`` so the start is device-independent.
    """
    if norm == "Linf":
        t = 2 * torch.rand(x.shape, generator=generator) - 1
        t = t.to(x.device, x.dtype)
        scale = t.reshape(t.shape[0], -1).abs().max(dim=1, keepdim=True)[0].reshape(-1, *([1] * (x.dim() - 1)))
        return x + mut * (eps * t / scale)
    if norm == "L2":
        t = torch.randn(x.shape, generator=generator).to(x.device, x.dtype)
        dims = tuple(range(1, x.dim()))
        return x + mut * (eps * t / (t.pow(2).sum(dim=dims, keepdim=True).sqrt() + 1e-12))
    raise ValueError(f"norm must be Linf or L2, got {norm}")


def capgd(evaluate, x0, *, eps, n_iter, rho, momentum, repair, mut, better, box,
          norm="L2", on_iter=None, x_init=None, post_step=None):
    """CAPGD. ``evaluate(x_scaled, need_grad) -> (loss, grad)``. Returns ``x_best, loss_best, hist``.

    ``x0`` is the clean point (the eps-ball centre). ``x_init`` (default ``x0``) is the
    starting iterate, e.g. from :func:`random_start_point`. ``repair(x_candidate, x_ref,
    pad)`` projects onto the eps-ball (and the box, if enabled). ``post_step(x)``, if
    given, runs after each gradient step and before evaluation -- where the reference
    applies ``fix_equality_constraints`` when ``fix_equality_constraints_iter`` is on.

    Linf and L2 differ in three places, all taken from the reference:
      * the step direction is sign(grad) vs the row-normalised grad;
      * the mutable mask is applied to neither momentum term under Linf, and to
        both of them under L2;
      * the ball projection is per-coordinate clamping vs a radial rescale.
    """
    if norm not in ("Linf", "L2"):
        raise ValueError(f"norm must be Linf or L2, got {norm}")
    x_ref = x0.clone()
    x_adv = (x0 if x_init is None else x_init).clone()
    if box:
        x_adv = x_adv.clamp(0.0, 1.0)

    loss, grad = evaluate(x_adv, True)
    loss_best, x_best, grad_best = float(loss), x_adv.clone(), grad.clone()
    step_size = torch.full_like(x_adv, 2.0 * eps)
    x_adv_old = x_adv.clone()

    steps_2 = max(int(0.22 * n_iter), 1)
    steps_min = max(int(0.06 * n_iter), 1)
    size_decr = max(int(0.03 * n_iter), 1)
    k, counter3 = steps_2, 0
    # Fixed-size zero-filled buffer, exactly as the reference allocates it. At the
    # first checkpoint the oscillation check indexes loss_steps[-1], which wraps to a
    # still-unwritten 0.0; Python lists wrap negative indices the same way.
    loss_steps = [0.0] * n_iter
    reduced_last_check, loss_best_last_check = True, loss_best

    hist = [{"iter": 0, "loss": float(loss), "best": loss_best,
             "step_size": float(step_size.mean()), "halved": False}]
    if on_iter:
        on_iter(hist[0], False)

    for i in range(n_iter):
        with torch.no_grad():
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            mom = momentum if i > 0 else 1.0
            if norm == "Linf":
                x1 = repair(x_adv + mut * (step_size * torch.sign(grad)), x_ref, False)
                x1 = repair(x_adv + (x1 - x_adv) * mom + grad2 * (1 - mom), x_ref, False)
            else:
                dims = tuple(range(1, grad.dim()))
                gn = (grad.pow(2) * mut).sum(dim=dims, keepdim=True).sqrt() + 1e-12
                x1 = repair(x_adv + mut * (step_size * grad / gn), x_ref, False)
                x1 = repair(x_adv + mut * (x1 - x_adv) * mom + mut * grad2 * (1 - mom),
                            x_ref, True)
            x_adv = x1
            if post_step is not None:
                x_adv = post_step(x_adv)

        loss, grad = evaluate(x_adv, True)
        lv = float(loss)
        loss_steps[i] = lv
        improved = better(lv, loss_best)
        if improved:
            loss_best, x_best, grad_best = lv, x_adv.clone(), grad.clone()

        counter3 += 1
        halved = False
        if counter3 == k:
            t = sum(1 for c in range(k) if better(loss_steps[i - c], loss_steps[i - c - 1]))
            osc = t <= k * rho
            no_impr = (not reduced_last_check) and not better(loss_best, loss_best_last_check)
            fl = osc or no_impr
            reduced_last_check, loss_best_last_check = fl, loss_best
            if fl:
                step_size = step_size / 2.0
                x_adv, grad = x_best.clone(), grad_best.clone()
                halved = True
            counter3, k = 0, max(k - size_decr, steps_min)

        rec = {"iter": i + 1, "loss": lv, "best": loss_best,
               "step_size": float(step_size.mean()), "halved": halved}
        hist.append(rec)
        if on_iter:
            on_iter(rec, improved)

    return x_best, loss_best, hist

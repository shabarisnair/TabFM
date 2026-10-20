"""The learned row filters of the tiny-V benchmark (bench_tinyv.py), at fixed settings.

Both learn on the clean rows V alone (no held-out part: at one row per class there is none), for a
fixed, small number of updates, and never see the test set or the poisoned-row identity. With so few
clean rows the objective is easy to overfit, so every learner stops after 20 updates (per restart for
the gate). Each returns the k rows it switches off; the benchmark scores V + the other rows.

  attn_weights   per-row attention weights (tabfm_experiments/row_weights.py: log w in all 12 blocks,
                 the weighted label mean and z-norms, the discrete fits ignoring w = 0), trained by
                 attn_learn.py's code with its fitted validation rows set to all of V. The two
                 configurations of the attention study (results/tables/attn_learn/findings.md):
                   topk  (M1) hard top-k, the gradient at the relaxed point 0.05, PGD-L-inf lr 0.2,
                         started from the V-filter ranking; 20 updates
                   soft  (A3) log w in [-1000, 0], Adam lr 0.1, started from the V-filter ranking, no
                         prior pull; 20 updates
                 (The study ran 200 updates with V = 50/500 and chose the iteration on held-out V.)
                 Deployed hard: the k lowest-ranked rows removed. The soft variant also reports its
                 soft weights with V added at weight 1 (extra, not the headline).
  nan_topk_gate  the feature-input gate (gate_context.py --gate topk): context rows switched off by
                 setting their encoder input to the all-missing input, exactly k rows at a time,
                 random N(0, 1) scores, identity straight-through with the gradient taken at the
                 relaxed point 0.05, L-inf score steps of lr 5, 20 updates per restart, 4 restarts
                 (seeds 0-3) combined by a vote of their final masks (the k rows masked by the most
                 restarts; ties: mean within-restart score rank) -- the version of results/tables/
                 gate_topk/night_log.md that needs no held-out rows (which ran 300 updates with V = 500).
"""
from __future__ import annotations

import time

import numpy as np
import torch

ATTN_CONFIGS = {
    "topk": dict(param="topk", opt="pgd-linf", lr=0.2, probe=0.05, init="vfilter", steps=20),
    "soft": dict(param="log", opt="adam", lr=0.1, floor=-1000.0, init="vfilter", prior=0.0, steps=20),
    # the best soft form of results/tables/attn_form/findings.md: w in [0, 1], V-filter start, PGD-L-inf lr 0.8;
    # deployed soft (w = 0 deletes a row exactly); its update count comes with the job name (attn_wsoft_it<N>)
    "wsoft": dict(param="w", opt="pgd-linf", lr=0.8, init="vfilter", prior=0.0, steps=20),
}
NAN_GATE = dict(restarts=4, steps=20, lr=5.0, probe=0.05, score_std=1.0)


def attn_weights(clf, d: dict, variant: str, *, steps: int | None = None, lr: float | None = None, log=print) -> dict:
    """Learn the weights on all of V; returns the removed rows (k) and, for the soft variant, the final
    log-weights (for the extra soft deployment)."""
    import attn_learn as A
    from tabfm_experiments.row_weights import evaluate_weighted
    c = dict(A.DEFAULTS) | ATTN_CONFIGS[variant] | {"seed": d["seed"]}
    if steps is not None:
        c["steps"] = steps
    if lr is not None:
        c["lr"] = lr
    device = d["Xc"].device
    dd = {"n": d["n"], "Xc": d["Xc"], "yp": d["y_given"], "Xf": d["Xv"], "yf": d["yv"]}  # fitted rows = all of V
    torch.manual_seed(c["seed"])
    p = A.Param(c["param"], A.init_raw(c, dd, clf, device), c, d["k"])
    p.project()
    raw_start = p.raw.detach().clone()
    st = A.Stepper(p, c)
    t0 = time.time()
    for it in range(c["steps"]):
        lw = p.probe_log_w() if c["param"] == "topk" else p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        loss, g, _ = evaluate_weighted(clf, d["Xc"], d["y_given"], d["Xv"], d["yv"], lw_d, need_grad=True,
                                       grad_channels="all")
        if c["param"] == "topk":
            g_raw = g
        else:
            p.raw.grad = None
            lw.backward(g)
            g_raw = torch.nan_to_num(p.raw.grad.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        if c["prior"] > 0:
            g_raw = g_raw / g_raw.pow(2).mean().sqrt().clamp_min(1e-30) + c["prior"] * (p.raw.detach() - raw_start)
        st.step(g_raw)
        if it % 5 == 0:
            log(f"      attn_{variant} it {it}: CE on V {loss:.4f}  ({time.time() - t0:.0f} s)")
    removed = p.deleted().cpu().numpy()
    out = {"removed": np.flatnonzero(removed), "steps": c["steps"], "config": c, "seconds": round(time.time() - t0, 1)}
    if c["param"] != "topk":
        out["log_w"] = p.log_w().detach()
    return out


def nan_topk_gate(clf, d: dict, *, restarts: int | None = None, steps: int | None = None, log=print) -> dict:
    """The NaN-input top-k gate, trained on all of V; the vote of the restarts' final masks."""
    import gate_context as G
    cfg = dict(NAN_GATE) | {k: v for k, v in (("restarts", restarts), ("steps", steps)) if v is not None}
    device = d["Xc"].device
    n, k = d["n"], d["k"]
    yp_t = torch.as_tensor(d["y_given"], dtype=torch.float32, device=device)
    yv_t = torch.as_tensor(d["yv"], dtype=torch.long, device=device)
    null_x = G.capture_null_input(clf, d["Xc"], yp_t)
    masks, ranks, t0 = [], [], time.time()
    for r in range(cfg["restarts"]):
        scores = G.init_scores(n, cfg["score_std"], r, device)
        for it in range(cfg["steps"]):
            g = G.topk_mask(scores, k)
            probe = (g * (1 - cfg["probe"]) + (1 - g) * cfg["probe"]).requires_grad_(True)
            ce, grad, _ = G.evaluate(clf, d["Xc"], yp_t, d["Xv"], yv_t, probe, null_x, need_grad=True, batch_size=None)
            G.score_step(scores, G.ste_grad(grad, scores, "identity", k_remove=k), cfg["lr"], "linf")
            if it % 5 == 0:
                log(f"      nan_topk restart {r} it {it}: CE on V (relaxed) {ce:.4f}  ({time.time() - t0:.0f} s)")
        s = scores.detach().cpu().numpy()
        masks.append(G.topk_mask(scores, k).cpu().numpy())
        ranks.append(np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(n - 1, 1))  # 0 = lowest score
    votes = sum(1 - m for m in masks)
    mean_rank = np.mean(ranks, axis=0)
    removed = np.sort(np.lexsort((mean_rank, -votes))[:k])
    return {"removed": removed, "config": cfg, "votes_hist": np.bincount(votes.astype(int), minlength=cfg["restarts"] + 1).tolist(),
            "seconds": round(time.time() - t0, 1)}


def soft_plus_v_probs(clf, d: dict, log_w: torch.Tensor, Xt: torch.Tensor, yt: np.ndarray) -> np.ndarray:
    """Test probabilities with the soft weights on the poisoned rows and V added at weight 1."""
    from tabfm_experiments.row_weights import evaluate_weighted
    X = torch.cat([d["Xc"], d["Xv"]])
    y = np.concatenate([d["y_given"], d["yv"]])
    lw = torch.cat([log_w, torch.zeros(len(d["yv"]), device=log_w.device)])
    return evaluate_weighted(clf, X, y, Xt, yt, lw, need_grad=False)[2]


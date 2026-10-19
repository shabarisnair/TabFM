#!/usr/bin/env python
"""Step schedules for growing a trusted set from a tiny clean set: which one is robust?

Greedy growth: the trusted set T starts as the V clean rows; every step TabPFN with T as its context
scores each pool row's given label (CE = -log p, optionally class-prior corrected toward the
context's label mix: cprior), and the m most plausible rows join T, until k = R% of the context
is left out; the final context is T. A schedule decides m at every step:

  oneshot               the V-filter: every row decided by V alone (m = n - k)
  fixed:<variant>       iterative_filter.py's grow variants, e.g. fixed:pct1 (m = 1% of n),
                        fixed:pct10, fixed:frac0.1 (m = 10% of |T|), fixed:frac0.1:cprior
  slope:d<delta>[...]   adaptive: re-scorings per doubling of T chosen from the learning curve.
                        Every time T has grown by --refit, TabPFN is cross-validated on T: a fixed
                        random quarter of T is held out and predicted from random subsets of the
                        rest of 1/8, 1/4, 1/2 and all of it; the slope of held-out CE against
                        log2(subset size) is the CE gained per doubling, g. Then
                          r = clamp(g / delta, rmin, rmax)   re-scorings per doubling
                          w = 1 - 2^(-1/r),  m = ceil(w |T| / (1 - w))
                        Before T has start rows per class (and at least smin rows), geometric
                        steps of f0 |T|. Options: cprior, rmin<x>, rmax<x>, start<x>, smin<x>,
                        refit<x>, f0<x>, err (slope of the error rate instead of CE); and for the
                        estimate: rest (predict every row outside the training subset instead of
                        a fixed quarter), reps<n> (subsets per size), cap<n> (T subsampled to n),
                        z<x> (decide from g + z * its standard error: noisy -> small steps)
  switch:d<delta>[...]  geometric steps of f0 |T| (default 0.02) until the slope g drops below
                        delta, then every remaining row at once (the slope is fitted as above)
                        (rel: the slope of log loss, i.e. the relative reduction per doubling)

Options of every schedule:
  class prior           TabPFN(T)'s posteriors are moved from T's class mix f_T to a target mix of
                        the pool, p * target / f_T, before scoring: cprior (the context's given
                        labels), pprior (the pool's given labels), eprior (Saerens EM on the pool's
                        features: the mix that makes the posteriors self-consistent; em<n>
                        iterations, default 100), oprior (the pool's clean labels: an oracle);
                        pa<x>: a partial correction, p * (target / f_T)^x
  vmax, vb<x>           V-anchored scores: a row's score with T is replaced by the larger of it and
                        its score with V alone (vmax), or by (1 - x) score_T + x score_V (vb<x>)
  warm<q>               warm start: before growing, the q n context rows whose given labels are most
                        plausible under a 5-fold cross-fitted self model (TabPFN with V + the other
                        folds of the poisoned context) join T at once
  wb<q>                 the same warm start stratified by given label: each class contributes in
                        proportion to its count of given labels
  qb<f>                 per-class quotas (in proportion to the pool's given-label counts) for every
                        step until f (n - k) rows were accepted, against a runaway of T's class mix
  lo<p>, hi<p>          (fixed) steps of at least / at most p% of n (e.g. fixed:frac0.1:lo1: 1% of n
                        while 10% of |T| is smaller, then geometric)
  rs<j>                 a randomised repeat (seed j): each step (but the last) only a random 80% of
                        the pool may join; for post-hoc ensembles of repeats
  sw<q>                 (fixed) every remaining row at once after q (n - k) rows were accepted
  vacc[:a1<x>:a2<x>]    the step rule chosen from V's estimated clean accuracy: the agreement of
                        TabPFN(V)'s predictions with the context's given labels divided by 1 - R (the
                        share of unflipped labels), at the first scoring; pct1 below a1 (0.55), frac0.1
                        below a2 (0.75), pct10 above (subs=<s1>/<s2>/<s3> to change them). A weak V:
                        many re-scorings; a strong V: few
  fdr[:eps<x>...]       risk control: each step the pool's label log-odds are fitted as a mixture of a
                        clean and a poisoned Gaussian, the poisoned weight fixed at the poisoned rows
                        estimated to be left (k minus the estimated poisoned rows accepted so far);
                        the batch is the largest top-m whose mean posterior poisoned probability is
                        <= eps (0.05), at least floor% of n (0.2); known: the true poisoned count of
                        the pool (an oracle for the weight)
  ctrl[:tau<x>...]      step control: the relative step w (m = w |T|, from f0 = 0.1) is multiplied
                        after every re-scoring by clamp((tau / averted)^gain, 1/2, 2), averted = the
                        share of the batch the previous scores would accept next (at the current w)
                        that the new scores rank beyond mu x its size (tau 0.05, mu 2, gain 0.5,
                        wmin 0.01, wmax 1)

Per run: the test accuracy, balanced accuracy, ROC-AUC (binary: of class 1; multi-class:
one-vs-rest) and log loss of TabPFN with the final T as the context; the precision of the rows left
out; the steps; and per step |T|, m, the poisoned share of the accepted batch and the slope. The
acceptance position of every row is saved (<run>.npz) for post-hoc ensembles of schedules
(--posthoc: mean normalised acceptance position over a set of schedules, the k latest left out).

Draws (eval): V = c random clean rows of every class from the context file, the rest poisoned with
hub-flip-local at R% (bench_small_v.py's draws), the test file as the test set.

    python scripts/schedule_study.py --gpu 1 --jobs results/sched/jobs_p1.jsonl
    python scripts/schedule_study.py --gpu 1 --posthoc results/sched/posthoc.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("results/sched")
CONTEXT = {"gas-drift": "datasets/gas-drift/splits/context_5000.csv",
           "url_unique": "datasets/url_unique/splits/selected/natural/context_5000.csv",
           "heloc": "datasets/heloc/splits/selected/natural/context_5000.csv",
           "credit-approval": "datasets/credit-approval/splits/train.csv",
           "students_dropout": "datasets/students_dropout/splits/context_3000.csv",
           "coil2000_insurance_policies": "datasets/coil2000_insurance_policies/splits/selected/natural/context_5000.csv"}
TEST = {"gas-drift": "datasets/gas-drift/splits/test_attack_1000.csv",
        "url_unique": "datasets/url_unique/splits/test_attack_1000.csv",
        "heloc": "datasets/heloc/splits/test_attack_1000.csv",
        "credit-approval": "datasets/credit-approval/splits/test.csv",
        "students_dropout": "datasets/students_dropout/splits/test_attack_500.csv",
        "coil2000_insurance_policies": "datasets/coil2000_insurance_policies/splits/test_attack_1000.csv"}


# ---------------------------------------------------------------- data and draws

class Data:
    def __init__(self, ds: str, device: str):
        import gate_context as G
        from tabfm_experiments.config import Seeds
        from tabfm_experiments.runtime import build_tabpfn_v2
        self.ds, self.device = ds, device
        self.X, self.y = G.load(Path(CONTEXT[ds]), device)
        self.Xt, self.yt = G.load(Path(TEST[ds]), device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        self.K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(self.y)))
        self.clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=self.K)

    def idx(self, rows):
        return torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)

    def probs(self, X_ctx, y_ctx, Xq, yq=None) -> np.ndarray:
        import gate_context as G
        yq = np.zeros(len(Xq), dtype=np.int64) if yq is None else yq
        return G.evaluate(self.clf, X_ctx, torch.as_tensor(np.asarray(y_ctx), dtype=torch.float32, device=self.device), Xq,
                          torch.as_tensor(np.asarray(yq), dtype=torch.long, device=self.device), None, None,
                          need_grad=False, batch_size=None)[2]


def tag_of(ds, R, c, seed) -> str:
    return f"{ds}_R{R:g}_c{c}_s{seed}"


def get_draw(data: Data, R: float, c: int, seed: int) -> dict:
    """The eval draw (bench_small_v.py's construction), built once and saved."""
    from bench_small_v import draw_per_class
    path = ROOT / "draws" / f"{tag_of(data.ds, R, c, seed)}.npz"
    if not path.exists():
        from tabfm_experiments.config import Seeds
        from tabfm_experiments.multiclass_attacks import poison
        from tabfm_experiments.sampling import k_from_percent
        v_rows = draw_per_class(data.y, c, seed)
        ctx_rows = np.setdiff1d(np.arange(len(data.y)), v_rows)
        rows, y_pois = poison("hub-flip-local", data.X[data.idx(ctx_rows)], data.y[ctx_rows], data.K, R, run_id=0,
                              row_seed=Seeds.row_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, v_rows=v_rows, ctx_rows=ctx_rows, y_given=np.asarray(y_pois).astype(np.int64),
                            pois_rows=np.asarray(rows, dtype=np.int64), k=np.int64(k_from_percent(len(ctx_rows), R)))
        os.replace(tmp, path)
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["n"] = len(d["ctx_rows"])
    d["k"] = int(d["k"])
    d["pois"] = np.zeros(d["n"], dtype=bool)
    d["pois"][d["pois_rows"]] = True
    d["Xc"], d["Xv"], d["yv"] = data.X[data.idx(d["ctx_rows"])], data.X[data.idx(d["v_rows"])], data.y[d["v_rows"]]
    d["y_true"] = data.y[d["ctx_rows"]]  # diagnostics only (oprior, the prior-estimate error in the trace)
    d["tag"] = tag_of(data.ds, R, c, seed)
    return d


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = p.argmax(1)
    out = {"acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred)),
           "ce": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())}
    try:
        out["auc"] = float(roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr"))
    except ValueError:
        out["auc"] = float("nan")
    return out


def test_scores(data: Data, d: dict, keep: np.ndarray, with_v: bool = True) -> dict:
    rows = np.flatnonzero(keep)
    X_ctx = torch.cat([d["Xc"][data.idx(rows)], d["Xv"]]) if with_v else d["Xc"][data.idx(rows)]
    y_ctx = np.concatenate([d["y_given"][rows], d["yv"]]) if with_v else d["y_given"][rows]
    return metrics(data.probs(X_ctx, y_ctx, data.Xt, data.yt), data.yt)


# ---------------------------------------------------------------- schedules

PRIORS = ("cprior", "eprior", "oprior", "pprior")


def _own_option(o: str) -> bool:
    return (o in PRIORS or (o.startswith("sw") and o[2:3].isdigit()) or (o.startswith("em") and o[2:].isdigit())
            or (o.startswith("pa") and o[2:3].isdigit()) or o == "vmax" or (o.startswith("vb") and o[2:3].isdigit())
            or (o.startswith("rs") and o[2:].isdigit()) or (o.startswith("lo") and o[2:3].isdigit())
            or (o.startswith("hi") and o[2:3].isdigit()) or (o.startswith("warm") and o[4:5].isdigit())
            or (o.startswith("wb") and o[2:3].isdigit()) or (o.startswith("qb") and o[2:3].isdigit()))


def parse(spec: str) -> dict:
    head, *opts = spec.split(":")
    c = {"spec": spec, "kind": head, "prior": next((o for o in opts if o in PRIORS), None), "sw": 1.0, "em": 100,
         "pa": 1.0, "vmax": "vmax" in opts, "vb": 0.0, "rs": None, "lo": 0.0, "hi": 100.0, "warm": 0.0,
         "wb": 0.0, "qb": 0.0}
    for o in opts:
        if o.startswith("sw") and o[2:3].isdigit():
            c["sw"] = float(o[2:])
        if o.startswith("em") and o[2:].isdigit():
            c["em"] = int(o[2:])
        if o.startswith("pa") and o[2:3].isdigit():
            c["pa"] = float(o[2:])
        if o.startswith("vb") and o[2:3].isdigit():
            c["vb"] = float(o[2:])
        if o.startswith("rs") and o[2:].isdigit():
            c["rs"] = int(o[2:])
        for key in ("lo", "hi"):
            if o.startswith(key) and o[2:3].isdigit():
                c[key] = float(o[2:])
        if o.startswith("warm") and o[4:5].isdigit():
            c["warm"] = float(o[4:])
        for key in ("wb", "qb"):
            if o.startswith(key) and o[2:3].isdigit():
                c[key] = float(o[2:])
    if head in ("oneshot", "fixed"):
        c["variant"] = "oneshot" if head == "oneshot" else ":".join(["grow"] + [o for o in opts if not _own_option(o)])
        return c
    if head == "vacc":
        c |= {"a1": 0.55, "a2": 0.75, "subs": ("pct1", "frac0.1", "pct10")}
        for o in opts:
            for key in ("a1", "a2"):
                if o.startswith(key) and o[2:3].isdigit():
                    c[key] = float(o[2:])
            if o.startswith("subs="):
                c["subs"] = tuple(o[5:].split("/"))
        return c
    if head == "fdr":
        c |= {"eps": 0.05, "floor": 0.2, "known": "known" in opts}
        for o in opts:
            for key in ("eps", "floor"):
                if o.startswith(key) and o[len(key):len(key) + 1].isdigit():
                    c[key] = float(o[len(key):])
        return c
    if head == "ctrl":
        c |= {"tau": 0.05, "mu": 2.0, "f0": 0.1, "wmin": 0.01, "wmax": 1.0, "gain": 0.5}
        for o in opts:
            for key in ("tau", "mu", "f0", "wmin", "wmax", "gain"):
                if o.startswith(key) and o[len(key):len(key) + 1].isdigit():
                    c[key] = float(o[len(key):])
        return c
    if head not in ("slope", "switch"):
        raise ValueError(spec)
    c |= {"delta": None, "rmin": 0.5, "rmax": 10.0, "start": 3.0, "smin": 20.0, "refit": 1.5,
          "f0": 0.1 if head == "slope" else 0.02, "metric": "err" if "err" in opts else "ce",
          "z": 0.0, "reps": 1.0, "cap": 0.0, "holdout": "rest" if "rest" in opts else "quarter",
          "rel": "rel" in opts}
    for o in opts:
        for key in ("rmin", "rmax", "start", "smin", "refit", "f0", "z", "reps", "cap"):
            if o.startswith(key):
                c[key] = float(o[len(key):])
        if o.startswith("d") and o[1:2].isdigit():
            c["delta"] = float(o[1:])
    if c["delta"] is None:
        raise ValueError(f"{spec}: needs d<delta>")
    return c


def em_prior(p: np.ndarray, f_T: np.ndarray, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Saerens et al. (2002): the class mix of the scored rows under which TabPFN's posteriors (fitted
    with T's class mix f_T) are self-consistent, from the features only (label flips cannot bias it)."""
    pi = f_T.copy()
    for _ in range(iters):
        q = p * (pi / f_T)[None, :]
        q = q / q.sum(1, keepdims=True)
        new = q.mean(0)
        done = np.abs(new - pi).max() < tol
        pi = new
        if done:
            break
    return pi


def self_scores(data: Data, d: dict, folds: int = 5, seed: int = 0) -> np.ndarray:
    """CE of every context row's given label under TabPFN with V and the other folds of the
    (poisoned) context as its context: a model of the mostly clean majority, much stronger than V
    alone when V is tiny."""
    n, yg = d["n"], d["y_given"]
    fold = np.random.default_rng(seed + 777).integers(0, folds, n)
    out = np.zeros(n)
    for f in range(folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        q = data.probs(torch.cat([d["Xc"][data.idx(tr)], d["Xv"]]), np.concatenate([yg[tr], d["yv"]]),
                       d["Xc"][data.idx(te)], yg[te])
        out[te] = -np.log(np.clip(q[np.arange(len(te)), yg[te]], 1e-12, None))
    return out


def quota_pick(s: np.ndarray, labels: np.ndarray, m: int, K: int) -> np.ndarray:
    """Positions of the m rows to accept: per given-label class, its share of m in proportion to the
    class's count among the candidates (largest remainder), the lowest scores within the class."""
    from iterative_filter import quotas
    q = quotas(m, np.bincount(labels, minlength=K))
    pick = [np.flatnonzero(labels == cl)[np.argsort(s[labels == cl], kind="stable")[:q[cl]]] for cl in range(K) if q[cl]]
    return np.concatenate(pick).astype(np.int64)


def poison_posterior(q: np.ndarray, pi_p: float, iters: int = 200) -> np.ndarray:
    """P(poisoned | q) for the pool's label log-odds q under a two-Gaussian mixture whose poisoned
    (lower) component has the known weight pi_p (EM over the means and spreads only)."""
    pi_p = float(np.clip(pi_p, 1e-3, 1 - 1e-3))
    w = np.array([pi_p, 1 - pi_p])
    mu = np.quantile(q, [pi_p / 2, 1 - (1 - pi_p) / 2]).astype(float)
    sd = np.full(2, q.std() / 2 + 1e-3)
    for _ in range(iters):
        logp = -0.5 * ((q[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        r = np.exp(logp - logp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        new_mu = (r * q[:, None]).sum(0) / np.maximum(r.sum(0), 1e-9)
        sd = np.sqrt((r * (q[:, None] - new_mu) ** 2).sum(0) / np.maximum(r.sum(0), 1e-9)) + 1e-3
        done = np.abs(new_mu - mu).max() < 1e-6
        mu = new_mu
        if done:
            break
    if mu[0] > mu[1]:  # the poisoned component is the one with the less plausible labels
        r = r[:, ::-1]
    return r[:, 0]


def cv_slope(data: Data, d: dict, T_rows: np.ndarray, rng: np.random.Generator, metric: str, reps: int = 1,
             cap: int = 0, holdout: str = "quarter", rel: bool = False) -> dict | None:
    """Held-out loss of TabPFN on T against log2(training size): the gain per doubling, g, and its
    standard error (ordinary least squares over every (size, repetition) point).

    holdout "quarter": a fixed random quarter of T is predicted from subsets of the rest (the first
    estimator); "rest": every row of T outside the training subset is predicted (at most 1000).
    cap > 0: T is first subsampled to cap rows (the slope then describes the curve up to cap).
    rel: the slope of log(loss), i.e. the relative loss reduction per doubling (scale-free)."""
    X_T = torch.cat([d["Xc"][data.idx(T_rows)], d["Xv"]])
    y_T = np.concatenate([d["y_given"][T_rows], d["yv"]])
    N, K = len(y_T), data.K
    base = np.arange(N)
    if cap and N > cap:
        base = rng.choice(N, cap, replace=False)
    if holdout == "quarter":
        perm = rng.permutation(base)
        n_h = max(K, int(round(0.25 * len(base))))
        H_fixed, A = perm[:n_h], perm[n_h:]
    else:
        H_fixed, A = None, base
    top = 1.0 if holdout == "quarter" else 0.75
    sizes = sorted({max(2 * K, int(round(len(A) * f))) for f in (top / 8, top / 4, top / 2, top)})
    sizes = [z for z in sizes if z <= len(A) - (0 if holdout == "quarter" else K)]
    if len(sizes) < 3:
        return None
    xs, ys = [], []
    for z in sizes:
        for _ in range(reps):  # a random subset of A with every class of A in it
            must = np.array([rng.choice(A[y_T[A] == cl]) for cl in np.unique(y_T[A])])
            rest = np.setdiff1d(A, must)
            S = np.concatenate([must, rng.choice(rest, max(0, min(z - len(must), len(rest))), replace=False)]).astype(np.int64)
            H = H_fixed if H_fixed is not None else np.setdiff1d(A, S)
            if H_fixed is None and len(H) > 1000:
                H = rng.choice(H, 1000, replace=False)
            q = data.probs(X_T[data.idx(S)], y_T[S], X_T[data.idx(H)], y_T[H])
            loss = (-np.log(np.clip(q[np.arange(len(H)), y_T[H]], 1e-12, None)).mean() if metric == "ce"
                    else float((q.argmax(1) != y_T[H]).mean()))
            xs.append(math.log2(len(S)))
            ys.append(float(loss))
    x, yv = np.array(xs), np.array(ys)
    level = float(yv[x == x.max()].mean())
    if rel:
        yv = np.log(np.maximum(yv, 1e-4))
    b, a0 = np.polyfit(x, yv, 1)
    resid = yv - (a0 + b * x)
    se = float(np.sqrt(resid.var(ddof=2) / max(((x - x.mean()) ** 2).sum(), 1e-12))) if len(x) > 2 else float("inf")
    return {"g": max(0.0, -float(b)), "raw": -float(b), "se": se, "level": level, "sizes": sorted({int(2 ** v) for v in x})}


def run_schedule(data: Data, d: dict, spec: str, seed: int = 0, log=print) -> dict:
    from iterative_filter import parse_variant, step_size
    c = parse(spec)
    n, k, K = d["n"], d["k"], data.K
    nv = len(d["yv"])
    prior_ctx = np.bincount(d["y_given"], minlength=K) / n
    v_fixed = parse_variant(c["variant"]) if c["kind"] in ("oneshot", "fixed") else None
    rng = np.random.default_rng(seed + 4242)
    rs_rng = np.random.default_rng(1000 * seed + (c["rs"] or 0) + 99)
    in_T = np.zeros(n, dtype=bool)
    pos = np.full(n, n, dtype=np.int64)
    accepted, steps, fit, fits, switched = 0, [], None, 0, False
    last_fit_size = 0
    w_ctrl, s_prev = c.get("f0", 0.1), None  # ctrl: the relative step and the last scores (by context row)
    pois_in_T = 0.0  # fdr: the poisoned rows estimated to have joined T so far
    if c["warm"] > 0 or c["wb"] > 0:  # warm start: the self model's most plausible rows join T at once
        m0 = min(math.ceil(max(c["warm"], c["wb"]) * n), n - k)
        s_self = self_scores(data, d, seed=seed)
        # wb: per given-label class in proportion to its count (an unstratified top can be one class only)
        pick = quota_pick(s_self, d["y_given"], m0, K) if c["wb"] > 0 else np.argsort(s_self, kind="stable")[:m0]
        in_T[pick] = True
        pos[pick] = np.arange(m0)
        accepted = m0
        steps.append({"size_T": nv, "m": m0, "batch_pois": float(d["pois"][pick].mean()), "warm": True})
    t0 = time.time()
    while n - accepted > k:
        P = np.flatnonzero(~in_T)
        acc_rows = np.flatnonzero(in_T)
        X_ctx = torch.cat([d["Xc"][data.idx(acc_rows)], d["Xv"]])
        y_ctx = np.concatenate([d["y_given"][acc_rows], d["yv"]])
        p = data.probs(X_ctx, y_ctx, d["Xc"][data.idx(P)], d["y_given"][P])
        info = {}
        if c["kind"] == "vacc" and v_fixed is None:
            # V's clean accuracy, estimated from its agreement with the given labels (R% of them flipped)
            acc_est = float((p.argmax(1) == d["y_given"][P]).mean()) / (1 - k / n)
            sub = c["subs"][0 if acc_est < c["a1"] else 1 if acc_est < c["a2"] else 2]
            v_fixed = parse_variant("grow:" + sub)
            info |= {"acc_est": acc_est, "sub": sub}
        if c["prior"]:  # posteriors moved from T's class frequencies to a target class mix of the pool
            f_T = np.bincount(y_ctx, minlength=K) / len(y_ctx)
            true_mix = np.bincount(d["y_true"][P], minlength=K) / len(P)
            target = {"cprior": lambda: prior_ctx,  # the context's given labels (poisoned)
                      "pprior": lambda: np.bincount(d["y_given"][P], minlength=K) / len(P),  # the pool's given labels
                      "eprior": lambda: em_prior(p, np.maximum(f_T, 1e-12), c["em"]),  # EM on the pool's features
                      "oprior": lambda: true_mix}[c["prior"]]()  # the pool's clean labels (oracle)
            p = p * ((target / np.maximum(f_T, 1e-12)) ** c["pa"])[None, :]
            p = p / p.sum(1, keepdims=True)
            info["prior_err"] = float(np.abs(target - true_mix).sum())
        s = -np.log(np.clip(p[np.arange(len(P)), d["y_given"][P]], 1e-12, None))
        if accepted == 0:  # T = V: these are the V-filter's scores of every context row
            s_V = s.copy()
        elif c["vmax"]:  # a row joins only once both T and V alone find its label plausible
            s = np.maximum(s, s_V[P])
        elif c["vb"]:  # the V-filter's opinion kept in the score: a check on rows vouched for by poison in T
            s = (1 - c["vb"]) * s + c["vb"] * s_V[P]
        size_T = nv + accepted
        left = n - accepted - k
        if c["kind"] in ("oneshot", "fixed", "vacc"):
            m = left if accepted >= c["sw"] * (n - k) else step_size(v_fixed, n, k, size_T, left)
            m = min(max(m, math.ceil(c["lo"] / 100 * n)), math.ceil(c["hi"] / 100 * n))
        elif c["kind"] == "fdr":
            # the largest batch whose estimated poisoned share stays below eps: the label log-odds of the pool
            # as a two-component mixture with the poisoned weight known from the budget (k minus the poisoned
            # rows estimated to have joined T: the recurrence of T's contamination, run on the estimates)
            pk = np.clip(p[np.arange(len(P)), d["y_given"][P]], 1e-6, 1 - 1e-6)
            q = np.log(pk) - np.log1p(-pk)
            k_left = d["pois"][P].sum() if c["known"] else max(k - pois_in_T, 1.0)
            post = poison_posterior(q, k_left / len(P))
            order0 = np.argsort(s, kind="stable")
            fdr = np.cumsum(post[order0]) / np.arange(1, len(P) + 1)
            ok = np.flatnonzero(fdr[:left] <= c["eps"])
            m = max(int(ok[-1]) + 1 if len(ok) else 1, math.ceil(c["floor"] / 100 * n))
            m = min(m, left)
            pois_in_T += float(post[order0[:m]].sum())
            info |= {"fdr": float(fdr[m - 1]), "k_left": float(k_left), "true_k_left": int(d["pois"][P].sum())}
        elif c["kind"] == "ctrl":
            # step control: of the batch the previous scores would accept next, the share that the new
            # scores push beyond mu x the batch size ("averted"); the relative step w is steered so that
            # this share stays near tau (the rejections a skipped re-scoring would have let through)
            if s_prev is not None:
                m_c = max(1, min(math.ceil(w_ctrl * size_T), left))
                stale = np.argsort(s_prev[P], kind="stable")[:m_c]
                fresh_rank = np.empty(len(P), dtype=np.int64)
                fresh_rank[np.argsort(s, kind="stable")] = np.arange(len(P))
                averted = float((fresh_rank[stale] >= c["mu"] * m_c).mean())
                w_ctrl *= min(2.0, max(0.5, (c["tau"] / max(averted, 1e-3)) ** c["gain"]))
                w_ctrl = min(max(w_ctrl, c["wmin"]), c["wmax"])
                info |= {"averted": averted, "w": w_ctrl}
            s_prev = np.full(n, np.inf)
            s_prev[P] = s
            m = math.ceil(w_ctrl * size_T)
        else:
            ready = size_T >= max(c["start"] * K, c["smin"])
            if ready and (fit is None or size_T >= c["refit"] * last_fit_size):
                fit = cv_slope(data, d, acc_rows, rng, c["metric"], reps=int(c["reps"]), cap=int(c["cap"]),
                               holdout=c["holdout"], rel=c["rel"])
                fits += 1
                last_fit_size = size_T
            # the step is decided from an upper confidence bound on the gain: a noisy estimate keeps steps small
            g = None if (fit is None or not ready) else fit["g"] + c["z"] * (fit["se"] if np.isfinite(fit["se"]) else 1e9)
            info |= {"g": g, "g_hat": None if fit is None else fit["g"], "se": None if fit is None else fit["se"],
                     "level": None if fit is None else fit["level"]}
            if g is None:
                m = math.ceil(c["f0"] * size_T)
            elif c["kind"] == "switch":
                if switched or g < c["delta"]:
                    switched = True
                    m = left
                else:
                    m = math.ceil(c["f0"] * size_T)
            else:
                r = min(max(g / c["delta"], c["rmin"]), c["rmax"])
                w = 1 - 2 ** (-1 / r)
                m = math.ceil(w * size_T / (1 - w))
                info["r"] = r
        m = max(1, min(int(m), left))
        if c["rs"] is not None and m < left:  # randomised repeat: each step only a random 80% of the pool is eligible
            s = np.where(rs_rng.random(len(P)) < 0.8, s, np.inf)
        if accepted < c["qb"] * (n - k):  # early growth: per-class quotas in proportion to the pool's given labels
            order = quota_pick(s, d["y_given"][P], m, K)
        else:
            order = np.argsort(s, kind="stable")[:m]
        pick = P[order]
        in_T[pick] = True
        pos[pick] = accepted + np.arange(m)
        accepted += m
        steps.append({"size_T": size_T, "m": m, "batch_pois": float(d["pois"][pick].mean()), **info})
    keep = in_T
    res = {"spec": spec, "steps": len(steps), "fits": fits, "prec": float(d["pois"][~keep].mean()),
           "n_pois_kept": int(d["pois"][keep].sum()), "test": test_scores(data, d, keep), "seconds": round(time.time() - t0, 1),
           "trace": steps}
    return res, pos


def references(data: Data, d: dict) -> dict:
    n = d["n"]
    zv = metrics(data.probs(d["Xv"], d["yv"], data.Xt, data.yt), data.yt)
    return {"v_only": zv, "poisoned_plus_v": test_scores(data, d, np.ones(n, dtype=bool)),
            "oracle_plus_v": test_scores(data, d, ~d["pois"])}


# ---------------------------------------------------------------- job queue

def claim(lock: Path) -> bool:
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                host, pid = lock.read_text().split()
                if host == socket.gethostname():
                    os.kill(int(pid), 0)
                return False
            except ProcessLookupError:
                lock.unlink(missing_ok=True)
            except (ValueError, FileNotFoundError):
                return False
    return False


def safe(spec: str) -> str:
    return spec.replace(":", "_")


def worker(a) -> None:
    device = f"cuda:{a.gpu}" if a.gpu != "cpu" else "cpu"
    jobs = [json.loads(x) for x in a.jobs.read_text().splitlines() if x.strip()]
    data = None
    for j in jobs:
        ds, R, c, s, spec = j["dataset"], float(j["R"]), int(j["c"]), int(j["seed"]), j["spec"]
        out = ROOT / "runs" / tag_of(ds, R, c, s) / f"{safe(spec)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() or not claim(out.with_suffix(".lock")):
            continue
        try:
            if data is None or data.ds != ds:
                data = None
                torch.cuda.empty_cache() if device.startswith("cuda") else None
                data = Data(ds, device)
            d = get_draw(data, R, c, s)
            ref = out.parent / "references.json"
            if not ref.exists() and claim(ref.with_suffix(".lock")):
                ref.write_text(json.dumps(references(data, d)))
                ref.with_suffix(".lock").unlink(missing_ok=True)
            res, pos = run_schedule(data, d, spec, seed=s)
            res["draw"] = {"ds": ds, "R": R, "c": c, "seed": s, "n": d["n"], "k": d["k"], "V": len(d["yv"])}
            np.savez_compressed(out.with_suffix(".npz"), pos=pos, pois=d["pois"])
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res))
            os.replace(tmp, out)
            t = res["test"]
            print(f"{d['tag']} {spec}: acc {100 * t['acc']:.1f} auc {100 * t['auc']:.1f} ce {t['ce']:.3f} prec {res['prec']:.3f} "
                  f"steps {res['steps']} fits {res['fits']} ({res['seconds']:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tag_of(ds, R, c, s)} {spec}: FAILED {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            out.with_suffix(".lock").unlink(missing_ok=True)


def posthoc(a) -> None:
    """Ensembles of schedules: mean normalised acceptance position; the k latest rows left out."""
    device = f"cuda:{a.gpu}"
    spec_sets = json.loads(a.posthoc.read_text())  # {"name": [spec, ...], ...}
    data = None
    for draw in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        out = draw / "posthoc.json"
        done = json.loads(out.read_text()) if out.exists() else {}
        todo = {nm: specs for nm, specs in spec_sets.items()
                if nm not in done and all((draw / f"{safe(sp)}.npz").exists() for sp in specs)}
        if not todo:
            continue
        r0 = next(r for r in (json.loads(f.read_text()) for f in draw.glob("*.json")) if "draw" in r)["draw"]
        if data is None or data.ds != r0["ds"]:
            data = Data(r0["ds"], device)
        d = get_draw(data, r0["R"], r0["c"], r0["seed"])
        for nm, specs in todo.items():
            pos = np.mean([np.load(draw / f"{safe(sp)}.npz")["pos"] / d["n"] for sp in specs], axis=0)
            tie = np.random.default_rng(7).random(d["n"])
            out_rows = np.lexsort((tie, -pos))[:d["k"]]          # the latest mean positions
            keep = np.ones(d["n"], dtype=bool)
            keep[out_rows] = False
            done[nm] = {"test": test_scores(data, d, keep), "prec": float(d["pois"][~keep].mean()), "members": specs}
        out.write_text(json.dumps(done))
        print(draw.name, {nm: round(100 * v["test"]["acc"], 1) for nm, v in done.items()}, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--posthoc", type=Path, help="JSON {ensemble name: [schedule specs]}")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    if a.posthoc:
        posthoc(a)
    else:
        worker(a)


if __name__ == "__main__":
    main()

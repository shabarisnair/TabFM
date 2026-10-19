#!/usr/bin/env python
"""Learn per-row weights w in [0, 1] for the poisoned context rows from a clean validation set.

The weights act through every channel by which a context row reaches the others
(tabfm_experiments/row_weights.py: the attention bias log w in all 12 blocks, the weighted label
mean, the weighted z-norms, the discrete fits ignoring w = 0), so w = 0 is exactly deleting the
row. The loss is the cross-entropy of the fitted validation rows with the weighted context; the
iteration is chosen on the held-out validation rows. The draw is the soft gate's
(results/gate_valsplit: V clean rows drawn from the context file, 20% of them held out,
hub-flip-local at R%, run 0), checked against the saved run when one exists.

Parameterisations (--param), each with a box it is projected back into after every update:
  log      log w = b in [--floor, 0]                     (the older --gate attn's)
  sigmoid  w = sigmoid(t), log w = logsigmoid(t), t in [--floor, --t-max], starting at --t0
  w        w itself in [0, 1]; a weight projected to exactly 0 is deleted (and stays: its
           gradient is 0)
  scaled   log w = --scale * u, u in [-1, 0]             (an lr-free stretch of log space)
  topk     hard: a free score per row, the forward pass deletes the k = R% lowest exactly
           (log w = -inf) and keeps the rest (log w = 0); the gradient is taken at a relaxed
           point, log w = log(--probe) on the deleted rows and log(1 - --probe) on the kept
           ones (or --probe-logw L < 0: L on the deleted rows, 0 on the kept), and moves the scores
           like log-weights
Gradient (--grad): through every channel (all), the attention and the label mean (attn_label:
the statistics weighted but detached) or the attention alone (attn); --center class removes
its mean within each given label (no class-level push), global its overall mean.
--prior L adds L * (raw - raw at the start) to the loss gradient taken in RMS units (a pull
back to the start, e.g. the V-filter ranking); --vsub f fits each update on a random f of the
fitted validation rows (class-stratified).
--pseudo f adds, per update, the CE of --pseudo-batch rows drawn from the f most plausible
context rows by the V-filter, held out of the context for that forward (weight 0 for every query,
like a cross-validation fold), weighted --pseudo-weight.
Updates (--opt): pgd-linf (lr * g / max|g|), pgd-rms (lr * g / rms(g)), sign (lr * sign g),
sgd (lr * g), adam. Starts (--init): zero (every w = 1; topk: random scores, --init-std) or
vfilter: s = log p(given label | the fitted validation rows as the context), b = clip(--init-temp *
(s - max s), floor, 0) (sigmoid: t = b; w: w = exp(b); topk: the scores are s).

Every --eval-every updates, on the held-out validation rows: CE and accuracy with the soft
weights, and with the k lowest-ranked rows deleted (hard); the precision of those k rows and the
AUROC of the ranking against the poisoned rows are logged (never used); the test set is scored
for every deployment (soft, soft + V: the clean rows added at weight 1, hard, hard + V; never
used to choose). The iteration is chosen per deployment kind (soft / hard) by --select on the
held-out rows: ce (lowest log loss, default), acc or auc (highest; ties: log loss).

    python scripts/attn_learn.py --gpu 1 --dataset heloc --R 20 --param log --opt pgd-linf --lr 5 --floor -1000
    python scripts/attn_learn.py --gpu 1 --jobs jobs.jsonl --out results/attn_learn/screen   # a shared job list
"""
from __future__ import annotations

import argparse
import hashlib
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
import torch.nn.functional as F  # noqa: E402

CONTEXT = {"heloc": "datasets/heloc/splits/selected/natural/context_5000.csv",
           "students_dropout": "datasets/students_dropout/splits/context_3000.csv",
           "coil2000_insurance_policies": "datasets/coil2000_insurance_policies/splits/selected/natural/context_5000.csv",
           "gas-drift": "datasets/gas-drift/splits/context_5000.csv",
           "url_unique": "datasets/url_unique/splits/selected/natural/context_5000.csv",
           "ecommerce_shipping": "datasets/ecommerce_shipping/splits/selected/natural/context_5000.csv",
           "is_this_a_good_customer": "datasets/is_this_a_good_customer/splits/train.csv"}
TEST = {"heloc": "datasets/heloc/splits/test_attack_1000.csv",
        "students_dropout": "datasets/students_dropout/splits/test_attack_500.csv",
        "coil2000_insurance_policies": "datasets/coil2000_insurance_policies/splits/test_attack_1000.csv",
        "gas-drift": "datasets/gas-drift/splits/test_attack_1000.csv",
        "url_unique": "datasets/url_unique/splits/test_attack_1000.csv",
        "ecommerce_shipping": "datasets/ecommerce_shipping/splits/test_attack_1000.csv",
        "is_this_a_good_customer": "datasets/is_this_a_good_customer/splits/test.csv"}
DEFAULTS = dict(param="log", opt="pgd-linf", lr=1.0, floor=-1000.0, t_max=20.0, t0=4.0, scale=1000.0, probe=0.05,
                init="zero", init_temp=1.0, init_std=1.0, steps=200, eval_every=10, holdout=0.2, V=500, seed=0, probe_logw=0.0, grad="all", center="none", prior=0.0, vsub=1.0, pseudo=0.0, pseudo_weight=1.0, pseudo_batch=500, test_every_eval=1, select="ce",
                adam_beta1=0.9, adam_beta2=0.999, tag="")


def cfg_name(c: dict) -> str:
    parts = [c["param"], c["opt"], f"lr{c['lr']:g}"]
    if c["param"] in ("log", "sigmoid"):
        parts.append(f"floor{c['floor']:g}")
    if c["param"] == "scaled":
        parts.append(f"scale{c['scale']:g}")
    if c["param"] == "topk":
        parts.append(f"probelogw{c['probe_logw']:g}" if c["probe_logw"] < 0 else f"probe{c['probe']:g}")
    parts.append(f"init-{c['init']}" + (f"{c['init_temp']:g}" if c["init"] == "vfilter" else ""))
    if c["grad"] != "all":
        parts.append(f"grad-{c['grad']}")
    if c["center"] != "none":
        parts.append(f"center-{c['center']}")
    if c["prior"] > 0:
        parts.append(f"prior{c['prior']:g}")
    if c["select"] != "ce":
        parts.append(f"sel-{c['select']}")
    if c["vsub"] < 1:
        parts.append(f"vsub{c['vsub']:g}")
    if c["pseudo"] > 0:
        parts.append(f"pseudo{c['pseudo']:g}x{c['pseudo_weight']:g}b{c['pseudo_batch']}")
    parts.append(f"it{c['steps']}")
    return "_".join(parts) + (f"_{c['tag']}" if c["tag"] else "")


# ---------------------------------------------------------------- the draw

def make_draw(ds: str, R: float, V: int, seed: int, holdout: float, device: str) -> dict:
    import gate_context as G
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.sampling import k_from_percent
    X, y = G.load(Path(CONTEXT[ds]), device)
    Xt, yt = G.load(Path(TEST[ds]), device)
    info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
    K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y)))
    val_rows, _ = G.draw_validation_form(X, y, V, seed, "stratified")
    ctx_rows = np.setdiff1d(np.arange(len(y)), val_rows)
    yv_all = y[val_rows]
    fit_pos, hold_pos = G.split_holdout(yv_all, holdout, seed)
    idx = lambda r: torch.as_tensor(np.asarray(r), device=device)  # noqa: E731
    Xc = X[idx(ctx_rows)]
    rows, y_pois = poison("hub-flip-local", Xc, y[ctx_rows], K, R, run_id=0, row_seed=Seeds.row_seed)
    y_pois = np.asarray(y_pois).astype(int)
    n = len(ctx_rows)
    pois = np.zeros(n, dtype=bool)
    pois[rows] = True
    saved = sorted(Path("results/gate_valsplit").glob(f"{ds}/*_r{int(R):03d}/gate-valsplit_{ds}_*_R{int(R)}_V{V}s{seed}_H{round(100 * holdout)}_*"))
    if saved:  # the soft gate's draw
        assert np.array_equal(np.load(saved[0] / "split.npz")["context_rows"], ctx_rows), saved[0]
        assert np.array_equal(np.load(saved[0] / "poison.npz")["y_poisoned"], y_pois), saved[0]
    Xv_all = X[idx(val_rows)]
    return dict(ds=ds, R=R, V=V, seed=seed, K=K, n=n, k=k_from_percent(n, R), Xc=Xc, y_clean=y[ctx_rows], yp=y_pois,
                pois=pois, Xf=Xv_all[idx(fit_pos)], yf=yv_all[fit_pos], Xh=Xv_all[idx(hold_pos)], yh=yv_all[hold_pos],
                Xv=Xv_all, yv=yv_all, Xt=Xt, yt=yt, checked_against=str(saved[0]) if saved else None)


# ---------------------------------------------------------------- parameterisations

class Param:
    """raw: the optimised tensor; log_w(): differentiable in raw; key(): the ranking (lowest
    first = most suspicious); project(): back into the box."""

    def __init__(self, kind: str, raw0: torch.Tensor, c: dict, k: int):
        self.kind, self.c, self.k = kind, c, k
        self.raw = raw0.clone().float().requires_grad_(True)
        self.lo, self.hi = {"log": (c["floor"], 0.0), "sigmoid": (c["floor"], c["t_max"]), "w": (0.0, 1.0),
                            "scaled": (-1.0, 0.0), "topk": (-math.inf, math.inf)}[kind]
        self.tie = torch.as_tensor(np.random.default_rng(12345).random(len(raw0)), device=raw0.device)

    def log_w(self) -> torch.Tensor:
        r = self.raw
        if self.kind == "log":
            return r
        if self.kind == "sigmoid":
            return F.logsigmoid(r)
        if self.kind == "w":
            return torch.where(r > 0, torch.log(r.clamp_min(1e-45)), torch.full_like(r, -math.inf))
        if self.kind == "scaled":
            return self.c["scale"] * r
        return self.hard_log_w()

    def deleted(self) -> torch.Tensor:
        """The k lowest-ranked rows (ties broken at random, fixed)."""
        order = np.lexsort((self.tie.cpu().numpy(), self.key().cpu().numpy()))
        m = torch.zeros(len(self.raw), dtype=torch.bool, device=self.raw.device)
        m[torch.as_tensor(order[:self.k], device=self.raw.device)] = True
        return m

    def hard_log_w(self) -> torch.Tensor:
        return torch.where(self.deleted(), torch.full_like(self.raw, -math.inf), torch.zeros_like(self.raw)).detach()

    def probe_log_w(self) -> torch.Tensor:
        if self.c["probe_logw"] < 0:  # the deleted rows at this log-weight, the kept ones at 0
            return torch.where(self.deleted(), torch.full_like(self.raw, self.c["probe_logw"]),
                               torch.zeros_like(self.raw)).detach()
        p = self.c["probe"]
        return torch.where(self.deleted(), torch.full_like(self.raw, math.log(p)),
                           torch.full_like(self.raw, math.log1p(-p))).detach()

    def key(self) -> torch.Tensor:
        return self.raw.detach()

    def project(self) -> None:
        with torch.no_grad():
            self.raw.clamp_(self.lo, self.hi)


def init_raw(c: dict, d: dict, clf, device) -> torch.Tensor:
    from tabfm_experiments.row_weights import evaluate_weighted
    n = d["n"]
    if c["init"] == "zero":
        if c["param"] == "topk":
            return torch.as_tensor(np.random.default_rng(c["seed"]).normal(0, c["init_std"], n), dtype=torch.float32, device=device)
        return torch.full((n,), {"log": 0.0, "sigmoid": c["t0"], "w": 1.0, "scaled": 0.0}[c["param"]], device=device)
    # vfilter: log p(given label | the fitted validation rows as the context)
    p = evaluate_weighted(clf, d["Xf"], d["yf"], d["Xc"], d["yp"], torch.zeros(len(d["yf"]), device=device),
                          need_grad=False)[2]
    s = torch.as_tensor(np.log(np.clip(p[np.arange(n), d["yp"]], 1e-12, None)), dtype=torch.float32, device=device)
    if c["param"] == "topk":
        return s
    b = (c["init_temp"] * (s - s.max())).clamp(min=c["floor"] if c["param"] in ("log", "sigmoid") else -1000.0, max=0.0)
    if c["param"] == "w":
        return b.exp()
    if c["param"] == "scaled":
        return (b / c["scale"]).clamp(-1.0, 0.0)
    return b


# ---------------------------------------------------------------- training

class Stepper:
    def __init__(self, p: Param, c: dict):
        self.p, self.c = p, c
        self.adam = torch.optim.Adam([p.raw], lr=c["lr"], betas=(c["adam_beta1"], c["adam_beta2"])) if c["opt"] == "adam" else None

    def step(self, g: torch.Tensor) -> None:
        c = self.c
        with torch.no_grad():
            if self.adam is not None:
                self.p.raw.grad = g.clone()
                self.adam.step()
            else:
                if c["opt"] == "pgd-linf":
                    d = g / g.abs().max().clamp_min(1e-30)
                elif c["opt"] == "pgd-rms":
                    d = g / g.pow(2).mean().sqrt().clamp_min(1e-30)
                elif c["opt"] == "sign":
                    d = g.sign()
                else:
                    d = g
                self.p.raw -= c["lr"] * d
        self.p.project()


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    y = np.asarray(y)
    pred = p.argmax(1)
    out = {"acc": float((pred == y).mean()), "ce": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
           "bacc": float(balanced_accuracy_score(y, pred))}
    try:
        out["auc"] = float(roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr"))
    except ValueError:
        out["auc"] = float("nan")
    return out


def run(c: dict, d: dict, clf, device, log=print) -> dict:
    from sklearn.metrics import roc_auc_score
    from tabfm_experiments.row_weights import evaluate_weighted
    torch.manual_seed(c["seed"])
    raw0 = init_raw(c, d, clf, device)
    p = Param(c["param"], raw0, c, d["k"])
    p.project()
    raw_start = p.raw.detach().clone()
    pseudo_pool = None
    if c["pseudo"] > 0:  # the context rows the V-filter (fitted rows as the context) finds most plausible
        pv = evaluate_weighted(clf, d["Xf"], d["yf"], d["Xc"], d["yp"], torch.zeros(len(d["yf"]), device=device),
                               need_grad=False)[2]
        s_v = np.log(np.clip(pv[np.arange(d["n"]), d["yp"]], 1e-12, None))
        pseudo_pool = np.argsort(-s_v, kind="stable")[:int(round(c["pseudo"] * d["n"]))]
        log(f"    pseudo-queries: {len(pseudo_pool)} rows, {100 * d['pois'][pseudo_pool].mean():.1f}% of them poisoned")
    rng_v = np.random.default_rng(c["seed"] + 777)
    st = Stepper(p, c)
    zeros_v = torch.zeros(len(d["yv"]), device=device)
    Xcv = torch.cat([d["Xc"], d["Xv"]])
    ycv = np.concatenate([d["yp"], d["yv"]])

    def score(lw, Xq, yq):
        return metrics(evaluate_weighted(clf, d["Xc"], d["yp"], Xq, np.asarray(yq), lw, need_grad=False)[2], yq)

    evals, snaps, t0, snap_keys = [], {}, time.time(), []
    for it in range(c["steps"] + 1):
        if it % c["eval_every"] == 0 or it == c["steps"]:
            lw_soft = (p.hard_log_w() if c["param"] == "topk" else p.log_w().detach())
            lw_hard = p.hard_log_w()
            soft = score(lw_soft, d["Xh"], d["yh"])
            hard = soft if c["param"] == "topk" else score(lw_hard, d["Xh"], d["yh"])
            key = p.key().cpu().numpy()
            dele = p.deleted().cpu().numpy()
            e = {"it": it, "prec": float(d["pois"][dele].mean()),
                 "auroc": float(roc_auc_score(d["pois"], -key)) if np.ptp(key) > 0 else 0.5,
                 "w_mean": float(lw_soft.exp().mean()), "logw_min": float(lw_soft.min()),
                 "logw_med_pois": float(lw_soft[torch.as_tensor(d["pois"], device=device)].median()),
                 "logw_med_clean": float(lw_soft[torch.as_tensor(~d["pois"], device=device)].median()),
                 "sec": round(time.time() - t0, 1)}
            e |= {f"hold_soft_{m}": v for m, v in soft.items()} | {f"hold_hard_{m}": v for m, v in hard.items()}
            if c["test_every_eval"]:  # every deployment on the test set (never used to choose)
                for name, lw_dep in (("soft", lw_soft), ("hard", lw_hard)):
                    t1 = metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xt"], d["yt"], lw_dep, need_grad=False)[2], d["yt"])
                    t2 = metrics(evaluate_weighted(clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw_dep, zeros_v]),
                                                   need_grad=False)[2], d["yt"])
                    e |= {f"test_{name}_{m}": v for m, v in t1.items()} | {f"test_{name}_v_{m}": v for m, v in t2.items()}
            evals.append(e)
            snaps[it] = (lw_soft.detach().clone(), lw_hard.clone())
            snap_keys.append(key.astype(np.float32))
            if it % (5 * c["eval_every"]) == 0 or it == c["steps"]:
                log(f"    it {it:4d}  hold soft ce {e['hold_soft_ce']:.3f} auc {e['hold_soft_auc']:.3f} | hard ce "
                    f"{e['hold_hard_ce']:.3f} auc {e['hold_hard_auc']:.3f} | prec {e['prec']:.3f}  auroc {e['auroc']:.3f}  "
                    f"log w med pois {e['logw_med_pois']:.1f} clean {e['logw_med_clean']:.1f}  ({e['sec']:.0f} s)")
        if it == c["steps"]:
            break
        lw = p.probe_log_w() if c["param"] == "topk" else p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        Xf, yf = d["Xf"], d["yf"]
        if c["vsub"] < 1:  # a random part of the fitted validation rows this step
            import gate_context as G
            sub = G.draw_validation(yf, max(d["K"], int(round(c["vsub"] * len(yf)))), int(rng_v.integers(1 << 30)))
            Xf, yf = Xf[torch.as_tensor(sub, device=device)], yf[sub]
        loss, g, _ = evaluate_weighted(clf, d["Xc"], d["yp"], Xf, yf, lw_d, need_grad=True,
                                       grad_channels=c["grad"])
        if pseudo_pool is not None:  # CE of a random batch of the trusted rows, held out of the context (a fold)
            qb = np.sort(rng_v.choice(pseudo_pool, min(c["pseudo_batch"], len(pseudo_pool)), replace=False))
            qbt = torch.as_tensor(qb, device=device)
            lw_p = lw_d.detach().clone()
            lw_p[qbt] = -math.inf
            lw_p.requires_grad_(True)
            _, g_p, _ = evaluate_weighted(clf, d["Xc"], d["yp"], d["Xc"][qbt], d["yp"][qb], lw_p, need_grad=True,
                                          grad_channels=c["grad"])
            g_p[qbt] = 0.0
            g = g + c["pseudo_weight"] * g_p
        if c["center"] == "class":  # no class-level push: centre the gradient within each given label
            for cl in np.unique(d["yp"]):
                m = torch.as_tensor(d["yp"] == cl, device=g.device)
                g[m] -= g[m].mean()
        elif c["center"] == "global":
            g = g - g.mean()
        if c["param"] == "topk":
            g_raw = g
        else:
            p.raw.grad = None
            lw.backward(g)
            g_raw = p.raw.grad.detach().clone()
            g_raw = torch.nan_to_num(g_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if c["prior"] > 0:  # the loss gradient in RMS units plus a pull back to the start
            g_raw = g_raw / g_raw.pow(2).mean().sqrt().clamp_min(1e-30) + c["prior"] * (p.raw.detach() - raw_start)
        st.step(g_raw)

    def pick(by: str) -> int:  # --select: held-out log loss (lowest), accuracy or ROC-AUC (highest; ties: log loss)
        crit = c["select"]
        if crit == "ce":
            return min(evals, key=lambda e: (e[f"hold_{by}_ce"], e["it"]))["it"]
        return max(evals, key=lambda e: (np.nan_to_num(e[f"hold_{by}_{crit}"], nan=-1.0), -e[f"hold_{by}_ce"], -e["it"]))["it"]

    res = {"cfg": c, "draw": {k: d[k] for k in ("ds", "R", "V", "seed", "K", "n", "k", "checked_against")}, "evals": evals}
    for by in ("soft", "hard"):
        it = pick(by)
        lw_soft, lw_hard = snaps[it]
        e = next(x for x in evals if x["it"] == it)
        lw_use = lw_hard if by == "hard" else lw_soft
        if c["test_every_eval"]:  # already scored at that evaluation
            test = {m: e[f"test_{by}_{m}"] for m in ("acc", "ce", "bacc", "auc")}
            test_v = {m: e[f"test_{by}_v_{m}"] for m in ("acc", "ce", "bacc", "auc")}
        else:
            test = metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xt"], d["yt"], lw_use, need_grad=False)[2], d["yt"])
            test_v = metrics(evaluate_weighted(clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw_use, zeros_v]),
                                               need_grad=False)[2], d["yt"])
        res[f"sel_{by}"] = {"it": it, "prec": e["prec"], "auroc": e["auroc"], "test": test, "test_plus_v": test_v}
    res["seconds"] = round(time.time() - t0, 1)
    res["snap_keys"] = np.stack(snap_keys)  # the ranking at every evaluation (saved next to the JSON)
    return res


def references(d: dict, clf, device) -> dict:
    """Test scores of the comparison contexts for one draw (no training)."""
    from tabfm_experiments.row_weights import evaluate_weighted
    n, k = d["n"], d["k"]
    zeros = torch.zeros(n, device=device)
    Xcv = torch.cat([d["Xc"], d["Xv"]])
    ycv = np.concatenate([d["yp"], d["yv"]])
    zv = torch.zeros(len(d["yv"]), device=device)

    def test(Xc, yc, lw):
        return metrics(evaluate_weighted(clf, Xc, yc, d["Xt"], d["yt"], lw, need_grad=False)[2], d["yt"])

    def dele(mask):
        return torch.where(torch.as_tensor(mask, device=device), torch.full((n,), -math.inf, device=device), zeros)

    p = evaluate_weighted(clf, d["Xv"], d["yv"], d["Xc"], d["yp"], zv, need_grad=False)[2]
    s = np.log(np.clip(p[np.arange(n), d["yp"]], 1e-12, None))
    vf = np.zeros(n, dtype=bool)
    vf[np.argsort(s, kind="stable")[:k]] = True
    return {"poisoned": test(d["Xc"], d["yp"], zeros), "poisoned_plus_v": test(Xcv, ycv, torch.cat([zeros, zv])),
            "v_only": test(d["Xv"], d["yv"], zv), "vfilter": test(d["Xc"], d["yp"], dele(vf)),
            "vfilter_plus_v": test(Xcv, ycv, torch.cat([dele(vf), zv])), "vfilter_prec": float(d["pois"][vf].mean()),
            "oracle": test(d["Xc"], d["yp"], dele(d["pois"])), "oracle_plus_v": test(Xcv, ycv, torch.cat([dele(d["pois"]), zv]))}


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--jobs", type=Path, default=None, help="JSON lines, one config per line (shared by workers)")
    ap.add_argument("--out", type=Path, default=Path("results/attn_learn/manual"))
    ap.add_argument("--dataset", default="heloc")
    ap.add_argument("--R", type=float, default=20.0)
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val) if not isinstance(val, bool) else int, default=val)
    a = ap.parse_args(argv)
    device = f"cuda:{a.gpu}" if a.gpu != "cpu" else "cpu"
    torch.set_num_threads(8)
    if a.jobs is not None:
        jobs = [json.loads(line) for line in a.jobs.read_text().splitlines() if line.strip()]
    else:
        jobs = [{"dataset": a.dataset, "R": a.R} | {key: getattr(a, key) for key in DEFAULTS}]
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2
    clfs, draws = {}, {}
    for job in jobs:
        c = DEFAULTS | job
        ds, R = c["dataset"], float(c["R"])
        draw_tag = f"{ds}_R{R:g}_V{c['V']}s{c['seed']}"
        out = a.out / draw_tag / f"{cfg_name(c)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() or not claim(out.with_suffix(".lock")):
            continue
        t0 = time.time()
        if (ds, R, c["V"], c["seed"]) not in draws:
            draws.clear()  # one draw in memory at a time
            draws[(ds, R, c["V"], c["seed"])] = make_draw(ds, R, c["V"], c["seed"], c["holdout"], device)
        d = draws[(ds, R, c["V"], c["seed"])]
        if ds not in clfs:
            clfs.clear()
            clfs[ds] = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=d["K"])
        clf = clfs[ds]
        ref_path = a.out / draw_tag / "references.json"
        if not ref_path.exists() and claim(ref_path.with_suffix(".lock")):
            ref_path.write_text(json.dumps(references(d, clf, device), indent=1))
            ref_path.with_suffix(".lock").unlink(missing_ok=True)
        print(f"{draw_tag} {cfg_name(c)}: start (n {d['n']}, k {d['k']})", flush=True)
        try:
            res = run(c, d, clf, device)
        except Exception as e:  # noqa: BLE001 - one failing config must not stop the worker
            print(f"{draw_tag} {cfg_name(c)}: FAILED {type(e).__name__}: {e}", flush=True)
            out.with_suffix(".lock").unlink(missing_ok=True)
            raise
        np.savez_compressed(out.with_suffix(".npz"), keys=res.pop("snap_keys"), pois=d["pois"],
                            its=np.array([e["it"] for e in res["evals"]]))
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        os.replace(tmp, out)
        out.with_suffix(".lock").unlink(missing_ok=True)
        s, h = res["sel_soft"], res["sel_hard"]
        print(f"{draw_tag} {cfg_name(c)}: soft it {s['it']} test {s['test']['acc']:.3f} (+V {s['test_plus_v']['acc']:.3f}) | "
              f"hard it {h['it']} test {h['test']['acc']:.3f} (+V {h['test_plus_v']['acc']:.3f}) prec {h['prec']:.3f} | "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

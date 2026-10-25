#!/usr/bin/env python
"""Why do the learned and bilevel filters (Meta-Sift, TabPFN-sift, M1, the soft weights) fail at tiny V while greedy
growth works? Diagnostics on the tiny-V benchmark's instances (bench_tinyv.py draws: R = 30%, hub-flip-local) and on
control draws with the same V and context rows but other poisonings.

Subcommands (results/why_tinyv/<attack>/<instance>/<name>.json):
  draws      control poisonings of the benchmark's draws, same V and context rows, exactly k flipped rows:
               rand-flip        k uniformly random rows, each to a uniformly random other class (Meta-Sift's
                                "random label flipping")
               hub-flip-random  the hub-flip-local rows (the 30% densest), each to a uniformly random other class
             -> results/why_tinyv/roots/<attack>/draws (a BENCH_TINYV_ROOT for bench_tinyv.py)
  judge      E1: how well each judge's plausibility p(given label) ranks the poison (AUROC, precision of its k least
             plausible rows): V alone (the V-filter); the whole poisoned set + V, leave-one-out (what the sift
             objectives and a model trained on the poisoned data see); the clean rows + V (an oracle judge); greedy's
             trusted set at 25/50/100% of its growth; the 10 nearest neighbours' label agreement; and the per-sample
             loss of Meta-Sift's MLP after each epoch of plain training on the poisoned rows + V. Plus the poison's
             local structure: the share of a row's 10 nearest neighbours that are poisoned, and that carry its label
  grad       E2: the first-order signal, d loss / d log w per poisoned row (context: the poisoned rows), at w = 1 and
             at the V-filter split (M1's relaxed point): of V's loss (M1's objective) and of the test loss (an oracle
             objective) as poison rankers (AUROC; positive gradient = suspicious), how the two agree, and how V's
             gradient is spread (mass in the top 1% of rows; correlation with the distance to the nearest V row)
  landscape  E3: on candidate splits of known precision (random poisoned rows plus random, the hardest or the easiest
             clean rows by the V-filter) and on every method's split: V's loss with the kept rows alone as the
             context (M1's hard objective, V-selection's criterion), the sift objective at the hard split (the
             removed rows' loss with the kept rows + V as the context), and the test metrics of kept + V
  removed    E4: which clean rows each method removes: how hard they are (their p(true label) under the clean rows + V,
             the oracle judge) and how the removal shifts the kept rows' class mix
  influence  E7: real deletions: the change in test log loss from deleting one poisoned or clean row, a poisoned row's
             cluster (the poisoned rows among its 10 nearest neighbours) or as many scattered poisoned rows
  trace      E6: greedy's acceptance order (runs/*/greedy_*.npz): the poisoned share of each tenth of the accepted rows
  report     the tables -> results/tables/why_tinyv/*.md

    python scripts/why_tinyv.py draws --device cuda:0
    python scripts/why_tinyv.py judge --device cuda:0 [--attacks hub-flip-local rand-flip] [--datasets ...] [--cs 5]
    python scripts/why_tinyv.py report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bench_tinyv as B  # noqa: E402

OUT = Path("results/why_tinyv")
ATTACKS = ("hub-flip-local", "rand-flip", "hub-flip-random")
BENCH_ROOT = B.ROOT  # the benchmark's own root: hub-flip-local


def root_of(attack: str) -> Path:
    return BENCH_ROOT if attack == "hub-flip-local" else OUT / "roots" / attack


def auroc(pois: np.ndarray, score: np.ndarray) -> float:
    """AUROC of `score` (higher = more suspicious) against the poisoned rows."""
    from sklearn.metrics import roc_auc_score
    s = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30)
    return float(roc_auc_score(pois, s))


def prec_at_k(pois: np.ndarray, score: np.ndarray, k: int) -> float:
    """The poisoned share of the k most suspicious rows (stable ties)."""
    return float(pois[np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")[:k]].mean())


def p_given(clf, X_ctx, y_ctx, Xq, yq, self_rows=None, chunk: int = 1024) -> np.ndarray:
    """p(given label) of every query row under TabPFN with this context (a query never attends to its own copy in the
    context: self_rows[i] = that copy's index, -1 for none)."""
    from tabfm_experiments.row_weights import evaluate_weighted
    lw = torch.zeros(len(y_ctx), device=X_ctx.device)
    p = evaluate_weighted(clf, X_ctx, np.asarray(y_ctx), Xq, np.asarray(yq), lw, need_grad=False, batch_size=chunk,
                          query_ctx_rows=None if self_rows is None else np.asarray(self_rows))[2]
    return p[np.arange(len(yq)), np.asarray(yq)]


def zscore(d: dict) -> torch.Tensor:
    A = torch.cat([d["Xc"], d["Xv"]]).double()
    mu, sd = A.mean(0), A.std(0).clamp_min(1e-12)
    return (A - mu) / sd


def knn(Z: torch.Tensor, n: int, kk: int = 10, chunk: int = 1024) -> np.ndarray:
    """The kk nearest rows of each of the first n rows, among the first n (itself excluded)."""
    out = np.empty((n, kk), dtype=np.int64)
    for st in range(0, n, chunk):
        D = torch.cdist(Z[st:min(st + chunk, n)], Z[:n])
        D[torch.arange(D.shape[0]), torch.arange(st, st + D.shape[0])] = float("inf")
        out[st:st + D.shape[0]] = D.topk(kk, largest=False).indices.cpu().numpy()
    return out


def instances(a):
    for ds in a.datasets:
        for c in a.cs:
            yield ds, c, B.tag_of(ds, 30.0, c, 0)


def save(attack: str, tag: str, name: str, obj) -> None:
    p = OUT / attack / tag / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def done(attack: str, tag: str, name: str) -> bool:
    return (OUT / attack / tag / f"{name}.json").exists()


class Cache:
    """One dataset's BenchData (and TabPFN) at a time."""

    def __init__(self, device: str):
        self.device, self.bd = device, None

    def get(self, ds: str):
        if self.bd is None or self.bd.ds != ds:
            self.bd = None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.bd = B.BenchData(ds, self.device)
        return self.bd


_CACHES: dict = {}


def cache_for(device: str) -> Cache:
    """One Cache per device for the whole process (the stages run one instance at a time: per_instance)."""
    return _CACHES.setdefault(device, Cache(device))


def per_instance(fn, a) -> None:
    """Run a stage one instance at a time, so that a failure (e.g. out of GPU memory) skips only that instance."""
    import copy
    import traceback
    for attack in a.attacks:
        for ds in a.datasets:
            for c in a.cs:
                b = copy.copy(a)
                b.attacks, b.datasets, b.cs = [attack], [ds], [c]
                try:
                    fn(b)
                except Exception as e:  # noqa: BLE001
                    print(f"{fn.__name__} {attack} {ds} c{c}: FAILED {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    if a.device.startswith("cuda"):
                        torch.cuda.empty_cache()


def load(cache: Cache, attack: str, ds: str, tag: str) -> tuple:
    bd = cache.get(ds)
    B.ROOT = root_of(attack)
    try:
        d = B.load_draw(bd, tag, 0)
    finally:
        B.ROOT = BENCH_ROOT
    return bd, d


# ---------------------------------------------------------------- draws

def stage_draws(a) -> None:
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison, random_other_class
    cache = cache_for(a.device)
    for attack in ("rand-flip", "hub-flip-random"):
        (root_of(attack) / "draws").mkdir(parents=True, exist_ok=True)
        for ds, c, tag in instances(a):
            path = root_of(attack) / "draws" / f"{tag}.npz"
            if path.exists():
                continue
            bd = cache.get(ds)
            z = np.load(BENCH_ROOT / "draws" / f"{tag}.npz")
            ctx, k = z["ctx_rows"], int(z["k"])
            y_clean = bd.y[ctx]
            if attack == "rand-flip":
                rng = np.random.default_rng(12345 + int(c))
                rows = np.sort(rng.choice(len(ctx), k, replace=False))
                y_pois = y_clean.copy()
                y_pois[rows] = random_other_class(y_clean[rows], bd.K, rng)
            else:
                rows, y_pois = poison("hub-flip-random", bd.X[bd.idx(ctx)], y_clean, bd.K, 30.0, run_id=0,
                                      row_seed=Seeds.row_seed)
                rows = np.asarray(rows, dtype=np.int64)
            hub_rows = np.sort(z["pois_rows"])
            assert len(rows) == k and np.all(np.asarray(y_pois)[rows] != y_clean[rows])
            np.savez_compressed(path, v_rows=z["v_rows"], ctx_rows=ctx, y_given=np.asarray(y_pois).astype(np.int64),
                                pois_rows=rows, k=np.int64(k))
            print(f"{attack} {tag}: {k} rows flipped; overlap with the hub rows {np.isin(rows, hub_rows).mean():.2f}",
                  flush=True)


# ---------------------------------------------------------------- E1 judges

def mlp_losses(d: dict, K: int, epochs: int, seed: int = 0) -> list[dict]:
    """Meta-Sift's backbone (run_meta_sift_baseline.TabularMLP, AdamW 1e-3, batches of 128 poisoned rows + all of V,
    z-scored features), trained plainly; after each epoch every poisoned row's loss of its given label (eval mode)."""
    import torch.nn.functional as F
    from run_meta_sift_baseline import TabularMLP
    dev = d["Xc"].device
    torch.manual_seed(seed)
    A = torch.cat([d["Xc"], d["Xv"]]).float()
    mu, sd = A.mean(0), A.std(0, unbiased=False)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    Xp, Xv = (d["Xc"].float() - mu) / sd, (d["Xv"].float() - mu) / sd
    yp = torch.as_tensor(d["y_given"], dtype=torch.long, device=dev)
    yv = torch.as_tensor(d["yv"], dtype=torch.long, device=dev)
    model = TabularMLP(K, n_features=Xp.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for ep in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(yp), generator=gen).split(128):
            b = b.to(dev)
            opt.zero_grad()
            F.cross_entropy(model(torch.cat([Xp[b], Xv])), torch.cat([yp[b], yv])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xp)
            loss = F.cross_entropy(logits, yp, reduction="none").cpu().numpy()
            fit = (logits.argmax(1) == yp).cpu().numpy()
        out.append({"epoch": ep, "loss": loss, "fit_given": fit})
    return out


def stage_judge(a) -> None:
    cache = cache_for(a.device)
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            if done(attack, tag, "judge") and not a.force:
                continue
            t0 = time.time()
            bd, d = load(cache, attack, ds, tag)
            n, k, pois, clf = d["n"], d["k"], d["pois"], bd.clf
            res = {"attack": attack, "instance": tag, "n": n, "k": k, "n_v": len(d["yv"]), "judges": {}}
            X = torch.cat([d["Xc"], d["Xv"]])
            y = np.concatenate([d["y_given"], d["yv"]])

            def record(name, p):
                s = -np.log(np.clip(p, 1e-12, None))
                res["judges"][name] = {"auroc": auroc(pois, s), "prec_at_k": prec_at_k(pois, s, k),
                                       "ce_poisoned": float(s[pois].mean()), "ce_clean": float(s[~pois].mean()),
                                       "p_poisoned_median": float(np.median(p[pois])),
                                       "p_clean_median": float(np.median(p[~pois]))}

            record("V alone (V-filter)", p_given(clf, d["Xv"], d["yv"], d["Xc"], d["y_given"], chunk=a.chunk))
            record("poisoned set + V, leave-one-out", p_given(clf, X, y, d["Xc"], d["y_given"], np.arange(n), a.chunk))
            clean = np.flatnonzero(~pois)
            own = np.full(n, -1)
            own[clean] = np.arange(len(clean))
            record("clean rows + V (oracle judge)",
                   p_given(clf, torch.cat([d["Xc"][bd.idx(clean)], d["Xv"]]), np.concatenate([d["y_given"][clean], d["yv"]]),
                           d["Xc"], d["y_given"], own, a.chunk))
            gp = root_of(attack) / "runs" / tag / "greedy_pct1.npz"
            if gp.exists():
                pos = np.load(gp)["pos"]
                acc_order = np.argsort(np.where(pos < n, pos, np.inf), kind="stable")[:n - k]  # accepted, in order
                for f in (0.25, 0.5, 1.0):
                    T = np.sort(acc_order[:int(round(f * (n - k)))])
                    self_rows = np.full(n, -1)
                    self_rows[T] = np.arange(len(T))
                    record(f"greedy's trusted set at {int(100 * f)}% of its growth",
                           p_given(clf, torch.cat([d["Xc"][bd.idx(T)], d["Xv"]]), np.concatenate([d["y_given"][T], d["yv"]]),
                                   d["Xc"], d["y_given"], self_rows, a.chunk))
            # model-free local structure
            Z = zscore(d)
            nn_ = knn(Z.to(d["Xc"].device), n)
            same = (d["y_given"][nn_] == d["y_given"][:, None]).mean(1)
            record("10 nearest neighbours' label agreement", np.clip(same, 1e-3, 1.0))
            nn_pois = pois[nn_].mean(1)
            maj = np.array([np.bincount(d["y_given"][r], minlength=bd.K).argmax() for r in nn_])
            res["structure"] = {
                "poisoned: share of its 10 NN poisoned": float(nn_pois[pois].mean()),
                "clean: share of its 10 NN poisoned": float(nn_pois[~pois].mean()),
                "poisoned: share of its 10 NN with its given label": float(same[pois].mean()),
                "clean: share of its 10 NN with its given label": float(same[~pois].mean()),
                "poisoned rows whose label is their 10 NN's majority": float((maj == d["y_given"])[pois].mean()),
                "clean rows whose label is their 10 NN's majority": float((maj == d["y_given"])[~pois].mean()),
                "poisoned rows whose TRUE label is their 10 NN's given-label majority":
                    float((maj == d["y_true"])[pois].mean()),
            }
            # Meta-Sift's backbone, plain training
            res["mlp"] = []
            for e in mlp_losses(d, bd.K, a.mlp_epochs):
                res["mlp"].append({"epoch": e["epoch"], "auroc": auroc(pois, e["loss"]), "prec_at_k": prec_at_k(pois, e["loss"], k),
                                   "loss_poisoned": float(e["loss"][pois].mean()), "loss_clean": float(e["loss"][~pois].mean()),
                                   "fits_given_poisoned": float(e["fit_given"][pois].mean()),
                                   "fits_given_clean": float(e["fit_given"][~pois].mean())})
            # every method's final ranking (where its weights or removal are saved)
            res["methods"] = method_rankings(attack, tag, d)
            res["seconds"] = round(time.time() - t0, 1)
            save(attack, tag, "judge", res)
            j = res["judges"]
            print(f"judge {attack} {tag}: " + "  ".join(f"{nm[:22]} {v['auroc']:.2f}" for nm, v in j.items())
                  + f"  | MLP ep1 {res['mlp'][0]['auroc']:.2f} ep{a.mlp_epochs} {res['mlp'][-1]['auroc']:.2f}  ({res['seconds']:.0f} s)",
                  flush=True)


def method_rankings(attack: str, tag: str, d: dict) -> dict:
    """AUROC and precision of each method's final ranking or removal (saved weights / removed rows)."""
    pois, n, k = d["pois"], d["n"], d["k"]
    rd = root_of(attack) / "runs" / tag
    out = {}
    scores = {"metasift": ("metasift.npz", "tuned", True), "tabsift (sigmoid)": ("tabsift.npz", "sig_hard_it100", True),
              "tabsift_m1": ("tabsift_m1.npz", "m1_it100", True), "tabsift_m1 random start": ("tabsift_m1_rand_s0.npz", "rand_s0_it300", True),
              "tabsift_m1 random, vote of 5": ("tabsift_m1_rand_ens.npz", "vote", True)}
    for name, (f, key, high_clean) in scores.items():
        p = rd / f
        if p.exists():
            z = np.load(p)
            if key in z.files:
                w = z[key].astype(np.float64)
                s = -w if high_clean else w
                out[name] = {"auroc": auroc(pois, s), "prec_at_k": prec_at_k(pois, s, k)}
    for name in ("vfilter", "greedy_pct10", "greedy_pct1", "greedy_m1", "attn_topk_it300"):
        p = rd / f"{name}.npz"
        if p.exists():
            z = np.load(p)
            if "removed" in z.files:
                rem = z["removed"]
                out[name] = {"prec_at_k": float(pois[rem].mean()) if len(rem) else None}
                if "pos" in z.files:  # acceptance order: later = more suspicious
                    out[name]["auroc"] = auroc(pois, z["pos"].astype(np.float64))
    return out


# ---------------------------------------------------------------- E2 gradients

def grad_logw(clf, d: dict, Xq, yq, lw0: torch.Tensor, chunk: int) -> tuple[float, np.ndarray]:
    """(loss, d loss / d log w) with the poisoned rows (given labels) as the context at log-weights lw0."""
    from tabfm_experiments.row_weights import evaluate_weighted
    lw = lw0.detach().clone().requires_grad_(True)
    loss, g, _ = evaluate_weighted(clf, d["Xc"], d["y_given"], Xq, np.asarray(yq), lw, need_grad=True, batch_size=chunk)
    return float(loss), torch.nan_to_num(g).detach().cpu().numpy()


def stage_grad(a) -> None:
    from scipy.stats import spearmanr
    import tinyv_sift as T
    cache = cache_for(a.device)
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            if done(attack, tag, "grad") and not a.force:
                continue
            t0 = time.time()
            bd, d = load(cache, attack, ds, tag)
            n, k, pois, clf = d["n"], d["k"], d["pois"], bd.clf
            dev = d["Xc"].device
            s = T.vfilter_scores(clf, d)
            held = np.zeros(n, dtype=bool)
            held[np.argsort(-s, kind="stable")[n - k:]] = True
            points = {"w = 1": torch.zeros(n, device=dev),
                      "V-filter split, relaxed (M1's start)": torch.as_tensor(np.where(held, math.log(0.05), math.log(0.95)),
                                                                             dtype=torch.float32, device=dev)}
            Z = zscore(d)
            dist_v = torch.cdist(Z[:n], Z[n:]).min(1).values.cpu().numpy()
            rng = np.random.default_rng(0)
            tq = np.sort(rng.choice(len(bd.yt), min(a.test_queries, len(bd.yt)), replace=False))
            res = {"attack": attack, "instance": tag, "n": n, "k": k, "n_v": len(d["yv"]), "points": {}}
            for pname, lw0 in points.items():
                lv, gv = grad_logw(clf, d, d["Xv"], d["yv"], lw0, a.chunk)
                lt, gt = grad_logw(clf, d, bd.Xt[bd.idx(tq)], bd.yt[tq], lw0, a.chunk)
                av, at = np.abs(gv), np.abs(gt)
                top = max(1, n // 100)
                near = dist_v <= np.quantile(dist_v, 0.10)
                far = dist_v >= np.quantile(dist_v, 0.50)
                both = lambda m: bool(pois[m].any() and (~pois[m]).any())  # noqa: E731
                r = {"loss_v": lv, "loss_test": lt,
                     "auroc_g_v": auroc(pois, gv), "auroc_g_test": auroc(pois, gt),
                     "prec_g_v": prec_at_k(pois, gv, k), "prec_g_test": prec_at_k(pois, gt, k),
                     "spearman_gv_gt": float(spearmanr(gv, gt)[0]),
                     "top1pct_mass_gv": float(np.sort(av)[::-1][:top].sum() / max(av.sum(), 1e-30)),
                     "top1pct_mass_gt": float(np.sort(at)[::-1][:top].sum() / max(at.sum(), 1e-30)),
                     "spearman_absgv_negdist": float(spearmanr(av, -dist_v)[0]),
                     "auroc_g_v_near10": auroc(pois[near], gv[near]) if both(near) else None,
                     "auroc_g_v_far50": auroc(pois[far], gv[far]) if both(far) else None,
                     "auroc_g_test_far50": auroc(pois[far], gt[far]) if both(far) else None}
                if pname != "w = 1":  # at the split, among the kept rows only (what M1 can still remove)
                    kept = ~held
                    r["auroc_g_v_kept"] = auroc(pois[kept], gv[kept])
                    r["auroc_g_test_kept"] = auroc(pois[kept], gt[kept])
                res["points"][pname] = r
            res["seconds"] = round(time.time() - t0, 1)
            save(attack, tag, "grad", res)
            print(f"grad {attack} {tag}: " + " | ".join(f"{p}: V {r['auroc_g_v']:.2f} test {r['auroc_g_test']:.2f}"
                                                      for p, r in res["points"].items()) + f"  ({res['seconds']:.0f} s)", flush=True)


# ---------------------------------------------------------------- E3 objective landscape

def split_eval(bd, d: dict, removed: np.ndarray) -> dict:
    """V's loss with the kept rows alone as the context, the sift objective at this hard split, the test metrics."""
    from tabfm_experiments.row_weights import evaluate_weighted
    n, k = d["n"], len(removed)
    keep = np.ones(n, dtype=bool)
    keep[removed] = False
    v_ce, v_acc = B.v_score(bd, d, keep, None)
    rows = np.flatnonzero(keep)
    X = torch.cat([d["Xc"][bd.idx(rows)], d["Xv"]])
    y = np.concatenate([d["y_given"][rows], d["yv"]])
    lw = torch.zeros(len(y), device=X.device)
    l_held = evaluate_weighted(bd.clf, X, y, d["Xc"][bd.idx(removed)], d["y_given"][removed], lw, need_grad=False,
                               batch_size=2048)[0] if k else 0.0
    return {"precision": float(d["pois"][removed].mean()) if k else None, "n_removed": int(k), "v_ce": v_ce, "v_acc": v_acc,
            "sift_objective": float(l_held) * k / n, "test": B.test_scores(bd, d, keep)}


def method_splits(attack: str, tag: str, d: dict) -> dict:
    n, k = d["n"], d["k"]
    rd = root_of(attack) / "runs" / tag
    out = {}
    for name in ("vfilter", "greedy_pct10", "greedy_pct1", "greedy_m1", "attn_topk_it300"):
        p = rd / f"{name}.npz"
        if p.exists() and "removed" in np.load(p).files:
            out[name] = np.sort(np.load(p)["removed"])
    for name, (f, key) in {"tabsift (sigmoid)": ("tabsift.npz", "sig_hard_it100"), "tabsift_m1": ("tabsift_m1.npz", "m1_it100"),
                           "tabsift_m1 random start": ("tabsift_m1_rand_s0.npz", "rand_s0_it300"),
                           "tabsift_m1 random, vote of 5": ("tabsift_m1_rand_ens.npz", "vote"),
                           "metasift": ("metasift.npz", "tuned")}.items():
        p = rd / f
        if p.exists() and key in np.load(p).files:
            w = np.load(p)[key]
            out[name] = np.sort(np.argsort(-w, kind="stable")[n - k:])
    return out


def stage_landscape(a) -> None:
    import tinyv_sift as T
    cache = cache_for(a.device)
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            if done(attack, tag, "landscape") and not a.force:
                continue
            t0 = time.time()
            bd, d = load(cache, attack, ds, tag)
            n, k, pois = d["n"], d["k"], d["pois"]
            s = T.vfilter_scores(bd.clf, d)  # log p(given | V): low = implausible to V
            P, C = np.flatnonzero(pois), np.flatnonzero(~pois)
            rng = np.random.default_rng(0)
            cands = []
            for p in (0.3, 0.5, 0.7, 0.9, 1.0):
                n_p = int(round(p * k))
                n_c = k - n_p
                for choice in (("random", "hardest", "easiest") if n_c else ("random",)):
                    for rep in range(2 if choice == "random" and n_c else 1):
                        rp = rng.choice(P, n_p, replace=False)
                        if choice == "random":
                            rc = rng.choice(C, n_c, replace=False)
                        else:
                            order = C[np.argsort(s[C] if choice == "hardest" else -s[C], kind="stable")]
                            rc = order[:n_c]
                        cands.append({"kind": "constructed", "precision_target": p, "clean_choice": choice, "rep": rep,
                                      "removed": np.sort(np.concatenate([rp, rc]))})
            for name, rem in method_splits(attack, tag, d).items():
                cands.append({"kind": "method", "method": name, "removed": rem})
            res = {"attack": attack, "instance": tag, "n": n, "k": k, "n_v": len(d["yv"]), "splits": [],
                   "no_cleaning": B.test_scores(bd, d)}
            for cd in cands:
                r = {key: val for key, val in cd.items() if key != "removed"} | split_eval(bd, d, cd["removed"])
                res["splits"].append(r)
            res["seconds"] = round(time.time() - t0, 1)
            save(attack, tag, "landscape", res)
            ora = next(x for x in res["splits"] if x.get("precision_target") == 1.0)
            vf = next((x for x in res["splits"] if x.get("method") == "vfilter"), None)
            print(f"landscape {attack} {tag}: oracle V-loss {ora['v_ce']:.3f} sift-obj {ora['sift_objective']:.3f} test auc "
                  f"{ora['test']['auc']:.3f}" + (f" | V-filter V-loss {vf['v_ce']:.3f} sift-obj {vf['sift_objective']:.3f} test auc "
                                                 f"{vf['test']['auc']:.3f}" if vf else "") + f"  ({res['seconds']:.0f} s)", flush=True)


# ---------------------------------------------------------------- E4 which clean rows each method removes

def stage_removed(a) -> None:
    """Every method's removal (method_splits; and the oracle and a random removal for reference): how many clean rows it
    removes, how hard they are (their plausibility p(true label) under the clean rows + V as the context, leave-one-out:
    the oracle judge), and how it shifts the class mix of the kept rows (total variation between the kept rows' given-label
    mix and the clean rows' true-label mix; the share of the clean minority-class rows it removes)."""
    cache = cache_for(a.device)
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            if done(attack, tag, "removed") and not a.force:
                continue
            t0 = time.time()
            bd, d = load(cache, attack, ds, tag)
            n, k, pois = d["n"], d["k"], d["pois"]
            clean = np.flatnonzero(~pois)
            own = np.full(n, -1)
            own[clean] = np.arange(len(clean))
            p_true = p_given(bd.clf, torch.cat([d["Xc"][bd.idx(clean)], d["Xv"]]), np.concatenate([d["y_given"][clean], d["yv"]]),
                             d["Xc"], d["y_true"], own, a.chunk)
            K = bd.K
            clean_mix = np.bincount(d["y_true"][clean], minlength=K) / len(clean)
            minority = int(np.argmin(np.bincount(d["y_true"], minlength=K)))
            splits = method_splits(attack, tag, d) | {"oracle": np.flatnonzero(pois),
                                                      "random": np.sort(np.random.default_rng(0).choice(n, k, replace=False))}
            res = {"attack": attack, "instance": tag, "n": n, "k": k, "K": K, "minority_class": minority,
                   "clean_mix": clean_mix.tolist(), "p_true_clean_median": float(np.median(p_true[clean])), "methods": {}}
            for name, rem in splits.items():
                keep = np.ones(n, dtype=bool)
                keep[rem] = False
                rc = rem[~pois[rem]]
                kept_mix = np.bincount(d["y_given"][keep], minlength=K) / max(keep.sum(), 1)
                res["methods"][name] = {
                    "precision": float(pois[rem].mean()), "clean_removed": int(len(rc)),
                    "p_true_removed_clean_median": float(np.median(p_true[rc])) if len(rc) else None,
                    "removed_clean_hard_share": float((p_true[rc] < 0.5).mean()) if len(rc) else None,
                    "clean_hard_removed_share": float(np.isin(np.flatnonzero((~pois) & (p_true < 0.5)), rc).mean()),
                    "minority_clean_removed_share": float(np.isin(np.flatnonzero((~pois) & (d["y_true"] == minority)), rc).mean()),
                    "kept_mix_tv": float(0.5 * np.abs(kept_mix - clean_mix).sum())}
            res["seconds"] = round(time.time() - t0, 1)
            save(attack, tag, "removed", res)
            print(f"removed {attack} {tag}: " + "  ".join(f"{m} {v['precision']:.2f}/{v['removed_clean_hard_share'] if v['removed_clean_hard_share'] is None else round(v['removed_clean_hard_share'], 2)}"
                                                          for m, v in res["methods"].items()) + f"  ({res['seconds']:.0f} s)", flush=True)


# ---------------------------------------------------------------- E7 the marginal value of deleting one row vs a cluster

def stage_influence(a) -> None:
    """Real deletions instead of gradients: the change in test log loss (kept rows + V) when deleting one poisoned or one
    clean row (random samples), when deleting a poisoned row together with the poisoned rows among its 10 nearest
    neighbours (its cluster), and when deleting the same number of poisoned rows scattered at random. Redundant (coherent)
    poison: a single deletion does nothing, a cluster deletion helps far more than a scattered one."""
    from tabfm_experiments.row_weights import evaluate_weighted
    cache = cache_for(a.device)
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            if done(attack, tag, "influence") and not a.force:
                continue
            t0 = time.time()
            bd, d = load(cache, attack, ds, tag)
            n, pois = d["n"], d["pois"]
            X = torch.cat([d["Xc"], d["Xv"]])
            y = np.concatenate([d["y_given"], d["yv"]])
            yt = bd.yt

            def test_ce(deleted: np.ndarray) -> float:
                lw = torch.zeros(len(y), device=X.device)
                lw[torch.as_tensor(np.asarray(deleted, dtype=np.int64), device=X.device)] = float("-inf")
                return float(evaluate_weighted(bd.clf, X, y, bd.Xt, yt, lw, need_grad=False, batch_size=2048)[0])

            base = test_ce(np.zeros(0, dtype=np.int64))
            rng = np.random.default_rng(0)
            P, C = np.flatnonzero(pois), np.flatnonzero(~pois)
            sp, sc = rng.choice(P, a.n_rows, replace=False), rng.choice(C, a.n_rows, replace=False)
            single_p = [test_ce(np.array([r])) - base for r in sp]
            single_c = [test_ce(np.array([r])) - base for r in sc]
            nn_ = knn(zscore(d), n)
            clus, scat, sizes = [], [], []
            for r in sp[:a.n_clusters]:
                members = np.unique(np.concatenate([[r], nn_[r][pois[nn_[r]]]]))
                sizes.append(len(members))
                clus.append((test_ce(members) - base) / len(members))
                scat.append((test_ce(rng.choice(P, len(members), replace=False)) - base) / len(members))
            res = {"attack": attack, "instance": tag, "base_test_ce": base, "single_poisoned": single_p, "single_clean": single_c,
                   "cluster_per_row": clus, "scattered_per_row": scat, "cluster_sizes": sizes,
                   "auroc_single_deletion": auroc(np.r_[np.ones(len(single_p), bool), np.zeros(len(single_c), bool)],
                                                  -np.r_[single_p, single_c]),
                   "seconds": round(time.time() - t0, 1)}
            save(attack, tag, "influence", res)
            print(f"influence {attack} {tag}: single deletion: poisoned {np.mean(single_p):+.2e} clean {np.mean(single_c):+.2e} "
                  f"(AUROC {res['auroc_single_deletion']:.2f}) | per row: cluster {np.mean(clus):+.2e} vs scattered "
                  f"{np.mean(scat):+.2e} (cluster size {np.mean(sizes):.1f})  ({res['seconds']:.0f} s)", flush=True)


# ---------------------------------------------------------------- E6 greedy's acceptance order

def stage_trace(a) -> None:
    for attack in a.attacks:
        for ds, c, tag in instances(a):
            rd = root_of(attack) / "runs" / tag
            z = np.load(root_of(attack) / "draws" / f"{tag}.npz")
            n = len(z["ctx_rows"])
            pois = np.zeros(n, dtype=bool)
            pois[z["pois_rows"]] = True
            res = {"attack": attack, "instance": tag, "methods": {}}
            for m in ("greedy_pct10", "greedy_pct1", "greedy_m1"):
                p = rd / f"{m}.npz"
                if not p.exists() or "pos" not in np.load(p).files:
                    continue
                pos = np.load(p)["pos"]
                acc = np.argsort(np.where(pos < n, pos, np.inf), kind="stable")[: int((pos < n).sum())]
                tenths = [float(pois[ch].mean()) for ch in np.array_split(acc, 10)]
                res["methods"][m] = {"poisoned share by tenth of acceptance": tenths,
                                     "removed precision": float(pois[pos >= n].mean())}
            save(attack, tag, "trace", res)


# ---------------------------------------------------------------- report

def stage_report(a) -> None:
    import report_why_tinyv  # noqa: F401 - kept separate so the tables can be iterated on without rerunning
    report_why_tinyv.main([])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "judge", "grad", "landscape", "removed", "influence", "trace", "report"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attacks", nargs="+", choices=ATTACKS, default=["hub-flip-local"])
    ap.add_argument("--datasets", nargs="+", choices=list(B.DATASETS), default=list(B.DATASETS))
    ap.add_argument("--cs", type=int, nargs="+", default=list(B.CS))
    ap.add_argument("--chunk", type=int, default=1024, help="query rows per refit of the context (memory)")
    ap.add_argument("--test-queries", type=int, default=500, help="E2: test rows in the test-loss gradient")
    ap.add_argument("--mlp-epochs", type=int, default=6, help="E1: the MLP's epochs (Meta-Sift: 1 warmup + 5)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-rows", type=int, default=40, help="E7: single deletions per kind (poisoned, clean)")
    ap.add_argument("--n-clusters", type=int, default=25, help="E7: cluster and scattered deletions")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    fn = {"draws": stage_draws, "judge": stage_judge, "grad": stage_grad, "landscape": stage_landscape, "removed": stage_removed,
          "influence": stage_influence, "trace": stage_trace, "report": stage_report}[a.stage]
    if a.stage in ("judge", "grad", "landscape", "removed", "influence", "trace"):
        per_instance(fn, a)
    else:
        fn(a)


if __name__ == "__main__":
    main()

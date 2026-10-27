#!/usr/bin/env python
"""Tuning the free choices of the V + P benchmark on development draws, before the benchmark runs.

The benchmark: a dataset's fixed train and test sets; V = c clean rows per class drawn from the train set (c = 1, 3,
5), P = the rest, poisoned (label-flip-bernoulli at R = 45%: every row independently, with probability R, moved to a
uniformly random other class; or hub-flip-local at R = 30%). A method sees V, P and R and builds TabPFN's context; it
is scored by TabPFN on the test set (ROC-AUC, balanced accuracy, log loss). Choices settled here:

  vfilter    the V-filter's removal threshold (benchmark method 3): every rule on p = TabPFN(V)'s probability of a
             row's given label, with and without a floor of the bottom R%
  greedy     greedy at 1% per step (benchmark methods 4 and 5) run past (1 - R)% all the way to V + P, every step's
             statistics and test metrics recorded, so that any stopping rule can be scored afterwards
  baselines  EMLC, IEG and KNN-Shapley at their defaults (bench_tinyv.DEFAULTS), on CPU: their scores, relabels and
             values, for the two forms (remove the top R%, or their own rule)
  bleval     the baselines' two forms scored on the test set (GPU)

Development draws: the 6 datasets x c = 1, 3, 5 x the two poisonings x seeds 100 and 101 (the benchmark uses seeds
0-4). Seed s draws V (bench_small_v.draw_per_class) and P's random flips (run_id = s). k = round(R% of P): what R
says to remove, not the realised number of flips. Results in results/vp_tune/.

    python scripts/vp_tune.py draws
    python scripts/vp_tune.py vfilter --device cuda:1 --shard 0/3
    python scripts/vp_tune.py greedy --device cuda:1 --shard 0/3
    python scripts/vp_tune.py baselines --threads 4 --shard 0/16
    python scripts/vp_tune.py bleval --device cuda:1
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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_tinyv as B  # noqa: E402

ROOT = Path("results/vp_tune")
ATTACKS = {"rand45": ("label-flip-bernoulli", 45.0), "hub30": ("hub-flip-local", 30.0)}
SEEDS = (100, 101)
CS = (1, 3, 5)
BASELINES = ("emlc", "ieg", "knn_shapley")


def tag_of(ds: str, attack: str, c: int, seed: int) -> str:
    return f"{ds}_{attack}_c{c}_s{seed}"


def instances(shard: str | None = None) -> list[tuple[str, str, int, int]]:
    out = [(ds, a, c, s) for ds in B.DATASETS for a in ATTACKS for c in CS for s in SEEDS]
    if shard:
        i, n = map(int, shard.split("/"))
        out = out[i::n]
    return out


def build_draw(bd: B.BenchData, attack: str, c: int, seed: int) -> dict:
    from bench_small_v import draw_per_class
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.sampling import k_from_percent
    name, R = ATTACKS[attack]
    v_rows = draw_per_class(bd.y, c, seed)
    ctx_rows = np.setdiff1d(np.arange(len(bd.y)), v_rows)
    rows, y_pois = poison(name, bd.X[bd.idx(ctx_rows)], bd.y[ctx_rows], bd.K, R, run_id=seed, row_seed=Seeds.row_seed)
    return dict(v_rows=v_rows, ctx_rows=ctx_rows, y_given=np.asarray(y_pois).astype(np.int64),
                pois_rows=np.asarray(rows, dtype=np.int64), k=np.int64(k_from_percent(len(ctx_rows), R)), R=R)


def load_draw(bd: B.BenchData, ds: str, attack: str, c: int, seed: int) -> dict:
    z = np.load(ROOT / "draws" / f"{tag_of(ds, attack, c, seed)}.npz")
    d = {key: z[key] for key in ("v_rows", "ctx_rows", "y_given", "pois_rows")}
    d |= {"k": int(z["k"]), "R": float(z["R"]), "n": len(d["ctx_rows"]), "seed": seed, "K": bd.K,
          "tag": tag_of(ds, attack, c, seed)}
    d["pois"] = np.zeros(d["n"], dtype=bool)
    d["pois"][d["pois_rows"]] = True
    d["Xc"], d["Xv"] = bd.X[bd.idx(d["ctx_rows"])], bd.X[bd.idx(d["v_rows"])]
    d["yv"], d["y_true"] = bd.y[d["v_rows"]], bd.y[d["ctx_rows"]]
    return d


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def removal_stats(d: dict, removed: np.ndarray) -> dict:
    pois = d["pois"]
    return {"n_removed": int(removed.sum()), "share_removed": float(removed.mean()),
            "prec": float(pois[removed].mean()) if removed.any() else None,
            "recall": float(pois[removed].sum() / max(pois.sum(), 1)),
            "kept_pois_share": float(pois[~removed].mean()) if (~removed).any() else None}


def references(bd: B.BenchData, d: dict) -> dict:
    return {"v_only": B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt),
            "v_plus_p": B.test_scores(bd, d, np.ones(d["n"], dtype=bool)),
            "oracle": B.test_scores(bd, d, ~d["pois"])}


def bottom(score: np.ndarray, k: int) -> np.ndarray:
    """The k lowest scores as a mask (ties: file order)."""
    out = np.zeros(len(score), dtype=bool)
    out[np.argsort(score, kind="stable")[:k]] = True
    return out


def gmm_low(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Rows whose posterior under the lower-mean component of a 2-component Gaussian mixture on x exceeds 0.5."""
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed).fit(x[:, None])
    low = int(np.argmin(g.means_.ravel()))
    return g.predict_proba(x[:, None])[:, low] > 0.5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def vfilter_rules(P: np.ndarray, yg: np.ndarray, k: int, K: int) -> dict:
    """Removal masks of every candidate rule on V-only TabPFN's probabilities P (n x K) of the poisoned rows."""
    n = len(yg)
    p = P[np.arange(n), yg]
    rules = {"R": bottom(p, k)}
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        rules[f"abs{tau:g}"] = p < tau
    for a in (0.25, 0.5, 0.75, 1.0, 1.25):  # relative to chance, 1/K
        rules[f"chance{a:g}"] = p < a / K
    for b in (0.5, 0.75, 1.0):  # relative to the most probable class; b = 1: the given label is not the argmax
        rules[f"top{b:g}"] = p < b * P.max(1)
    t = np.array([P[yg == cl, cl].mean() if (yg == cl).any() else 1.0 for cl in range(K)])
    for lam in (0.5, 0.75, 1.0):  # confident learning: a class's mean self-confidence as its threshold
        rules[f"cl{lam:g}"] = p < lam * t[yg]
    try:
        rules["gmm"] = gmm_low(logit(p))
    except Exception:  # noqa: BLE001 - one value only: no mixture
        rules["gmm"] = np.zeros(n, dtype=bool)
    for name in [r for r in rules if r != "R"]:  # each with the bottom R% removed as well (removal >= R%)
        rules[f"{name}+R"] = rules[name] | rules["R"]
    return rules


def run_vfilter(bd: B.BenchData, d: dict) -> dict:
    t0 = time.time()
    P = bd.probs(d["Xv"], d["yv"], d["Xc"], d["y_given"])
    rules = vfilter_rules(P, d["y_given"], d["k"], bd.K)
    cache, out = {}, {}
    for name, removed in rules.items():
        key = np.packbits(removed).tobytes()
        if key not in cache:
            cache[key] = B.test_scores(bd, d, ~removed) if (~removed).any() else None
        out[name] = removal_stats(d, removed) | {"test": cache[key]}
    (ROOT / "vfilter").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "vfilter" / f"{d['tag']}.npz", P=P.astype(np.float32))
    return {"tag": d["tag"], "n": d["n"], "k": d["k"], "K": bd.K, "n_flipped": int(d["pois"].sum()),
            "refs": references(bd, d), "rules": out, "seconds": round(time.time() - t0, 1),
            "host": socket.gethostname(), "device": bd.device}


def batch_stats(probs: np.ndarray, yg: np.ndarray, K: int) -> dict:
    """Statistics of a set of rows' predicted distributions (probs, m x K) against their given labels."""
    m = len(yg)
    p = probs[np.arange(m), yg]
    other = probs.copy()
    other[np.arange(m), yg] = -1.0
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1) / math.log(K)
    return {"p_mean": float(p.mean()), "p_min": float(p.min()), "p_median": float(np.median(p)),
            "p_chance": float(((p - 1 / K) / (1 - 1 / K)).mean()), "agree": float((probs.argmax(1) == yg).mean()),
            "margin": float((p - other.max(1)).mean()), "maxprob": float(probs.max(1).mean()), "entropy": float(ent.mean())}


def run_greedy(bd: B.BenchData, d: dict) -> dict:
    """Greedy at 1% per step (iterative_filter's grow:pct1, as bench_tinyv's greedy_pct1: m = ceil(1% of n), the m
    rows whose given label TabPFN with V + the accepted rows finds most plausible join), through (1 - R)% exactly
    (that batch is cut to reach it, as greedy_pct1 stops there) and on until P is used up. Before each batch: the
    batch's and the pool's statistics; after it: the test metrics of V + the accepted rows."""
    t0 = time.time()
    n, k, K, yg = d["n"], d["k"], bd.K, d["y_given"]
    m = math.ceil(0.01 * n)
    in_T = np.zeros(n, dtype=bool)
    states = [{"accepted": 0, "test": B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)}]
    steps = []
    while not in_T.all():
        pool = np.flatnonzero(~in_T)
        acc_rows = np.flatnonzero(in_T)
        X_ctx = torch.cat([d["Xc"][bd.idx(acc_rows)], d["Xv"]])
        y_ctx = np.concatenate([yg[acc_rows], d["yv"]])
        probs = bd.probs(X_ctx, y_ctx, d["Xc"][bd.idx(pool)], yg[pool])
        p = probs[np.arange(len(pool)), yg[pool]]
        s = -np.log(np.clip(p, 1e-12, None))
        accepted = int(in_T.sum())
        size = min(m, (n - k) - accepted) if accepted < n - k else min(m, len(pool))
        order = np.argsort(s, kind="stable")[:size]
        batch = pool[order]
        st = {"accepted_before": accepted, "size": size, "batch": batch_stats(probs[order], yg[batch], K),
              "pool": batch_stats(probs, yg[pool], K), "batch_pois": float(d["pois"][batch].mean()),
              "pool_pois": float(d["pois"][pool].mean())}
        if len(pool) >= 20:
            try:
                st["pool"]["gmm_clean"] = float(1 - gmm_low(logit(p)).mean())
            except Exception:  # noqa: BLE001 - a degenerate pool (one value) has no mixture
                st["pool"]["gmm_clean"] = None
        steps.append(st)
        in_T[batch] = True
        states.append({"accepted": int(in_T.sum()), "test": B.test_scores(bd, d, in_T),
                       "kept_pois_share": float(d["pois"][in_T].mean())})
    return {"tag": d["tag"], "n": n, "k": k, "K": K, "m": m, "n_flipped": int(d["pois"].sum()), "steps": steps,
            "states": states, "refs": references(bd, d), "seconds": round(time.time() - t0, 1),
            "host": socket.gethostname(), "device": bd.device}


def run_baselines(bd: B.BenchData, d: dict, methods=BASELINES) -> dict:
    """Each baseline's default configuration (bench_tinyv.DEFAULTS): its suspicion score, its relabels (IEG, EMLC)."""
    import small_v_methods as M
    feats = B.Feats(bd, d)
    arrays, meta = {}, {}
    for method in methods:
        kw = dict(B.DEFAULTS[method])
        feat = kw.pop("feat", "z")
        t0 = time.time()
        res = M.METHODS[method](feats.draw(feat), **kw)
        arrays[f"{method}_score"] = np.asarray(res.score, dtype=np.float64)
        if res.relabel is not None:
            arrays[f"{method}_relabel"] = np.asarray(res.relabel, dtype=np.int16)
        meta[method] = {"seconds": round(time.time() - t0, 1)}
    return meta, arrays


def baseline_forms(method: str, z, d: dict) -> dict:
    """Removal masks: the top R% by the score, or the method's own rule (IEG, EMLC: the relabel disagrees with the
    given label; KNN-Shapley: a negative value)."""
    from bench_small_v import top_k
    score = z[f"{method}_score"]
    top = np.zeros(d["n"], dtype=bool)
    top[top_k(score, d["k"], d["seed"])] = True
    own = (z[f"{method}_relabel"].astype(np.int64) != d["y_given"]) if f"{method}_relabel" in z.files else score > 0
    return {"R": top, "own": own, "own+R": own | top}


def eval_baselines(bd: B.BenchData, d: dict) -> dict:
    z = np.load(ROOT / "baselines" / f"{d['tag']}.npz")
    out = {}
    for method in BASELINES:
        for form, removed in baseline_forms(method, z, d).items():
            out[f"{method}_{form}"] = removal_stats(d, removed) | {
                "test": B.test_scores(bd, d, ~removed) if (~removed).any() else None}
    return {"tag": d["tag"], "rows": out, "host": socket.gethostname()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "vfilter", "greedy", "baselines", "bleval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None, help="i/N: every N-th instance, from the i-th")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = "cpu" if a.stage in ("draws", "baselines") else a.device
    bd = None
    for ds, attack, c, seed in instances(a.shard):
        tag = tag_of(ds, attack, c, seed)
        if bd is None or bd.ds != ds:
            bd = B.BenchData(ds, device)
        if a.stage == "draws":
            path = ROOT / "draws" / f"{tag}.npz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **build_draw(bd, attack, c, seed))
                print(f"draw {tag}", flush=True)
            continue
        out = ROOT / a.stage / f"{tag}.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        d = load_draw(bd, ds, attack, c, seed)
        t0 = time.time()
        if a.stage == "vfilter":
            res = run_vfilter(bd, d)
        elif a.stage == "greedy":
            res = run_greedy(bd, d)
        elif a.stage == "baselines":
            meta, arrays = run_baselines(bd, d)
            np.savez_compressed(ROOT / "baselines" / f"{tag}.npz", **arrays)
            res = {"tag": tag, "methods": meta, "host": socket.gethostname()}
        else:
            res = eval_baselines(bd, d)
        write_json(out, res)
        print(f"{a.stage} {tag}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

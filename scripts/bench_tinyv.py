#!/usr/bin/env python
"""Tiny-V poisoning benchmark: how much of the clean performance does V + a cleaned poisoned set recover?

An instance is (dataset, c, R, seed):
  - V = c random rows of every class from the dataset's clean train context (c = 1, 3, 5);
  - the rest of the context is poisoned with hub-flip-local at R% (R = 10, 20, 30): k = R% of its rows
    get a wrong label;
  - the held-out test file is the test set.
Every method sees the same instance and returns a cleaned version of the poisoned rows (the k rows it
removes, or new labels). It is scored by TabPFNv2 with V + the cleaned rows as the context, on the test
set: accuracy, balanced accuracy, ROC-AUC (binary: of class 1; multi-class: one-vs-rest macro) and log
loss. Also reported: the precision of the removed rows (the share that is poisoned; for relabelling, of
the rows whose label changed).

Fairness: a method may use only V, the poisoned rows (features and given labels), k and the number of
classes. The test set and the poisoned-row identity are never used, except to score the result (and by
the oracle reference).

Methods (--methods; default: all):
  refs          V only; poisoned + V; oracle + V (the poisoned rows removed by their true identity)
  vfilter       TabPFN with V as its context scores every poisoned row's given label; the k least
                plausible removed
  greedy_pct10  greedy growth of a trusted set from V (schedule_study.py): every step TabPFN with V + the
  greedy_pct1   accepted rows scores the rest, and the most plausible join, 10% of the poisoned rows per
  greedy_m1     step, 1% per step, or one row per step, until k rows are left out
  attn_topk     learned attention weights, top-k version (M1), and soft version (A3), 20 updates each
  attn_soft     (tinyv_learned.py)
  nan_topk      the feature-input (NaN) top-k gate, 20 updates per restart, a 4-restart vote (tinyv_learned.py)
  label_spread, knn_shapley, emlc, ieg
                the four strongest label-noise baselines of the earlier tiny-V benchmark (small_v_methods.py);
                the other six (knn_sanitize, data_oob, cleanlab, simifeat, glc, mwnet) still run if named
  metasift      Meta-Sift with V on the train side of every batch; tabsift: the same soft split learned through
  tabsift       TabPFN's row weights (tinyv_sift.py; w = sigmoid(t) from the hard V-filter split, Adam lr 5). Each
                deployed as the top (1 - R)% + V (overall, and class by class as in the poisoned rows) and the top
                2% + V (a small clean subset), at its default configuration and V-selected (as the baselines); test
                metrics recorded along the training

The baselines' hyperparameters. The earlier benchmark tuned them on clean held-out rows of the context
file with the poisoned rows' identity, which is not allowed here, so each baseline is reported:
  default      the method's default configuration (DEFAULTS below: the defaults of small_v_methods.py,
               z-scored features), removal mode; also its relabelling mode where it has one
  V-selected   every configuration of its grid (small_v_methods.GRIDS) in every mode it has (remove,
               relabel, relabel_k, flagged), and the one with the lowest log loss of V under TabPFN with
               the cleaned rows alone as the context (V is not in that context). It uses only V and the
               poisoned rows. It is biased towards configurations that echo V's labels, and at one row per
               class it rests on K rows. --no-vsel skips it (default configurations only)
Our methods run at fixed settings chosen before the benchmark (tinyv_learned.py documents where they
come from); nothing is tuned per instance.

Stages:
  draws         build every instance once (copied from results/sched/draws when the schedule study
                already built it) into <root>/draws. Run it ONCE and copy the folder to every machine: the
                draws are the shared ground truth, and workers never build them
  import-sched  copy the schedule study's seed-0 results that match a draw exactly (vfilter, greedy_pct1,
                greedy_pct10 and refs for gas-drift, heloc, students_dropout and coil2000 at c = 1, 5)
  grid          the baselines' configuration grids, one job per (instance, baseline), saved to
                <root>/grids. Needs no GPU except for cleanlab (TabPFN out-of-fold probabilities)
  run           every method's job per instance -> <root>/runs/<instance>/<method>.json. A baseline uses
                its grid file (computed on the spot with --grid-if-missing)
  status        what is done, per method x dataset

Jobs are claimed through lock files, so any number of workers on one machine (or on machines that share
the folder) can split them; --shard i/N splits the job list without a shared folder (merge the runs/
and grids/ folders afterwards). Finished jobs are skipped.

    python scripts/bench_tinyv.py draws --device cuda:1
    python scripts/bench_tinyv.py import-sched
    python scripts/bench_tinyv.py grid --device cpu --methods emlc ieg                      # CPU machine
    python scripts/bench_tinyv.py grid --device cuda:1 --methods label_spread knn_shapley
    python scripts/bench_tinyv.py run --device cuda:1 --methods refs vfilter greedy_pct10 greedy_pct1
    python scripts/bench_tinyv.py run --device cuda:2 --methods greedy_m1 --datasets gas-drift heloc
    python scripts/bench_tinyv.py run --device cuda:1 --methods attn_topk attn_soft nan_topk --shard 0/2
    python scripts/bench_tinyv.py status
    python scripts/report_bench_tinyv.py

Cost: about 26 H100-hours (23 with --no-vsel) and 4 CPU worker-hours (8 threads each) for all 54
instances, of which the one-row-per-step greedy takes about 20 H100-hours (gas-drift and coil2000 most);
docs/bench_tinyv.md has the breakdown.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("BENCH_TINYV_ROOT", "results/bench_tinyv"))
DATASETS = {  # dataset: (clean train context, held-out test set)
    "gas-drift": ("datasets/gas-drift/splits/context_5000.csv", "datasets/gas-drift/splits/test_attack_1000.csv"),
    "heloc": ("datasets/heloc/splits/selected/natural/context_5000.csv", "datasets/heloc/splits/test_attack_1000.csv"),
    "ecommerce_shipping": ("datasets/ecommerce_shipping/splits/selected/natural/context_5000.csv",
                           "datasets/ecommerce_shipping/splits/test_attack_1000.csv"),
    "students_dropout": ("datasets/students_dropout/splits/context_3000.csv",
                         "datasets/students_dropout/splits/test_attack_500.csv"),
    "is_this_a_good_customer": ("datasets/is_this_a_good_customer/splits/train.csv",
                                "datasets/is_this_a_good_customer/splits/test.csv"),
    "coil2000_insurance_policies": ("datasets/coil2000_insurance_policies/splits/selected/natural/context_5000.csv",
                                    "datasets/coil2000_insurance_policies/splits/test_attack_1000.csv"),
}
CS, RS, SEEDS = (1, 3, 5), (10.0, 20.0, 30.0), (0,)
ATTACK = "hub-flip-local"
GREEDY = {"vfilter": "oneshot", "greedy_pct10": "fixed:pct10", "greedy_pct1": "fixed:pct1", "greedy_m1": "fixed:m1"}
LEARNED = ("attn_topk", "attn_soft", "nan_topk")
# the four strongest baselines of the earlier tiny-V benchmark (mean accuracy over its 30 cells: IEG 79.1,
# KNN-Shapley 76.2, EMLC 75.3, label spreading 74.8; the other six 66.7-73.7), run by default
BASELINES = ("label_spread", "knn_shapley", "emlc", "ieg")
ALL_BASELINES = ("knn_sanitize", "label_spread", "knn_shapley", "data_oob", "cleanlab", "simifeat", "glc", "mwnet", "emlc", "ieg")
# sifting with V fixed on the train side (tinyv_sift.py): Meta-Sift, and the same split learned through TabPFN
SIFT = ("metasift", "tabsift")
METHODS = ("refs", *GREEDY, *LEARNED, *BASELINES, *SIFT)
# the attention learners at another update count than tinyv_learned's 20 (scripts/tinyv_iter_study.py): M1 as
# attn_topk_it<N>, the best soft form (w / V-filter / PGD, deployed soft + V) as attn_wsoft_it<N> (lr 0.8) or
# attn_wsoft_lr<x>_it<N>
ITERS_NAME = re.compile(r"attn_(topk|wsoft)(?:_lr([0-9.]+))?_it(\d+)")
DEFAULTS = {  # the baselines' default configurations: small_v_methods.py's function defaults, z-scored features
    "knn_sanitize": {"feat": "z", "k_nn": 10, "eta": 0.5, "rounds": 1},
    "label_spread": {"feat": "z", "n_neighbors": 10, "alpha": 0.9, "beta": 0.0},
    "knn_shapley": {"feat": "z", "K_nn": 5},
    "data_oob": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "anchor_v": False},
    "cleanlab": {"model": "tabpfn", "method": "self_confidence"},
    "simifeat": {"feat": "z", "k_nn": 10, "metric": "cosine", "views": 1, "variant": "rank"},
    "glc": {"feat": "z", "lr": 1e-3, "epochs": 40, "trusted_weight": 1.0},
    "mwnet": {"feat": "z", "lr": 0.05, "meta_lr": 1e-3, "epochs": 40},
    "emlc": {"feat": "z", "lr": 0.02, "teacher_lr": 1e-3, "epochs": 40},
    "ieg": {"feat": "z", "lr": 1e-3, "kl_weight": 1.0, "epochs": 40},
}
SCHED = Path("results/sched")  # the schedule study (draws and seed-0 runs to import)


def tag_of(ds: str, R: float, c: int, seed: int) -> str:
    return f"{ds}_R{R:g}_c{c}_s{seed}"


def cfg_str(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def claim(lock: Path) -> bool:
    """Atomically take a job; a lock left by a dead process on this host is taken over."""
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


# ---------------------------------------------------------------- data, draws, scoring

class BenchData:
    """One dataset's clean context and test set on one device; TabPFNv2 for its class count, built on
    first use (a CPU-only grid job for the neural baselines never needs it)."""

    def __init__(self, ds: str, device: str):
        import gate_context as G
        ctx, test = DATASETS[ds]
        self.ds, self.device = ds, device
        self.X, self.y = G.load(Path(ctx), device)
        self.Xt, self.yt = G.load(Path(test), device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        self.K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(self.y)))
        self._clf = None

    @property
    def clf(self):
        if self._clf is None:
            from tabfm_experiments.config import Seeds
            from tabfm_experiments.runtime import build_tabpfn_v2
            self._clf = build_tabpfn_v2(self.device, Seeds.model_seed, recompute_layers=True, n_classes=self.K)
        return self._clf

    def idx(self, rows):
        return torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)

    def probs(self, X_ctx, y_ctx, Xq, yq=None) -> np.ndarray:
        import gate_context as G
        yq = np.zeros(len(Xq), dtype=np.int64) if yq is None else yq
        return G.evaluate(self.clf, X_ctx, torch.as_tensor(np.asarray(y_ctx), dtype=torch.float32, device=self.device), Xq,
                          torch.as_tensor(np.asarray(yq), dtype=torch.long, device=self.device), None, None,
                          need_grad=False, batch_size=None)[2]


def build_draw(bd: BenchData, R: float, c: int, seed: int) -> dict:
    """The construction of schedule_study.get_draw / bench_small_v.build_draw."""
    from bench_small_v import draw_per_class
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.sampling import k_from_percent
    v_rows = draw_per_class(bd.y, c, seed)
    ctx_rows = np.setdiff1d(np.arange(len(bd.y)), v_rows)
    rows, y_pois = poison(ATTACK, bd.X[bd.idx(ctx_rows)], bd.y[ctx_rows], bd.K, R, run_id=0, row_seed=Seeds.row_seed)
    k = k_from_percent(len(ctx_rows), R)
    rows = np.asarray(rows, dtype=np.int64)
    assert len(rows) == k and np.all(np.asarray(y_pois)[rows] != bd.y[ctx_rows][rows])
    return dict(v_rows=v_rows, ctx_rows=ctx_rows, y_given=np.asarray(y_pois).astype(np.int64), pois_rows=rows, k=np.int64(k))


def load_draw(bd: BenchData, tag: str, seed: int) -> dict:
    path = ROOT / "draws" / f"{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run the draws stage first (once) and share the folder")
    z = np.load(path)
    d = {key: z[key] for key in ("v_rows", "ctx_rows", "y_given", "pois_rows")}
    d |= {"k": int(z["k"]), "n": len(d["ctx_rows"]), "tag": tag, "seed": seed}
    d["pois"] = np.zeros(d["n"], dtype=bool)
    d["pois"][d["pois_rows"]] = True
    d["Xc"], d["Xv"] = bd.X[bd.idx(d["ctx_rows"])], bd.X[bd.idx(d["v_rows"])]
    d["yv"], d["y_true"] = bd.y[d["v_rows"]], bd.y[d["ctx_rows"]]
    return d


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from schedule_study import metrics as m
    return m(p, y)


def test_scores(bd: BenchData, d: dict, keep: np.ndarray | None = None, labels: np.ndarray | None = None) -> dict:
    """TabPFN with V + the kept poisoned rows (their given labels, or `labels`) as the context, on the test set."""
    keep = np.ones(d["n"], dtype=bool) if keep is None else keep
    lab = d["y_given"] if labels is None else np.asarray(labels)
    rows = np.flatnonzero(keep)
    X_ctx = torch.cat([d["Xc"][bd.idx(rows)], d["Xv"]])
    y_ctx = np.concatenate([lab[rows], d["yv"]])
    return metrics(bd.probs(X_ctx, y_ctx, bd.Xt, bd.yt), bd.yt)


def v_score(bd: BenchData, d: dict, keep: np.ndarray | None, labels: np.ndarray | None) -> tuple[float, float]:
    """(log loss, accuracy) of V under TabPFN with the cleaned poisoned rows alone as the context; (inf, 0)
    when the cleaned rows lose a whole class."""
    keep = np.ones(d["n"], dtype=bool) if keep is None else keep
    lab = (d["y_given"] if labels is None else np.asarray(labels))[keep]
    if len(np.unique(lab)) < bd.K:
        return float("inf"), 0.0
    p = bd.probs(d["Xc"][bd.idx(np.flatnonzero(keep))], lab, d["Xv"], d["yv"])
    ce = float(-np.log(np.clip(p[np.arange(len(d["yv"])), d["yv"]], 1e-12, None)).mean())
    return ce, float((p.argmax(1) == d["yv"]).mean())


def removal_row(name: str, bd: BenchData, d: dict, removed: np.ndarray, **extra) -> dict:
    keep = np.ones(d["n"], dtype=bool)
    keep[removed] = False
    return {"name": name, "test": test_scores(bd, d, keep), "prec": float(d["pois"][removed].mean()) if len(removed) else None,
            "n_removed": int(len(removed)), **extra}


# ---------------------------------------------------------------- methods

def run_refs(bd: BenchData, d: dict, log=print) -> dict:
    zv = metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)
    return {"rows": [{"name": "v_only", "test": zv},
                     {"name": "poisoned_plus_v", "test": test_scores(bd, d)},
                     removal_row("oracle_plus_v", bd, d, np.flatnonzero(d["pois"]))]}


def run_greedy(bd: BenchData, d: dict, method: str, log=print) -> tuple[dict, dict]:
    from schedule_study import run_schedule
    res, pos = run_schedule(bd, d, GREEDY[method], seed=d["seed"])
    row = {"name": method, "test": res["test"], "prec": res["prec"], "n_removed": int((pos == d["n"]).sum()),
           "steps": res["steps"], "seconds": res["seconds"]}
    return {"rows": [row], "spec": GREEDY[method]}, {"removed": np.flatnonzero(pos == d["n"]), "pos": pos}


def run_learned(bd: BenchData, d: dict, method: str, a, log=print) -> tuple[dict, dict]:
    import tinyv_learned as L
    if method == "nan_topk":
        out = L.nan_topk_gate(bd.clf, d, restarts=a.restarts, steps=a.max_steps, log=log)
        rows = [removal_row(method, bd, d, out["removed"], seconds=out["seconds"], votes_hist=out["votes_hist"])]
    elif ITERS_NAME.fullmatch(method):
        variant, lr, steps = ITERS_NAME.fullmatch(method).groups()
        out = L.attn_weights(bd.clf, d, variant, steps=a.max_steps or int(steps), lr=float(lr) if lr else None, log=log)
        if variant == "topk":
            rows = [removal_row(method, bd, d, out["removed"], seconds=out["seconds"], steps=out["steps"])]
        else:  # soft + V; the rows at w = 0 are deleted exactly, and their precision is reported
            zero = np.flatnonzero(torch.isinf(out["log_w"]).cpu().numpy())
            p = L.soft_plus_v_probs(bd.clf, d, out["log_w"], bd.Xt, bd.yt)
            rows = [{"name": method, "test": metrics(p, bd.yt), "prec": float(d["pois"][zero].mean()) if len(zero) else None,
                     "n_removed": int(len(zero)), "seconds": out["seconds"], "steps": out["steps"]}]
            out["removed"] = zero
    else:
        out = L.attn_weights(bd.clf, d, method.split("_")[1], steps=a.max_steps, log=log)
        rows = [removal_row(method, bd, d, out["removed"], seconds=out["seconds"], steps=out["steps"])]
        if "log_w" in out:  # the soft weights themselves, V added at weight 1 (extra)
            p = L.soft_plus_v_probs(bd.clf, d, out["log_w"], bd.Xt, bd.yt)
            rows.append({"name": f"{method}_softw", "test": metrics(p, bd.yt), "prec": None, "n_removed": 0})
    cfg = {key: (val if not isinstance(val, float) or np.isfinite(val) else str(val)) for key, val in out["config"].items()}
    return {"rows": rows, "config": cfg}, {"removed": out["removed"]}


SIFT_KINDS = (("full", ""), ("full_cb", "_cb"), ("small", "_small"))  # deployment, row-name suffix


def run_sift(bd: BenchData, d: dict, method: str, log=print) -> tuple[dict, dict]:
    """Every configuration of the method's grid (tinyv_sift.GRIDS; for tabsift every checkpoint of it) gives weights
    on the poisoned rows, deployed three ways with V added: the top (1 - R)% (the k lowest removed), the top (1 - R)%
    class by class (each given class its share of the poisoned rows: Meta-Sift's rule) and a small clean subset (the
    top 2% by that rule). Rows: the fixed configurations (tinyv_sift.ROWS; the default first) and the V-selected one
    (the lowest log loss of V under TabPFN with the kept rows alone as the context, as for the baselines), each
    deployment separately. While the recorded configurations train (tabsift: both starts; metasift: the default),
    every few iterations the three deployments' test metrics, the precision of the removed rows and the purity of
    the kept ones go to the trajectory."""
    import tinyv_sift as T
    n, k = d["n"], d["k"]

    def keep_of(w: np.ndarray, kind: str) -> np.ndarray:
        kept = (np.argsort(-w, kind="stable")[:n - k] if kind == "full"
                else T.class_proportional(w, d["y_given"], n - k if kind == "full_cb" else int(round(T.SMALL * n))))
        keep = np.zeros(n, dtype=bool)
        keep[kept] = True
        return keep

    def deployed(w: np.ndarray) -> dict:
        out = {}
        for kind, _ in SIFT_KINDS:
            keep = keep_of(w, kind)
            out[kind] = {"test": test_scores(bd, d, keep), "prec": float(d["pois"][~keep].mean()),
                         "purity": float(1 - d["pois"][keep].mean()), "n_removed": int(n - keep.sum())}
        return out

    traj, cands, seconds, used = [], [], {}, {}
    for name, cfg in T.GRIDS[method].items():
        recorded = method == "tabsift" or name == T.ROWS[method][""][0]

        def record(weights, name=name, **kw):
            traj.append({"cfg": name, **kw, "w_mean": float(np.mean(weights)), "deploy": deployed(weights)})

        if method == "metasift":
            out = T.metasift(d, cfg, log=log, record=record if recorded else None)
            cands.append((name, None, out["weights"]))
            if recorded:
                record(out["weights"], sifter="ensemble", it=None, objective=None, train_loss=None)
        else:
            out = T.tabsift(bd.clf, d, cfg, log=log, record=record)
            cands += [(name, it, ck["w"]) for it, ck in out["ckpts"].items() if it > 0 or out["config"]["init"] != "ones"]
        seconds[name] = out["seconds"]
        used[name] = {key: (list(val) if isinstance(val, tuple) else val) for key, val in out["config"].items()}

    rows, scored, arrays = [], [], {}
    for kind, suffix in SIFT_KINDS:
        cache = {}
        for order, (name, it, w) in enumerate(cands):
            keep = keep_of(w, kind)
            key = np.packbits(keep).tobytes()
            if key not in cache:
                cache[key] = v_score(bd, d, keep, None)
            scored.append({"deploy": kind, "cfg": name, "ckpt": it, "v_ce": cache[key][0], "v_acc": cache[key][1],
                           "order": order})

        def row(label: str, name: str, it, extra: dict) -> dict:
            w = next(w for nm, i, w in cands if nm == name and i == it)
            keep = keep_of(w, kind)
            return {"name": label, "cfg": name, "ckpt": it, "test": test_scores(bd, d, keep),
                    "prec": float(d["pois"][~keep].mean()), "purity": float(1 - d["pois"][keep].mean()),
                    "n_removed": int(n - keep.sum()), **extra}

        mine = [s for s in scored if s["deploy"] == kind]
        for extra, (cname, cit) in T.ROWS[method].items():
            rows.append(row(f"{method}{suffix}{extra}", cname, cit, {}))
        best = min(mine, key=lambda s: (s["v_ce"], -s["v_acc"], s["order"]))
        rows.append(row(f"{method}{suffix}_vsel", best["cfg"], best["ckpt"], {"v_ce": best["v_ce"], "v_acc": best["v_acc"]}))
        log(f"    {method}{suffix}: V-selected {best['cfg']} at {best['ckpt']} (V log loss {best['v_ce']:.3f}) of "
            f"{len(mine)} candidates, {len(cache)} distinct kept sets")
    for name, it, w in cands:
        arrays[f"{name}" + ("" if it is None else f"_it{it}")] = np.asarray(w, dtype=np.float32)
    return {"rows": rows, "trajectory": traj, "v_scores": scored, "grid": {nm: cfg for nm, cfg in T.GRIDS[method].items()},
            "fixed_rows": {key: list(val) for key, val in T.ROWS[method].items()}, "config_used": used,
            "seconds_per_config": seconds}, arrays


class Feats:
    """The baselines' feature views (small_v_methods.Draw): z-scores and normal quantiles, both fitted on the
    poisoned rows + V (never the test set)."""

    def __init__(self, bd: BenchData, d: dict):
        from sklearn.preprocessing import QuantileTransformer
        A = torch.cat([d["Xc"], d["Xv"]]).double().cpu().numpy()
        mu, sd = A.mean(0), A.std(0)
        sd[sd < 1e-8] = 1.0
        qt = QuantileTransformer(n_quantiles=min(1000, len(A)), output_distribution="normal", random_state=0).fit(A)
        self.views = {"z": (A - mu) / sd, "q": qt.transform(A)}
        self.bd, self.d, self.extra = bd, d, {}

    def draw(self, feat: str):
        import small_v_methods as M
        bd, d = self.bd, self.d
        B = torch.as_tensor(np.asarray(self.views[feat], dtype=np.float32), device=bd.device)

        def tabpfn_probs(tr, te):  # cleanlab: V + the other folds' poisoned rows as the context
            X_ctx = torch.cat([d["Xc"][bd.idx(tr)], d["Xv"]])
            return bd.probs(X_ctx, np.concatenate([d["y_given"][tr], d["yv"]]), d["Xc"][bd.idx(te)], d["y_given"][te])

        return M.Draw(B[:d["n"]], d["y_given"], B[d["n"]:], d["yv"], K=bd.K, k=d["k"], seed=d["seed"],
                      tabpfn_probs=tabpfn_probs, extra=self.extra)


def baseline_configs(method: str, limit: int | None = None) -> list[dict]:
    """The default first, then the grid (without a repeat of the default)."""
    import small_v_methods as M
    grid = [cfg for cfg in M.GRIDS[method] if cfg_str(cfg) != cfg_str(DEFAULTS[method])]
    return [DEFAULTS[method]] + (grid[:limit] if limit is not None else grid)


def compute_grid(bd: BenchData, d: dict, method: str, limit: int | None = None, log=print) -> tuple[dict, dict]:
    """Every configuration's outputs (scores, relabelling, best guesses, cleanlab's issue set)."""
    import small_v_methods as M
    feats, arrays, entries = Feats(bd, d), {}, []
    for i, cfg in enumerate(baseline_configs(method, limit)):
        kw = dict(cfg)
        feat = kw.pop("feat", "z")
        t0 = time.time()
        try:
            res = M.METHODS[method](feats.draw(feat), **kw)
        except Exception as e:  # noqa: BLE001 - one failing configuration must not stop the grid
            entries.append({"i": i, "cfg": cfg_str(cfg), "status": f"failed: {type(e).__name__}: {e}"})
            log(f"    {method} {cfg_str(cfg)}: FAILED {type(e).__name__}: {e}")
            continue
        arrays[f"{i}_score"] = np.asarray(res.score, dtype=np.float64)
        for name in ("relabel", "guess"):
            if getattr(res, name) is not None:
                arrays[f"{i}_{name}"] = np.asarray(getattr(res, name), dtype=np.int16)
        if res.flagged is not None:
            arrays[f"{i}_flagged"] = np.asarray(res.flagged, dtype=bool)
        entries.append({"i": i, "cfg": cfg_str(cfg), "status": "ok", "seconds": round(time.time() - t0, 2)})
    return {"method": method, "configs": entries, "device": bd.device, "host": socket.gethostname()}, arrays


def modes_of(method: str, cfg: dict, i: int, arrays: dict, d: dict) -> dict:
    """mode -> (kept rows or None, labels or None, the rows it flags: removed or relabelled)."""
    from bench_small_v import top_k
    get = lambda name: arrays.get(f"{i}_{name}")  # noqa: E731
    n, yg = d["n"], d["y_given"]
    out = {}
    top = top_k(get("score"), d["k"], d["seed"])
    keep = np.ones(n, dtype=bool)
    keep[top] = False
    out["remove"] = (keep, None, top)
    if get("relabel") is not None:
        lab = get("relabel").astype(np.int64)
        out["relabel"] = (None, lab, np.flatnonzero(lab != yg))
    if get("guess") is not None:
        lab = yg.copy()
        lab[top] = get("guess")[top]
        out["relabel_k"] = (None, lab, top)
    if get("flagged") is not None and (method != "cleanlab" or cfg.get("method") == "self_confidence"):
        fl = get("flagged")
        out["flagged"] = (~fl, None, np.flatnonzero(fl))
    return out


def run_baseline(bd: BenchData, d: dict, method: str, grid_meta: dict, arrays: dict, vsel: bool = True,
                 log=print) -> dict:
    """V-scores of every (configuration, mode); test scores of the default (remove, and relabel if any) and of
    the V-selected one (vsel=False: the default only). Configurations that give the same cleaned set share
    one V-score."""
    import hashlib
    scored, cache = [], {}  # scored: (V log loss, -V accuracy, order, configuration index, mode)
    ok = [x for x in grid_meta["configs"] if x["status"] == "ok" and (vsel or x["i"] == 0)]
    for order, e in enumerate(ok):
        cfg = json.loads(e["cfg"])
        for mode, (keep, labels, flagged) in modes_of(method, cfg, e["i"], arrays, d).items():
            key = hashlib.md5((b"all" if keep is None else np.packbits(keep).tobytes())
                              + (b"given" if labels is None else np.asarray(labels, dtype=np.int64).tobytes())).hexdigest()
            if vsel and key not in cache:
                cache[key] = v_score(bd, d, keep, labels)
            ce, acc = cache.get(key, (float("nan"), float("nan")))
            scored.append({"i": e["i"], "cfg": e["cfg"], "mode": mode, "v_ce": ce, "v_acc": acc, "order": order})
    if not any(s["i"] == 0 for s in scored):
        raise RuntimeError(f"{method}: the default configuration did not run")

    def row(name: str, i: int, mode: str) -> dict:
        cfg = json.loads(next(e["cfg"] for e in grid_meta["configs"] if e["i"] == i))
        keep, labels, flagged = modes_of(method, cfg, i, arrays, d)[mode]
        prec = float(d["pois"][flagged].mean()) if len(flagged) else None
        return {"name": name, "cfg": cfg_str(cfg), "mode": mode, "test": test_scores(bd, d, keep, labels), "prec": prec,
                "n_removed": int(d["n"] - keep.sum()) if keep is not None else 0,
                "n_relabelled": int((labels != d["y_given"]).sum()) if labels is not None else 0}

    default_modes = {s["mode"] for s in scored if s["i"] == 0}
    rows = []
    if "remove" in default_modes:
        rows.append(row(method, 0, "remove"))
    if "relabel" in default_modes:
        rows.append(row(f"{method}_relabel", 0, "relabel"))
    if vsel:
        best = min(scored, key=lambda s: (s["v_ce"], -s["v_acc"], s["order"]))
        rows.append(row(f"{method}_vsel", best["i"], best["mode"]) | {"v_ce": best["v_ce"], "v_acc": best["v_acc"]})
        log(f"    {method}: V-selected {best['cfg']} {best['mode']} (V log loss {best['v_ce']:.3f}) of {len(scored)} "
            f"(configuration, mode) pairs, {len(cache)} distinct cleaned sets")
    return {"rows": rows, "v_scores": scored, "grid": {k: grid_meta[k] for k in ("device", "host")},
            "n_configs": len(grid_meta["configs"]), "vsel": vsel}


# ---------------------------------------------------------------- stages

def instances(a):
    for ds in a.datasets:
        for c in a.cs:
            for R in a.Rs:
                for s in a.seeds:
                    yield ds, R, c, s


def parse_tag(tag: str) -> tuple[str, float, int, int]:
    ds, rest = tag.rsplit("_R", 1)
    R, c, s = rest.split("_")
    return ds, float(R), int(c[1:]), int(s[1:])


def read_job_list(path: Path) -> list[tuple]:
    """Lines '<instance> <method>' (e.g. 'heloc_R20_c3_s0 greedy_m1'); '#' starts a comment."""
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tag, method = line.split()
        if method not in METHODS + ALL_BASELINES and not ITERS_NAME.fullmatch(method):
            raise SystemExit(f"{path}: unknown method {method!r}")
        ds, R, c, s = parse_tag(tag)
        if ds not in DATASETS:
            raise SystemExit(f"{path}: unknown dataset in {tag!r}")
        out.append((ds, R, c, s, method))
    return out


def jobs(a, methods):
    if a.job_list:
        return read_job_list(a.job_list)
    out = [(ds, R, c, s, m) for ds, R, c, s in instances(a) for m in methods]
    if a.shard:
        i, N = (int(x) for x in a.shard.split("/"))
        out = out[i::N]
    return out


def stage_draws(a) -> None:
    (ROOT / "draws").mkdir(parents=True, exist_ok=True)
    bd = None
    for ds, R, c, s in instances(a):
        tag = tag_of(ds, R, c, s)
        path = ROOT / "draws" / f"{tag}.npz"
        if path.exists():
            continue
        src = SCHED / "draws" / f"{tag}.npz"
        if src.exists():  # the schedule study's draw (identical construction): its results stay importable
            shutil.copyfile(src, path)
            print(f"{tag}: copied from {src}", flush=True)
            continue
        if bd is None or bd.ds != ds:
            bd = BenchData(ds, a.device)
        dr = build_draw(bd, R, c, s)
        tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **dr)
        os.replace(tmp, path)
        print(f"{tag}: built on {a.device} (n {len(dr['ctx_rows'])}, k {int(dr['k'])}, V {len(dr['v_rows'])})", flush=True)


def stage_import_sched(a) -> None:
    """Copy the schedule study's seed-0 results whose draw is exactly this benchmark's draw."""
    n_done = 0
    for ds, R, c, s in instances(a):
        tag = tag_of(ds, R, c, s)
        src_dir, mine, theirs = SCHED / "runs" / tag, ROOT / "draws" / f"{tag}.npz", SCHED / "draws" / f"{tag}.npz"
        if not (src_dir.is_dir() and mine.exists() and theirs.exists()):
            continue
        z1, z2 = np.load(mine), np.load(theirs)
        if not all(np.array_equal(z1[key], z2[key]) for key in ("v_rows", "ctx_rows", "y_given", "pois_rows", "k")):
            print(f"{tag}: draw differs from the schedule study's, not imported", flush=True)
            continue
        out_dir = ROOT / "runs" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        prov = {"imported_from": str(src_dir), "imported_at": time.strftime("%Y-%m-%d %H:%M")}
        k, n = int(z1["k"]), len(z1["ctx_rows"])
        for method, spec in GREEDY.items():
            f = src_dir / f"{spec.replace(':', '_')}.json"
            out = out_dir / f"{method}.json"
            if out.exists() or not f.exists():
                continue
            r = json.loads(f.read_text())
            pos = np.load(f.with_suffix(".npz"))["pos"]
            removed = np.flatnonzero(pos == n)
            assert len(removed) == k, f
            write_json(out, {"method": method, "instance": tag, "spec": spec, "provenance": prov,
                             "rows": [{"name": method, "test": r["test"], "prec": r["prec"], "n_removed": k,
                                       "steps": r["steps"], "seconds": r["seconds"]}]})
            np.savez_compressed(out.with_suffix(".npz"), removed=removed, pos=pos)
            n_done += 1
        ref, out = src_dir / "references.json", out_dir / "refs.json"
        if ref.exists() and not out.exists():
            r = json.loads(ref.read_text())
            pois = np.zeros(n, dtype=bool)
            pois[z1["pois_rows"]] = True
            write_json(out, {"method": "refs", "instance": tag, "provenance": prov,
                             "rows": [{"name": "v_only", "test": r["v_only"]},
                                      {"name": "poisoned_plus_v", "test": r["poisoned_plus_v"]},
                                      {"name": "oracle_plus_v", "test": r["oracle_plus_v"], "prec": 1.0, "n_removed": k}]})
            n_done += 1
    print(f"imported {n_done} result files", flush=True)


def worker(a, stage: str) -> None:
    torch.set_num_threads(a.threads)
    methods = a.methods or (BASELINES if stage == "grid" else METHODS)
    if stage == "grid" and set(methods) - set(ALL_BASELINES):
        raise SystemExit(f"the grid stage is for the baselines only: {sorted(set(methods) - set(ALL_BASELINES))}")
    out_root = ROOT / ("grids" if stage == "grid" else "runs")
    todo = jobs(a, methods)
    missing = sorted({tag_of(ds, R, c, s) for ds, R, c, s, _ in todo if not (ROOT / "draws" / f"{tag_of(ds, R, c, s)}.npz").exists()})
    if missing:
        raise SystemExit(f"{len(missing)} draws missing under {ROOT / 'draws'} (e.g. {missing[0]}): build them once with the "
                         "draws stage and copy the folder here")
    if stage == "grid" and any(m not in ALL_BASELINES for *_, m in todo):
        raise SystemExit("the grid stage is for the baselines only")
    attn = ("attn_topk", "attn_soft", "tabsift")
    if any(m in attn or ITERS_NAME.fullmatch(m) for *_, m in todo):  # they need the row-weight hooks of this repository's TabPFN copy
        import tabpfn
        try:
            from tabpfn.preprocessing.steps.context_row_weights import context_row_weights_installed  # noqa: F401
        except ImportError:
            print(f"SKIPPING the attention jobs: tabpfn is imported from {Path(tabpfn.__file__).parent}, which lacks the "
                  "row-weight hooks of this repository's TabPFN (TabPFN/src, commit 86df5d0). Install that copy into "
                  "this environment (pip install -e TabPFN) and run again.", flush=True)
            todo = [j for j in todo if j[-1] not in attn and not ITERS_NAME.fullmatch(j[-1])]
            if not todo:
                raise SystemExit("nothing left to run")
    bd = None
    for ds, R, c, s, method in todo:
        tag = tag_of(ds, R, c, s)
        out_dir = out_root / tag
        out = out_dir / f"{method}.json"
        if out.exists():
            continue
        if stage == "run" and method in ALL_BASELINES and not (ROOT / "grids" / tag / f"{method}.json").exists() \
                and not a.grid_if_missing:
            continue  # its grid is not there yet (another machine's grid stage, or --grid-if-missing)
        out_dir.mkdir(parents=True, exist_ok=True)
        lock = out_dir / f"{method}.lock"
        if not claim(lock):
            continue
        t0 = time.time()
        log = lambda msg: print(msg, flush=True)  # noqa: E731
        try:
            if bd is None or bd.ds != ds:
                bd = None
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bd = BenchData(ds, a.device)
            d = load_draw(bd, tag, s)
            log(f"{stage} {tag} {method}: start (n {d['n']}, k {d['k']}, V {len(d['yv'])})")
            arrays_out = None
            if stage == "grid":
                res, arrays_out = compute_grid(bd, d, method, limit=0 if a.no_vsel else a.grid_limit, log=log)
            elif method == "refs":
                res = run_refs(bd, d, log=log)
            elif method in GREEDY:
                res, arrays_out = run_greedy(bd, d, method, log=log)
            elif method in LEARNED or ITERS_NAME.fullmatch(method):
                res, arrays_out = run_learned(bd, d, method, a, log=log)
            elif method in SIFT:
                res, arrays_out = run_sift(bd, d, method, log=log)
            else:
                gpath = ROOT / "grids" / tag / f"{method}.json"
                if gpath.exists():
                    meta, arrays = json.loads(gpath.read_text()), dict(np.load(gpath.with_suffix(".npz")))
                else:  # --grid-if-missing: computed here, and saved for the record
                    meta, arrays = compute_grid(bd, d, method, limit=0 if a.no_vsel else a.grid_limit, log=log)
                    gpath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(gpath.with_suffix(".npz"), **arrays)
                    write_json(gpath, meta)
                res = run_baseline(bd, d, method, meta, arrays, vsel=not a.no_vsel, log=log)
            res = {"method": method, "instance": tag, "device": a.device, "host": socket.gethostname(),
                   "seconds": round(time.time() - t0, 1)} | res
            if arrays_out is not None:
                np.savez_compressed(out.with_suffix(".npz"), **arrays_out)
            write_json(out, res)
            summary = "  ".join(f"{r['name']}: acc {100 * r['test']['acc']:.1f} auc {100 * r['test']['auc']:.1f} "
                                f"ce {r['test']['ce']:.3f}" + (f" prec {r['prec']:.3f}" if r.get("prec") is not None else "")
                                for r in res.get("rows", []))
            log(f"{stage} {tag} {method}: done in {time.time() - t0:.0f} s  {summary}")
        except Exception as e:  # noqa: BLE001 - a failing job must not stop the worker
            log(f"{stage} {tag} {method}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            lock.unlink(missing_ok=True)


def stage_status(a) -> None:
    if a.job_list:  # progress of one job list
        todo = read_job_list(a.job_list)
        done = [(ds, R, c, s, m) for ds, R, c, s, m in todo if (ROOT / "runs" / tag_of(ds, R, c, s) / f"{m}.json").exists()]
        print(f"{a.job_list}: {len(done)} of {len(todo)} jobs done")
        for m in METHODS + tuple(x for x in ALL_BASELINES if x not in METHODS):
            n_m = sum(1 for *_, x in todo if x == m)
            if n_m:
                print(f"  {m:<14s} {sum(1 for *_, x in done if x == m):3d} / {n_m}")
        return
    for stage, methods in (("grids", BASELINES), ("runs", METHODS)):
        print(f"\n{stage}: finished instances per method x dataset (of {len(a.cs) * len(a.Rs) * len(a.seeds)})")
        print(f"  {'method':<14s} " + " ".join(f"{ds[:10]:>10s}" for ds in a.datasets))
        for m in methods:
            cnt = [sum((ROOT / stage / tag_of(ds, R, c, s) / f"{m}.json").exists() for c in a.cs for R in a.Rs for s in a.seeds)
                   for ds in a.datasets]
            print(f"  {m:<14s} " + " ".join(f"{x:>10d}" for x in cnt))
    locks = sorted((ROOT / "runs").glob("*/*.lock")) + sorted((ROOT / "grids").glob("*/*.lock"))
    print(f"\nlocks held: {len(locks)}" + "".join(f"\n  {p.parent.name}/{p.name}: {p.read_text()}" for p in locks[:20]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "import-sched", "grid", "run", "status"])
    ap.add_argument("--device", default="cuda:0", help="cuda:<i> or cpu")
    ap.add_argument("--methods", nargs="+", choices=sorted(set(METHODS + ALL_BASELINES)), default=None)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--cs", type=int, nargs="+", default=list(CS))
    ap.add_argument("--Rs", type=float, nargs="+", default=list(RS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--shard", default=None, help="i/N: only every N-th job, starting at i")
    ap.add_argument("--job-list", type=Path, default=None,
                    help="run / grid / status: exactly these jobs, lines '<instance> <method>', in this order")
    ap.add_argument("--grid-if-missing", action="store_true", help="run: compute a baseline's grid if absent")
    ap.add_argument("--no-vsel", action="store_true",
                    help="baselines: default configurations only (grid: only the default; run: no V-selection)")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--grid-limit", type=int, default=None, help="testing only: the default + the first N grid configs")
    ap.add_argument("--max-steps", type=int, default=None, help="testing only: override the learned methods' updates")
    ap.add_argument("--restarts", type=int, default=None, help="testing only: override nan_topk's restarts")
    a = ap.parse_args(argv)
    if a.stage == "draws":
        stage_draws(a)
    elif a.stage == "import-sched":
        stage_import_sched(a)
    elif a.stage == "status":
        stage_status(a)
    else:
        worker(a, a.stage)


if __name__ == "__main__":
    main()

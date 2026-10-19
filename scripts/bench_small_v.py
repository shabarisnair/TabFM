#!/usr/bin/env python
"""Tiny clean set benchmark: greedy growth of a trusted set vs label-noise / data-cleaning baselines.

A draw is (dataset, R, c, seed): V = c random clean rows of every class from the dataset's context
file, the rest of the file poisoned with hub-flip-local at R% (k = R% of its rows flipped), and an
evaluation set. Every method sees exactly the same draw. Evaluation draws (seeds 0-2) are scored on
the dataset's test file. Tuning draws (seeds 100-102) never touch it: 20% of the context file
(stratified, seed 100) is held out as their evaluation set and V and the poisoned rows come from the
other 80%. For c = 1 and R = 20 the evaluation draws are those of night 2's V-size runs (E17).

Every method is a filter, a relabeller or a classifier of its own (small_v_methods.py); its output
is used in each of its modes:
  remove     the k most suspicious rows removed (ties broken at random); TabPFN with V + the rest
  relabel    every row relabelled; TabPFN with V + the relabelled rows
  relabel_k  the k most suspicious rows relabelled to the method's best guess, the rest kept
  flagged    (cleanlab) the method's own issue set removed
  native     (training methods) the method's own classifier
The references: V only (TabPFN with V), the V-filter (V + the rows left after removing the k whose
given label is least plausible to TabPFN with V), poisoned + V and oracle + V. Ours: the greedy
growth of iterative_filter.py with a few step schedules (OURS); every candidate is run.

Tuning, separately for every (dataset, R, c) and every method, on the tuning draws only:
  tune1      every config of every method's grid; its cheap metric per mode (remove: precision of
             the k removed rows; relabel / relabel_k: share of correct labels after relabelling;
             these use the poisoned rows' identity, which is generous) and, for native, its accuracy
             on the tuning evaluation set; the outputs are saved. Also every candidate of ours and
             the references, with TabPFN's accuracy on the tuning evaluation set
  shortlist  per mode, the SHORTLIST configs with the best mean cheap metric over the tuning seeds
             (every flagged config)
  tune2      TabPFN's accuracy on the tuning evaluation set for every shortlisted config
  select     per method: the (mode, config) with the best mean accuracy on the tuning evaluation
             sets, and the best config of every mode; ours: the best candidate
  eval       on the evaluation draws: the references, every candidate of ours and every method's
             selected configs, with the test accuracy (per-row correctness saved) and precision

Stages that run models are split into tasks (one draw, one part: ours / free / neural) claimed
through lock files, so any number of workers can share them; finished tasks are skipped.

    python scripts/bench_small_v.py draws --gpu 0
    python scripts/bench_small_v.py tune1 --gpu 0 &  python scripts/bench_small_v.py tune1 --gpu 1 &
    python scripts/bench_small_v.py shortlist
    python scripts/bench_small_v.py tune2 --gpu 0 &  ...
    python scripts/bench_small_v.py select
    python scripts/bench_small_v.py eval --gpu 0 &  ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("SMALL_V_ROOT", "results/small_v"))  # a scratch root for pipeline checks
DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")  # heaviest first
ATTACK = "hub-flip-local"
RS = (20.0, 30.0)
CS = (1, 3, 5)
EVAL_SEEDS = (0, 1, 2)
TUNE_SEEDS = (100, 101, 102)
TUNE_EVAL_FRAC, TUNE_EVAL_SEED = 0.2, 100
SHORTLIST = 4
OURS = ("grow:pct1", "grow:pct1:cprior", "grow:pct10", "grow:pct10:cprior", "grow:frac0.1", "grow:frac0.1:cprior")
REFS = ("v_only", "vfilter", "poisoned_plus_v", "oracle_plus_v")
FREE = ("knn_sanitize", "label_spread", "knn_shapley", "data_oob", "cleanlab", "simifeat")
NEURAL = ("glc", "mwnet", "emlc", "ieg")
PARTS = {"ours": None, "free": FREE, "neural": NEURAL}
FIELDS = ["draw", "dataset", "R", "c", "seed", "n", "k", "method", "cfg", "mode", "cheap", "acc", "prec", "n_removed",
          "steps", "seconds"]


def tag_of(ds: str, R: float, c: int, seed: int) -> str:
    return f"{ds}_R{R:g}_c{c}_s{seed}"


def cfg_str(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def cfg_hash(method: str, cfg: str) -> str:
    return hashlib.md5(f"{method}|{cfg}".encode()).hexdigest()[:12]


def draw_per_class(y: np.ndarray, c: int, seed: int) -> np.ndarray:
    """Sorted indices of c random rows of every class. Within a class the rows follow one random
    order fixed by the seed (drawn as in gate_context.draw_validation), so a larger c contains every
    smaller one, and c = 1 is draw_validation(y, K, seed) whenever its quotas are one per class."""
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    order = [rng.permutation(np.flatnonzero(y == cl)) for cl in classes]
    if min(len(o) for o in order) < c:
        raise ValueError(f"a class has fewer than {c} rows")
    return np.sort(np.concatenate([o[:c] for o in order]))


def top_k(score: np.ndarray, k: int, seed: int) -> np.ndarray:
    """The k highest scores, ties broken at random (seeded)."""
    tie = np.random.default_rng(seed + 7919).random(len(score))
    return np.lexsort((tie, -np.asarray(score, dtype=np.float64)))[:k]


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in FIELDS})
    os.replace(tmp, path)


def save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def read_rows(paths) -> list[dict]:
    out = []
    for p in paths:
        out += list(csv.DictReader(open(p)))
    return out


def claim(lock: Path) -> bool:
    """Atomically take a task; a lock left by a dead process on this host is taken over."""
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


# ---------------------------------------------------------------- data and models

class Data:
    """One dataset's context and test files on one device, and TabPFN for its class count."""

    def __init__(self, ds: str, device: str):
        import gate_context as G
        from eval_vfilter_attacks import CONTEXT, TEST
        from tabfm_experiments.config import Seeds
        from tabfm_experiments.runtime import build_tabpfn_v2
        self.ds, self.device = ds, device
        self.X, self.y = G.load(Path(CONTEXT[ds]), device)
        self.Xt, self.yt = G.load(Path(TEST[ds]), device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        self.K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(self.y)))
        self._build = lambda: build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=self.K)
        self._clf = None

    @property
    def clf(self):
        if self._clf is None:
            self._clf = self._build()
        return self._clf

    def idx(self, rows):
        import torch
        return torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)

    def probs(self, X_ctx, y_ctx: np.ndarray, Xq, yq: np.ndarray) -> np.ndarray:
        import torch
        import gate_context as G
        return G.evaluate(self.clf, X_ctx, torch.as_tensor(np.asarray(y_ctx), dtype=torch.float32, device=self.device), Xq,
                          torch.as_tensor(np.asarray(yq), dtype=torch.long, device=self.device), None, None,
                          need_grad=False, batch_size=None)[2]


def build_draw(data: Data, R: float, c: int, seed: int) -> dict:
    import gate_context as G
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.sampling import k_from_percent
    N = len(data.y)
    if seed in TUNE_SEEDS:
        eval_rows = G.draw_validation(data.y, int(round(TUNE_EVAL_FRAC * N)), TUNE_EVAL_SEED)
    else:
        eval_rows = np.zeros(0, dtype=np.int64)
    pool = np.setdiff1d(np.arange(N), eval_rows)
    v_rows = pool[draw_per_class(data.y[pool], c, seed)]
    ctx_rows = np.setdiff1d(pool, v_rows)
    rows, y_pois = poison(ATTACK, data.X[data.idx(ctx_rows)], data.y[ctx_rows], data.K, R, run_id=0, row_seed=Seeds.row_seed)
    k = k_from_percent(len(ctx_rows), R)
    assert len(rows) == k and np.all(np.asarray(y_pois)[rows] != data.y[ctx_rows][rows])
    return dict(v_rows=v_rows, ctx_rows=ctx_rows, eval_rows=eval_rows, y_given=np.asarray(y_pois).astype(np.int64),
                pois_rows=np.asarray(rows, dtype=np.int64), k=np.int64(k), K=np.int64(data.K))


class Ctx:
    """One draw, ready for the methods: raw rows for TabPFN, z-scored and quantile features for the
    baselines (both fitted on the poisoned + clean rows), the evaluation rows."""

    def __init__(self, data: Data, tag: str, seed: int):
        import torch
        from sklearn.preprocessing import QuantileTransformer
        z = np.load(ROOT / "draws" / f"{tag}.npz")
        self.data, self.tag, self.seed = data, tag, seed
        self.v_rows, self.ctx_rows, self.eval_rows = z["v_rows"], z["ctx_rows"], z["eval_rows"]
        self.y_given, self.k, self.K = z["y_given"], int(z["k"]), int(z["K"])
        self.n = len(self.ctx_rows)
        self.pois = np.zeros(self.n, dtype=bool)
        self.pois[z["pois_rows"]] = True
        self.y_clean = data.y[self.ctx_rows]
        self.yv = data.y[self.v_rows]
        self.Xc, self.Xv = data.X[data.idx(self.ctx_rows)], data.X[data.idx(self.v_rows)]
        if len(self.eval_rows):
            self.Xe, self.ye = data.X[data.idx(self.eval_rows)], data.y[self.eval_rows]
        else:
            self.Xe, self.ye = data.Xt, data.yt
        A = torch.cat([self.Xc, self.Xv]).double().cpu().numpy()
        mu, sd = A.mean(0), A.std(0)
        sd[sd < 1e-8] = 1.0
        qt = QuantileTransformer(n_quantiles=min(1000, len(A)), output_distribution="normal", random_state=0).fit(A)
        Xe = self.Xe.double().cpu().numpy()
        tr = {"z": lambda B: (B - mu) / sd, "q": qt.transform}
        t = lambda B: torch.as_tensor(np.asarray(B, dtype=np.float32), device=data.device)  # noqa: E731
        self.feat = {f: (t(fn(A[:self.n])), t(fn(A[self.n:])), t(fn(Xe))) for f, fn in tr.items()}
        self.extra: dict = {}  # shared by the feature views (cleanlab's out-of-fold probabilities)

    def correct(self, keep: np.ndarray | None = None, labels: np.ndarray | None = None) -> np.ndarray:
        """Per evaluation row: is TabPFN right with V + the kept rows (labels: given) as its context."""
        import torch
        labels = self.y_given if labels is None else labels
        rows = np.arange(self.n) if keep is None else np.flatnonzero(keep)
        X_ctx = torch.cat([self.Xc[self.data.idx(rows)], self.Xv])
        p = self.data.probs(X_ctx, np.concatenate([labels[rows], self.yv]), self.Xe, self.ye)
        return p.argmax(1) == self.ye

    def method_draw(self, feat: str):
        import small_v_methods as M
        Xp, Xv, _ = self.feat[feat]

        def tabpfn_probs(tr, te):
            import torch
            X_ctx = torch.cat([self.Xc[self.data.idx(tr)], self.Xv])
            return self.data.probs(X_ctx, np.concatenate([self.y_given[tr], self.yv]), self.Xc[self.data.idx(te)],
                                   self.y_given[te])

        return M.Draw(Xp, self.y_given, Xv, self.yv, K=self.K, k=self.k, seed=self.seed, tabpfn_probs=tabpfn_probs,
                      extra=self.extra)


# ---------------------------------------------------------------- one draw's work

def outputs_of(ctx: Ctx, method: str, cfg: dict):
    """Run one config; returns (result, seconds, feature view)."""
    import small_v_methods as M
    kw = dict(cfg)
    feat = kw.pop("feat", "z")
    t0 = time.time()
    res = M.METHODS[method](ctx.method_draw(feat), **kw)
    return res, time.time() - t0, feat


def mode_inputs(ctx: Ctx, method: str, cfg: dict, score, relabel, guess, flagged) -> dict:
    """mode -> (keep mask or None, labels or None, cheap metric, precision, rows removed)."""
    out = {}
    top = top_k(score, ctx.k, ctx.seed)
    keep = np.ones(ctx.n, dtype=bool)
    keep[top] = False
    prec = float(ctx.pois[top].mean())
    out["remove"] = (keep, None, prec, prec, ctx.k)
    if relabel is not None:
        out["relabel"] = (None, relabel, float((relabel == ctx.y_clean).mean()), "", 0)
    if guess is not None:
        lab = ctx.y_given.copy()
        lab[top] = guess[top]
        out["relabel_k"] = (None, lab, float((lab == ctx.y_clean).mean()), prec, 0)
    if flagged is not None and (method != "cleanlab" or cfg.get("method") == "self_confidence"):
        p = float(ctx.pois[flagged].mean()) if flagged.any() else 0.0
        out["flagged"] = (~flagged, None, p, p, int(flagged.sum()))
    return out


def base_row(ctx: Ctx, ds: str, R: float, c: int, seed: int) -> dict:
    return dict(draw=ctx.tag, dataset=ds, R=R, c=c, seed=seed, n=ctx.n, k=ctx.k)


def run_ours(ctx: Ctx, key: dict, variants=OURS, log=print):
    """The references and our candidates: rows and per-row correctness."""
    from iterative_filter import grow, parse_variant
    rows, correct = [], {}
    prior_ctx = np.bincount(ctx.y_given, minlength=ctx.K) / ctx.n
    prior_v = np.bincount(ctx.yv, minlength=ctx.K) / len(ctx.yv)

    def score_fn(acc_rows, q_rows, v):  # -log p(given label) with V + the accepted rows as the context
        import torch
        X_ctx = torch.cat([ctx.Xc[ctx.data.idx(acc_rows)], ctx.Xv])
        y_ctx = np.concatenate([ctx.y_given[acc_rows], ctx.yv])
        p = ctx.data.probs(X_ctx, y_ctx, ctx.Xc[ctx.data.idx(q_rows)], ctx.y_given[q_rows])
        if v.prior or v.cprior:  # from T's class frequencies to V's (or the context's)
            f_T = np.bincount(y_ctx, minlength=ctx.K) / len(y_ctx)
            p = p * ((prior_v if v.prior else prior_ctx) / np.maximum(f_T, 1e-12))[None, :]
            p = p / p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(q_rows)), ctx.y_given[q_rows]], 1e-12, None))

    for ref in REFS:
        t0 = time.time()
        prec, steps = "", 0
        if ref == "v_only":
            p = ctx.data.probs(ctx.Xv, ctx.yv, ctx.Xe, ctx.ye)
            cor = p.argmax(1) == ctx.ye
        elif ref == "poisoned_plus_v":
            cor = ctx.correct()
        elif ref == "oracle_plus_v":
            cor = ctx.correct(keep=~ctx.pois)
        else:  # vfilter: the one-shot V-filter (iterative_filter.py's oneshot)
            keep = grow(lambda acc_rows, q_rows: score_fn(acc_rows, q_rows, parse_variant("oneshot")), ctx.n, ctx.k,
                        len(ctx.yv), parse_variant("oneshot"), ctx.y_given)[0]
            cor, prec, steps = ctx.correct(keep=keep), float(ctx.pois[~keep].mean()), 1
        correct[ref] = cor
        rows.append(key | dict(method=ref, cfg="", mode="", acc=float(cor.mean()), prec=prec, steps=steps,
                               seconds=round(time.time() - t0, 1)))
    for spec in variants:
        v = parse_variant(spec)
        t0 = time.time()
        keep, _, _, steps = grow(lambda acc_rows, q_rows, v=v: score_fn(acc_rows, q_rows, v), ctx.n, ctx.k, len(ctx.yv), v,
                                 ctx.y_given)
        cor = ctx.correct(keep=keep)
        correct[f"ours|{spec}"] = cor
        rows.append(key | dict(method="ours", cfg=spec, mode="remove", acc=float(cor.mean()),
                               prec=float(ctx.pois[~keep].mean()), n_removed=int((~keep).sum()), steps=steps,
                               seconds=round(time.time() - t0, 1)))
        log(f"  {ctx.tag} ours {spec}: acc {100 * rows[-1]['acc']:.1f} prec {rows[-1]['prec']:.3f} steps {steps} "
            f"({rows[-1]['seconds']:.0f} s)")
    return rows, correct


def run_grid(ctx: Ctx, key: dict, methods, log=print):
    """tune1: every config of every method; rows with cheap metrics (native: accuracy) and saved outputs."""
    import small_v_methods as M
    rows, arrays = [], {}
    for method in methods:
        t_m = time.time()
        for cfg in M.GRIDS[method]:
            cs = cfg_str(cfg)
            try:
                res, sec, feat = outputs_of(ctx, method, cfg)
            except Exception as e:  # noqa: BLE001 - one failing config must not stop the draw
                log(f"  {ctx.tag} {method} {cs}: FAILED {type(e).__name__}: {e}")
                continue
            h = cfg_hash(method, cs)
            arrays[f"{h}_score"] = np.asarray(res.score, dtype=np.float64)
            for name in ("relabel", "guess"):
                if getattr(res, name) is not None:
                    arrays[f"{h}_{name}"] = np.asarray(getattr(res, name), dtype=np.int16)
            if res.flagged is not None:
                arrays[f"{h}_flagged"] = np.asarray(res.flagged, dtype=bool)
            for mode, (_, _, cheap, prec, n_rem) in mode_inputs(ctx, method, cfg, res.score, res.relabel, res.guess,
                                                                  res.flagged).items():
                rows.append(key | dict(method=method, cfg=cs, mode=mode, cheap=cheap, prec=prec, n_removed=n_rem,
                                       seconds=round(sec, 2)))
            if res.predict is not None:
                cor = res.predict(ctx.feat[feat][2]).argmax(1) == ctx.ye
                rows.append(key | dict(method=method, cfg=cs, mode="native", acc=float(cor.mean()), seconds=round(sec, 2)))
        best = max((r for r in rows if r["method"] == method and r["mode"] == "remove"), key=lambda r: r["cheap"], default=None)
        log(f"  {ctx.tag} {method}: {len(M.GRIDS[method])} configs in {time.time() - t_m:.0f} s; best precision "
            f"{best['cheap'] if best else float('nan'):.3f}")
    return rows, arrays


def modes_from_saved(ctx: Ctx, method: str, cfg: dict, arrays) -> dict:
    h = cfg_hash(method, cfg_str(cfg))
    get = lambda name: arrays[f"{h}_{name}"] if f"{h}_{name}" in arrays else None  # noqa: E731
    return mode_inputs(ctx, method, cfg, get("score"), get("relabel"), get("guess"), get("flagged"))


# ---------------------------------------------------------------- stages

def selected_key(ds, R, c) -> str:
    return f"{ds}|{R:g}|{c}"


def task_list(stage: str, datasets, Rs, cs, seeds, parts=None):
    parts = (parts or ["ours", "neural", "free"]) if stage in ("tune1", "eval") else [None]
    for ds in datasets:
        for part in parts:
            for R in Rs:
                for c in cs:
                    for s in seeds:
                        yield ds, R, c, s, part


def worker(a) -> None:
    import torch
    torch.set_num_threads(a.threads)
    device = f"cuda:{a.gpu}" if a.gpu != "cpu" else "cpu"
    seeds = a.seeds or (TUNE_SEEDS if a.stage in ("tune1", "tune2") else EVAL_SEEDS)
    if a.stage in ("tune1", "eval") and "neural" in (a.parts or PARTS) and device != "cpu":
        raise SystemExit("the neural part runs on the CPU only (--gpu cpu), so that tuning and evaluation share one device")
    out_dir = ROOT / a.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist = json.loads((ROOT / "shortlist.json").read_text()) if a.stage == "tune2" else None
    selected = (json.loads((ROOT / "selected.json").read_text())
                if a.stage == "eval" and set(a.parts or PARTS) != {"ours"} else None)
    data = {}
    log = lambda m: print(m, flush=True)  # noqa: E731
    for ds, R, c, s, part in task_list(a.stage, a.datasets, a.R, a.c, seeds, a.parts):
        tag = tag_of(ds, R, c, s)
        name = f"{tag}__{part}" if part else tag
        out = out_dir / f"{name}.csv"
        if out.exists() or not claim(out_dir / f"{name}.lock"):
            continue
        t0 = time.time()
        if ds not in data:
            data = {ds: Data(ds, device)}  # one dataset at a time
        ctx = Ctx(data[ds], tag, s)
        key = base_row(ctx, ds, R, c, s)
        log(f"{a.stage} {name}: start (n {ctx.n}, k {ctx.k}, V {len(ctx.yv)})")
        if a.stage == "tune1":
            if part == "ours":
                rows, _ = run_ours(ctx, key, log=log)
            else:
                rows, arrays = run_grid(ctx, key, PARTS[part], log=log)
                save_npz(out_dir / f"{name}.npz", **arrays)
        elif a.stage == "tune2":
            rows = []
            arrays = {}
            for p in ("free", "neural"):
                arrays.update(dict(np.load(ROOT / "tune1" / f"{tag}__{p}.npz")))
            for method, by_mode in shortlist[selected_key(ds, R, c)].items():
                for mode, cfgs in by_mode.items():
                    for cs in cfgs:
                        keep, labels, cheap, prec, n_rem = modes_from_saved(ctx, method, json.loads(cs), arrays)[mode]
                        t1 = time.time()
                        cor = ctx.correct(keep=keep, labels=labels)
                        rows.append(key | dict(method=method, cfg=cs, mode=mode, cheap=cheap, acc=float(cor.mean()),
                                               prec=prec, n_removed=n_rem, seconds=round(time.time() - t1, 2)))
        else:  # eval
            correct = {}
            if part == "ours":
                rows, correct = run_ours(ctx, key, log=log)
            else:
                rows = []
                sel = selected[selected_key(ds, R, c)]
                for method in PARTS[part]:
                    if method not in sel:
                        continue
                    by_cfg = defaultdict(list)
                    for mode, cs in sel[method]["modes"].items():
                        by_cfg[cs].append(mode)
                    for cs, modes in by_cfg.items():
                        cfg = json.loads(cs)
                        res, sec, feat = outputs_of(ctx, method, cfg)
                        mi = mode_inputs(ctx, method, cfg, res.score, res.relabel, res.guess, res.flagged)
                        for mode in modes:
                            if mode == "native":
                                cor = res.predict(ctx.feat[feat][2]).argmax(1) == ctx.ye
                                cheap, prec, n_rem = "", "", ""
                            else:
                                keep, labels, cheap, prec, n_rem = mi[mode]
                                cor = ctx.correct(keep=keep, labels=labels)
                            correct[f"{method}|{mode}"] = cor
                            rows.append(key | dict(method=method, cfg=cs, mode=mode, cheap=cheap, acc=float(cor.mean()),
                                                   prec=prec, n_removed=n_rem, seconds=round(sec, 2)))
                        log(f"  {tag} {method} {cs}: " + ", ".join(f"{m} {100 * float(correct[f'{method}|{m}'].mean()):.1f}"
                                                                  for m in modes))
            save_npz(out_dir / f"{name}_correct.npz", **{k_.replace("|", "__").replace(":", "_"): v for k_, v in correct.items()})
        write_rows(out, rows)
        (out_dir / f"{name}.lock").unlink(missing_ok=True)
        log(f"{a.stage} {name}: done in {time.time() - t0:.0f} s")


def stage_draws(a) -> None:
    (ROOT / "draws").mkdir(parents=True, exist_ok=True)
    device = f"cuda:{a.gpu}" if a.gpu != "cpu" else "cpu"
    for ds in a.datasets:
        data = Data(ds, device)
        for R in a.R:
            for c in a.c:
                for s in (a.seeds or EVAL_SEEDS + TUNE_SEEDS):
                    path = ROOT / "draws" / f"{tag_of(ds, R, c, s)}.npz"
                    if path.exists():
                        continue
                    d = build_draw(data, R, c, s)
                    save_npz(path, **d)
                    print(f"{path.name}: n {len(d['ctx_rows'])}, k {int(d['k'])}, V {len(d['v_rows'])}, "
                          f"eval {len(d['eval_rows']) or 'test'}", flush=True)


def _mean_by(rows, keys, value) -> dict:
    acc = defaultdict(list)
    for r in rows:
        if r[value] != "":
            acc[tuple(r[k] for k in keys)].append(float(r[value]))
    return acc


def stage_shortlist(a) -> None:
    rows = read_rows(sorted((ROOT / "tune1").glob("*__free.csv")) + sorted((ROOT / "tune1").glob("*__neural.csv")))
    vals = _mean_by([r for r in rows if r["mode"] != "native"], ("dataset", "R", "c", "method", "mode", "cfg"), "cheap")
    out: dict = {}
    groups = defaultdict(list)
    for (ds, R, c, method, mode, cs), v in vals.items():
        if len(v) == len(TUNE_SEEDS):
            groups[(ds, R, c, method, mode)].append((np.mean(v), cs))
    for (ds, R, c, method, mode), lst in groups.items():
        lst.sort(key=lambda t: -t[0])
        keep = lst if mode == "flagged" else lst[:SHORTLIST]
        out.setdefault(selected_key(ds, float(R), int(c)), {}).setdefault(method, {})[mode] = [cs for _, cs in keep]
    (ROOT / "shortlist.json").write_text(json.dumps(out, indent=1))
    print(f"shortlist: {len(out)} (dataset, R, c) groups, {sum(len(m) for v in out.values() for m in v.values())} "
          f"(method, mode) lists -> {ROOT / 'shortlist.json'}")


def stage_select(a) -> None:
    t1 = read_rows(sorted((ROOT / "tune1").glob("*.csv")))
    t2 = read_rows(sorted((ROOT / "tune2").glob("*.csv")))
    accs = _mean_by([r for r in t1 if r["mode"] == "native" or r["method"] == "ours"] + t2,
                    ("dataset", "R", "c", "method", "mode", "cfg"), "acc")
    cheap = _mean_by(t1, ("dataset", "R", "c", "method", "mode", "cfg"), "cheap")
    best: dict = {}
    for (ds, R, c, method, mode, cs), v in accs.items():
        if len(v) != len(TUNE_SEEDS):
            continue
        g = best.setdefault(selected_key(ds, float(R), int(c)), {}).setdefault(method, {"modes": {}, "scores": {}})
        score = (float(np.mean(v)), float(np.mean(cheap.get((ds, R, c, method, mode, cs), [0.0]) or [0.0])))
        if mode not in g["scores"] or score > g["scores"][mode]:
            g["scores"][mode] = score
            g["modes"][mode] = cs
    for groups in best.values():
        for g in groups.values():
            g["best"] = max(g["scores"], key=lambda m: g["scores"][m])
    (ROOT / "selected.json").write_text(json.dumps(best, indent=1))
    print(f"selected -> {ROOT / 'selected.json'}")
    for key in sorted(best):
        print(key, " ".join(f"{m}:{g['best']}={100 * g['scores'][g['best']][0]:.1f}" for m, g in sorted(best[key].items())))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "tune1", "shortlist", "tune2", "select", "eval"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--R", type=float, nargs="+", default=list(RS))
    ap.add_argument("--c", type=int, nargs="+", default=list(CS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=None, help="tune1 / eval: only these parts")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args(argv)
    if a.stage == "draws":
        stage_draws(a)
    elif a.stage == "shortlist":
        stage_shortlist(a)
    elif a.stage == "select":
        stage_select(a)
    else:
        worker(a)


if __name__ == "__main__":
    main()

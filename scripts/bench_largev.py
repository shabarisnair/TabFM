#!/usr/bin/env python
"""Larger-V poisoning benchmark: the attention-weight learners against the NaN gates and the references,
with V = 50 to 500 clean rows.

The draw is the soft gate's (results/gate_valsplit) and the attention study's (attn_learn.make_draw, which
checks itself against the saved soft-gate draws): V clean rows, class-stratified, drawn from the clean train
context with validation seed s (one seed gives nested draws: the V = 200 draw contains the V = 100 one); 20%
of them held out, class-stratified; the rest of the context poisoned with hub-flip-local at R% (run 0).
Grid: 6 datasets x V in {50, 100, 200, 300, 400, 500} x R in {10, 20, 30}% x one sampling of V (seed 0) = 108
instances (--seeds adds samplings: their draws are built too).

Jobs, one result file per instance and job (<root>/runs/<instance>/<job>.json):
  refs         V only; poisoned + V; oracle + V; V-filter + V (all of V scores the poisoned rows' given
               labels and the k = R% least plausible are removed); each also without V (attn_learn.references)
  attn_topk    attention weights, hard top-k (M1): PGD-L-inf lr 0.2, probe 0.05, V-filter start
  attn_soft    attention weights, log w in [-1000, 0] (A3): Adam lr 0.1, V-filter start, no prior
               (both attn_learn.run: 200 updates, the held-out log loss every 10)
  nan_soft     the soft feature-input (NaN) gate: g in [0, 1] from 1, PGD-L-inf lr 0.05, 200 updates
  nan_topk_rN  the top-k NaN gate, restart N = 0-3: scores N(0, 1) (seed N), the k lowest gated to the
               all-missing input, the gradient at the relaxed point 0.05, L-inf score steps of lr 5, 200
               updates. Once all four exist, their masks vote (the k rows masked by the most restarts, each
               restart at its chosen iteration; ties: mean score rank) into nan_topk.
               (NaN gates: the held-out log loss every 5 updates; results/tables/gate_topk/night_log.md)
Every learner fits on 80% of V and keeps the iteration with the lowest log loss on the held-out 20% (ties: the
earlier one). It is deployed hard (its k lowest rows removed) and hard + V (all of V added to the context);
the soft learners also soft and soft + V. The headline entries all add V: v_only, poisoned_plus_v,
oracle_plus_v, vfilter, attn_topk, attn_soft, nan_soft, nan_topk (suffix _nov: the same without V; _softw /
_softg: soft weights / soft gates; nan_topk_gated: the voted mask applied as the gate instead of removing the
rows). Metrics: test accuracy, balanced accuracy, ROC-AUC (binary: class 1; multi-class: one-vs-rest macro),
log loss; the precision of the removed rows.

Already done and imported, not re-run (import-attn): the attention study's runs on identical draws with the
same configurations, holdout and selection (results/attn_learn/confirm): refs and attn_topk for heloc,
coil2000 and students, attn_soft for heloc and students, x V 50 and 500 x R 10/20/30 (seed 0; the study also has seeds 1-2).

    python scripts/bench_largev.py draws --device cuda:0     # once; copy <root>/draws to every machine
    python scripts/bench_largev.py import-attn
    python scripts/bench_largev.py status                    # done / left, per job and dataset
    python scripts/bench_largev.py plan --scripts 8          # -> scripts/bench_largev_jobs/gpu_XX.{sh,txt}
    python scripts/bench_largev.py run --device cuda:1 --job-list scripts/bench_largev_jobs/gpu_01.txt
    python scripts/bench_largev.py votes --device cuda:1     # any top-k vote whose restarts are all done
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("BENCH_LARGEV_ROOT", "results/bench_largev"))
DATASETS = ("gas-drift", "heloc", "ecommerce_shipping", "students_dropout", "is_this_a_good_customer",
            "coil2000_insurance_policies")
VS, RS, SEEDS, HOLDOUT = (50, 100, 200, 300, 400, 500), (10.0, 20.0, 30.0), (0,), 0.2
ATTN = {  # the attention study's M1 and A3 (results/tables/attn_learn/findings.md), at its budget
    "attn_topk": dict(param="topk", opt="pgd-linf", lr=0.2, probe=0.05, init="vfilter"),
    "attn_soft": dict(param="log", opt="adam", lr=0.1, floor=-1000.0, init="vfilter", prior=0.0),
}
ATTN_COMMON = dict(steps=200, eval_every=10, holdout=HOLDOUT, select="ce", test_every_eval=0)
# the NaN gates at the attention learners' 200 updates (their studies ran 300)
NAN_SOFT = dict(steps=200, lr=0.05, eval_every=5)                          # results/gate_valsplit's soft gate
NAN_TOPK = dict(restarts=4, steps=200, lr=5.0, probe=0.05, eval_every=5)  # results/tables/gate_topk's best
RESTARTS = tuple(f"nan_topk_r{r}" for r in range(NAN_TOPK["restarts"]))
JOBS = ("refs", "attn_topk", "attn_soft", "nan_soft", *RESTARTS)
CONFIRM = Path("results/attn_learn/confirm")
CONFIRM_FILES = {"attn_topk": "topk_pgd-linf_lr0.2_probe0.05_init-vfilter1_it200.json",
                 "attn_soft": "log_adam_lr0.1_floor-1000_init-vfilter1_it200.json"}
TIE_SEED = 12345  # attn_learn.Param's tie order among equal keys
# the planner's cost model: seconds per update on an H100 alone, measured at tiny V (bench_tinyv's attention
# and NaN-gate jobs, 3 per GPU, divided by the 2.8 jobs per GPU measured there); it scales with the context
# size squared
U = {"gas-drift": 4.8, "coil2000_insurance_policies": 3.6, "heloc": 1.07, "students_dropout": 0.79,
     "ecommerce_shipping": 0.64, "is_this_a_good_customer": 0.14}
N_CTX = {"gas-drift": 5000, "heloc": 5000, "ecommerce_shipping": 5000, "students_dropout": 3000,
         "is_this_a_good_customer": 1400, "coil2000_insurance_policies": 5000}
UPDATES = {"refs": 4, "attn_topk": 215, "attn_soft": 215, "nan_soft": 215} | {r: 215 for r in RESTARTS}


def tag_of(ds: str, R: float, V: int, s: int) -> str:
    return f"{ds}_R{R:g}_V{V}_s{s}"


def parse_tag(tag: str) -> tuple[str, float, int, int]:
    ds, rest = tag.rsplit("_R", 1)
    R, V, s = rest.split("_")
    return ds, float(R), int(V[1:]), int(s[1:])


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


# ---------------------------------------------------------------- data and draws

class Data:
    """One dataset's clean context and test set on one device, and TabPFN v2 for its class count (built on
    first use)."""

    def __init__(self, ds: str, device: str):
        import attn_learn as A
        import gate_context as G
        self.ds, self.device = ds, device
        self.X, self.y = G.load(Path(A.CONTEXT[ds]), device)
        self.Xt, self.yt = G.load(Path(A.TEST[ds]), device)
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


def build_draw(ds: str, R: float, V: int, s: int, device: str) -> dict:
    """attn_learn.make_draw's draw (checked against results/gate_valsplit where saved) as row indices."""
    import attn_learn as A
    import gate_context as G
    d = A.make_draw(ds, R, V, s, HOLDOUT, device)
    y = G.load(Path(A.CONTEXT[ds]), "cpu")[1]
    val_rows = G.draw_validation(y, V, s)  # draw_validation_form's stratified form
    ctx_rows = np.setdiff1d(np.arange(len(y)), val_rows)
    fit_pos, hold_pos = G.split_holdout(y[val_rows], HOLDOUT, s)
    assert np.array_equal(y[ctx_rows], d["y_clean"]) and np.array_equal(y[val_rows][hold_pos], d["yh"])
    return dict(val_rows=val_rows, ctx_rows=ctx_rows, fit_pos=fit_pos, hold_pos=hold_pos,
                y_given=np.asarray(d["yp"]).astype(np.int64), pois_rows=np.flatnonzero(d["pois"]),
                k=np.int64(d["k"]), checked_against=np.str_(d["checked_against"] or ""))


def load_draw(data: Data, tag: str) -> dict:
    """The draw in attn_learn.make_draw's form (what attn_learn.run and references take)."""
    path = ROOT / "draws" / f"{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: build the draws once (draws stage) and copy the folder here")
    z = np.load(path)
    ds, R, V, s = parse_tag(tag)
    idx = lambda r: torch.as_tensor(np.asarray(r, dtype=np.int64), device=data.device)  # noqa: E731
    val_rows, ctx_rows, fit_pos, hold_pos = z["val_rows"], z["ctx_rows"], z["fit_pos"], z["hold_pos"]
    Xv, yv = data.X[idx(val_rows)], data.y[val_rows]
    n = len(ctx_rows)
    pois = np.zeros(n, dtype=bool)
    pois[z["pois_rows"]] = True
    return dict(ds=ds, R=R, V=V, seed=s, K=data.K, n=n, k=int(z["k"]), Xc=data.X[idx(ctx_rows)], y_clean=data.y[ctx_rows],
                yp=z["y_given"].astype(int), pois=pois, Xf=Xv[idx(fit_pos)], yf=yv[fit_pos], Xh=Xv[idx(hold_pos)],
                yh=yv[hold_pos], Xv=Xv, yv=yv, Xt=data.Xt, yt=data.yt, checked_against=str(z["checked_against"]) or None)


# ---------------------------------------------------------------- scoring

def score_removed(data: Data, d: dict, removed, plus_v: bool) -> dict:
    """Test metrics of TabPFN with the poisoned rows minus `removed` (+ all of V) as its context, by
    attn_learn.references' deletion path (log w = -inf deletes a row exactly)."""
    import attn_learn as A
    from tabfm_experiments.row_weights import evaluate_weighted
    dev = data.device
    lw = torch.zeros(d["n"], device=dev)
    if removed is not None and len(removed):
        lw[torch.as_tensor(np.asarray(removed, dtype=np.int64), device=dev)] = -math.inf
    X, y = d["Xc"], d["yp"]
    if plus_v:
        X, y = torch.cat([X, d["Xv"]]), np.concatenate([y, d["yv"]])
        lw = torch.cat([lw, torch.zeros(len(d["yv"]), device=dev)])
    return A.metrics(evaluate_weighted(data.clf, X, y, d["Xt"], d["yt"], lw, need_grad=False)[2], d["yt"])


def score_gated(data: Data, d: dict, g: np.ndarray, plus_v: bool) -> dict:
    """Test metrics with the poisoned rows' features gated by g (1 unchanged, 0 all missing) (+ all of V,
    ungated)."""
    import attn_learn as A
    import gate_context as G
    dev = data.device
    X, y, gate = d["Xc"], d["yp"], np.asarray(g, dtype=np.float32)
    if plus_v:
        X, y = torch.cat([X, d["Xv"]]), np.concatenate([y, d["yv"]])
        gate = np.concatenate([gate, np.ones(len(d["yv"]), dtype=np.float32)])
    y_t = torch.as_tensor(y, dtype=torch.float32, device=dev)
    null_x = G.capture_null_input(data.clf, X, y_t)
    p = G.evaluate(data.clf, X, y_t, d["Xt"], torch.as_tensor(d["yt"], dtype=torch.long, device=dev),
                   torch.as_tensor(gate, device=dev), null_x, need_grad=False, batch_size=None)[2]
    return A.metrics(p, d["yt"])


def ranking_auroc(pois: np.ndarray, suspicion: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pois, suspicion)) if np.ptp(suspicion) > 0 else 0.5


# ---------------------------------------------------------------- the jobs

def refs_rows(r: dict, k: int) -> list[dict]:
    return [{"name": "v_only", "test": r["v_only"]},
            {"name": "poisoned_plus_v", "test": r["poisoned_plus_v"]},
            {"name": "oracle_plus_v", "test": r["oracle_plus_v"], "prec": 1.0, "n_removed": k},
            {"name": "vfilter", "test": r["vfilter_plus_v"], "prec": r["vfilter_prec"], "n_removed": k},
            {"name": "poisoned", "test": r["poisoned"]},
            {"name": "oracle", "test": r["oracle"], "prec": 1.0, "n_removed": k},
            {"name": "vfilter_nov", "test": r["vfilter"], "prec": r["vfilter_prec"], "n_removed": k}]


def attn_rows(job: str, res: dict, k: int) -> list[dict]:
    """attn_learn.run's selections as report rows: hard (+ V), and for A3 also its soft weights (+ V)."""
    h, s = res["sel_hard"], res["sel_soft"]
    hard = {"prec": h["prec"], "auroc": h["auroc"], "n_removed": k, "it": h["it"]}
    rows = [{"name": job, "test": h["test_plus_v"]} | hard, {"name": f"{job}_nov", "test": h["test"]} | hard]
    if job == "attn_soft":
        rows += [{"name": "attn_soft_softw", "test": s["test_plus_v"], "it": s["it"]},
                 {"name": "attn_soft_softw_nov", "test": s["test"], "it": s["it"]}]
    return rows


def removed_at(keys: np.ndarray, its, it: int, k: int) -> np.ndarray:
    """The k lowest-ranked rows at evaluation `it`, ties broken as attn_learn.Param.deleted breaks them."""
    key = keys[list(its).index(it)]
    tie = np.random.default_rng(TIE_SEED).random(len(key))
    return np.sort(np.lexsort((tie, key))[:k])


def run_attn(data: Data, d: dict, job: str, log=print) -> tuple[dict, dict]:
    import attn_learn as A
    c = A.DEFAULTS | ATTN[job] | ATTN_COMMON | {"dataset": d["ds"], "R": d["R"], "V": d["V"], "seed": d["seed"]}
    res = A.run(c, d, data.clf, data.device, log=log)
    keys = res.pop("snap_keys")
    its = [e["it"] for e in res["evals"]]
    removed = removed_at(keys, its, res["sel_hard"]["it"], d["k"])
    return {"rows": attn_rows(job, res, d["k"]), "evals": res["evals"], "config": c}, {"removed": removed}


def run_nan_soft(data: Data, d: dict, log=print) -> tuple[dict, dict]:
    """The soft NaN gate fitted on 80% of V; the gates at the lowest held-out log loss."""
    import gate_context as G
    cfg, dev = NAN_SOFT, data.device
    n, k = d["n"], d["k"]
    yp_t = torch.as_tensor(d["yp"], dtype=torch.float32, device=dev)
    yf_t, yh_t = (torch.as_tensor(v, dtype=torch.long, device=dev) for v in (d["yf"], d["yh"]))
    null_x = G.capture_null_input(data.clf, d["Xc"], yp_t)
    g = torch.ones(n, device=dev, requires_grad=True)
    curve, snaps, t0 = [], {}, time.time()
    for it in range(cfg["steps"] + 1):
        if it % cfg["eval_every"] == 0 or it == cfg["steps"]:
            ce, _, p = G.evaluate(data.clf, d["Xc"], yp_t, d["Xh"], yh_t, g.detach(), null_x, need_grad=False, batch_size=None)
            curve.append({"it": it, "hold_ce": ce, "hold_acc": float((p.argmax(1) == d["yh"]).mean())})
            snaps[it] = g.detach().cpu().numpy().copy()
            if it % 50 == 0:
                log(f"      nan_soft it {it}: held-out CE {ce:.4f}  ({time.time() - t0:.0f} s)")
        if it == cfg["steps"]:
            break
        _, grad, _ = G.evaluate(data.clf, d["Xc"], yp_t, d["Xf"], yf_t, g, null_x, need_grad=True, batch_size=None)
        G.pgd_step(g, grad, cfg["lr"], "linf")
    best = min(curve, key=lambda e: (e["hold_ce"], e["it"]))["it"]
    gb = snaps[best]
    removed = G.lowest_gated(gb, k, np.random.default_rng(0).permutation(n))
    hard = {"prec": float(d["pois"][removed].mean()), "auroc": ranking_auroc(d["pois"], -gb), "n_removed": k, "it": best}
    rows = [{"name": "nan_soft", "test": score_removed(data, d, removed, True)} | hard,
            {"name": "nan_soft_nov", "test": score_removed(data, d, removed, False)} | hard,
            {"name": "nan_soft_softg", "test": score_gated(data, d, gb, True), "it": best},
            {"name": "nan_soft_softg_nov", "test": score_gated(data, d, gb, False), "it": best}]
    return {"rows": rows, "curve": curve, "config": cfg}, {"removed": removed, "g": gb.astype(np.float32)}


def run_nan_topk_restart(data: Data, d: dict, r: int, log=print) -> tuple[dict, dict]:
    """One restart of the top-k NaN gate fitted on 80% of V; its mask at the lowest held-out log loss."""
    import gate_context as G
    cfg, dev = NAN_TOPK, data.device
    n, k = d["n"], d["k"]
    yp_t = torch.as_tensor(d["yp"], dtype=torch.float32, device=dev)
    yf_t, yh_t = (torch.as_tensor(v, dtype=torch.long, device=dev) for v in (d["yf"], d["yh"]))
    null_x = G.capture_null_input(data.clf, d["Xc"], yp_t)
    scores = G.init_scores(n, 1.0, r, dev)
    curve, snaps, t0 = [], {}, time.time()
    for it in range(cfg["steps"] + 1):
        g = G.topk_mask(scores, k)
        if it % cfg["eval_every"] == 0 or it == cfg["steps"]:
            ce, _, p = G.evaluate(data.clf, d["Xc"], yp_t, d["Xh"], yh_t, g, null_x, need_grad=False, batch_size=None)
            curve.append({"it": it, "hold_ce": ce, "hold_acc": float((p.argmax(1) == d["yh"]).mean()),
                          "prec": float(d["pois"][(g == 0).cpu().numpy()].mean())})
            snaps[it] = scores.detach().cpu().numpy().copy()
            if it % 50 == 0:
                log(f"      nan_topk restart {r} it {it}: held-out CE {ce:.4f}  ({time.time() - t0:.0f} s)")
        if it == cfg["steps"]:
            break
        probe = (g * (1 - cfg["probe"]) + (1 - g) * cfg["probe"]).requires_grad_(True)
        grad = G.evaluate(data.clf, d["Xc"], yp_t, d["Xf"], yf_t, probe, null_x, need_grad=True, batch_size=None)[1]
        G.score_step(scores, G.ste_grad(grad, scores, "identity", k_remove=k), cfg["lr"], "linf")
    best = min(curve, key=lambda e: (e["hold_ce"], e["it"]))["it"]
    s = snaps[best]
    masked = np.sort(np.argsort(s, kind="stable")[:k])  # topk_mask's rows: the k lowest scores, stable ties
    rank = np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(n - 1, 1)  # 0 = the lowest score
    return ({"restart": r, "it": best, "prec": float(d["pois"][masked].mean()), "curve": curve, "config": cfg},
            {"masked": masked, "rank": rank.astype(np.float32)})


def run_vote(data: Data, d: dict, tag: str) -> tuple[dict, dict]:
    """The vote of the four restarts' chosen masks: the k rows masked by the most (ties: mean score rank)."""
    out = ROOT / "runs" / tag
    res = [json.loads((out / f"{r}.json").read_text()) for r in RESTARTS]
    arrs = [np.load(out / f"{r}.npz") for r in RESTARTS]
    n, k = d["n"], d["k"]
    votes = np.zeros(n)
    for a in arrs:
        votes[a["masked"]] += 1
    mean_rank = np.mean([a["rank"] for a in arrs], axis=0)
    removed = np.sort(np.lexsort((mean_rank, -votes))[:k])
    g = np.ones(n, dtype=np.float32)
    g[removed] = 0.0
    hard = {"prec": float(d["pois"][removed].mean()), "auroc": ranking_auroc(d["pois"], votes - mean_rank),
            "n_removed": k, "its": [x["it"] for x in res]}
    rows = [{"name": "nan_topk", "test": score_removed(data, d, removed, True)} | hard,
            {"name": "nan_topk_nov", "test": score_removed(data, d, removed, False)} | hard,
            {"name": "nan_topk_gated", "test": score_gated(data, d, g, True)} | hard,
            {"name": "nan_topk_gated_nov", "test": score_gated(data, d, g, False)} | hard]
    return ({"rows": rows, "restart_prec": [x["prec"] for x in res],
             "votes_hist": np.bincount(votes.astype(int), minlength=len(RESTARTS) + 1).tolist()}, {"removed": removed})


def run_job(data: Data, d: dict, job: str, log=print) -> tuple[dict, dict | None]:
    import attn_learn as A
    if job == "refs":
        return {"rows": refs_rows(A.references(d, data.clf, data.device), d["k"])}, None
    if job in ATTN:
        return run_attn(data, d, job, log=log)
    if job == "nan_soft":
        return run_nan_soft(data, d, log=log)
    if job in RESTARTS:
        return run_nan_topk_restart(data, d, RESTARTS.index(job), log=log)
    raise ValueError(f"unknown job {job!r}")


def save(out: Path, res: dict, arrays: dict | None) -> None:
    if arrays is not None:
        np.savez_compressed(out.with_suffix(".npz"), **arrays)
    write_json(out, res)


def summary(res: dict) -> str:
    return "  ".join(f"{r['name']}: acc {100 * r['test']['acc']:.1f} auc {100 * r['test']['auc']:.1f} ce {r['test']['ce']:.3f}"
                     + (f" prec {r['prec']:.3f}" if r.get("prec") is not None else "")
                     for r in res.get("rows", []) if not r["name"].endswith("_nov"))


def try_vote(data: Data, d: dict, tag: str, device: str, log=print) -> None:
    out = ROOT / "runs" / tag
    target = out / "nan_topk.json"
    if target.exists() or not all((out / f"{r}.json").exists() for r in RESTARTS):
        return
    lock = target.with_suffix(".lock")
    if not claim(lock):
        return
    try:
        t0 = time.time()
        res, arrays = run_vote(data, d, tag)
        res = {"method": "nan_topk", "instance": tag, "device": device, "host": socket.gethostname(),
               "seconds": round(time.time() - t0, 1)} | res
        save(target, res, arrays)
        log(f"vote {tag}: done  {summary(res)}")
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------- stages

def grid(a) -> list[tuple[str, float, int, int]]:
    return [(ds, R, V, s) for ds in a.datasets for V in a.Vs for R in a.Rs for s in a.seeds]


def read_job_list(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tag, job = line.split()
            out.append((tag, job))
    return out


def jobs_of(a) -> list[tuple[str, str]]:
    if a.job_list:
        return read_job_list(a.job_list)
    todo = [(tag_of(*inst), job) for inst in grid(a) for job in (a.methods or JOBS)]
    if a.shard:
        i, m = (int(x) for x in a.shard.split("/"))
        todo = todo[i::m]
    return todo


def stage_draws(a) -> None:
    (ROOT / "draws").mkdir(parents=True, exist_ok=True)
    made, checked = 0, 0
    for ds, R, V, s in grid(a):
        path = ROOT / "draws" / f"{tag_of(ds, R, V, s)}.npz"
        if path.exists():
            continue
        z = build_draw(ds, R, V, s, a.device)
        np.savez_compressed(path, **z)
        made += 1
        checked += bool(str(z["checked_against"]))
        if made % 30 == 0:
            print(f"  {made} draws ({checked} checked against results/gate_valsplit)", flush=True)
    print(f"{made} new draws in {ROOT / 'draws'} ({checked} of them checked against the soft gate's saved draws)")


def stage_import_attn(a) -> None:
    """The attention study's runs whose draw is this benchmark's (same function; checked on the poison)."""
    done = 0
    for f in sorted(CONFIRM.glob("*/references.json")):
        ds, rest = f.parent.name.rsplit("_R", 1)
        R, vs = rest.split("_V")
        V, s = (int(x) for x in vs.split("s"))
        tag = tag_of(ds, float(R), V, s)
        draw = ROOT / "draws" / f"{tag}.npz"
        if ds not in a.datasets or s not in a.seeds or not draw.exists():
            continue
        z = np.load(draw)
        pois = np.zeros(len(z["ctx_rows"]), dtype=bool)
        pois[z["pois_rows"]] = True
        k = int(z["k"])
        out = ROOT / "runs" / tag
        out.mkdir(parents=True, exist_ok=True)
        prov = {"provenance": str(f.parent), "imported": time.strftime("%Y-%m-%d")}
        if not (out / "refs.json").exists():
            write_json(out / "refs.json", {"method": "refs", "instance": tag} | prov
                       | {"rows": refs_rows(json.loads(f.read_text()), k)})
            done += 1
        for job, name in CONFIRM_FILES.items():
            src = f.parent / name
            if not src.exists() or (out / f"{job}.json").exists():
                continue
            arr = np.load(src.with_suffix(".npz"))
            assert np.array_equal(arr["pois"], pois), f"{src}: its poisoned rows differ from {draw}"
            res = json.loads(src.read_text())
            removed = removed_at(arr["keys"], arr["its"].tolist(), res["sel_hard"]["it"], k)
            save(out / f"{job}.json", {"method": job, "instance": tag, "seconds": res.get("seconds")} | prov
                 | {"rows": attn_rows(job, res, k), "evals": res["evals"], "config": res["cfg"]}, {"removed": removed})
            done += 1
    print(f"imported {done} results from {CONFIRM}")


def worker(a) -> None:
    torch.set_num_threads(a.threads)
    todo = jobs_of(a)
    missing = sorted({tag for tag, _ in todo if not (ROOT / "draws" / f"{tag}.npz").exists()})
    if missing:
        raise SystemExit(f"{len(missing)} draws missing under {ROOT / 'draws'} (e.g. {missing[0]}): build them once "
                         "with the draws stage and copy the folder here")
    if any(job in ATTN for _, job in todo):  # they need the row-weight hooks of this repository's TabPFN copy
        import tabpfn
        try:
            from tabpfn.preprocessing.steps.context_row_weights import context_row_weights_installed  # noqa: F401
        except ImportError:
            print(f"SKIPPING the attention jobs: tabpfn is imported from {Path(tabpfn.__file__).parent}, which lacks "
                  "the row-weight hooks of this repository's TabPFN (TabPFN/src). Install that copy into this "
                  "environment (pip install -e TabPFN) and run again.", flush=True)
            todo = [j for j in todo if j[1] not in ATTN]
    data = None
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    for tag, job in todo:
        out = ROOT / "runs" / tag / f"{job}.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        lock = out.with_suffix(".lock")
        if not claim(lock):
            continue
        t0 = time.time()
        d = None
        try:
            ds = parse_tag(tag)[0]
            if data is None or data.ds != ds:
                data = None
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                data = Data(ds, a.device)
            d = load_draw(data, tag)
            log(f"run {tag} {job}: start (n {d['n']}, k {d['k']}, V {len(d['yv'])}: {len(d['yf'])} fitted, "
                f"{len(d['yh'])} held out)")
            res, arrays = run_job(data, d, job, log=log)
            res = {"method": job, "instance": tag, "device": a.device, "host": socket.gethostname(),
                   "seconds": round(time.time() - t0, 1)} | res
            save(out, res, arrays)
            log(f"run {tag} {job}: done in {time.time() - t0:.0f} s  {summary(res)}")
        except Exception as e:  # noqa: BLE001 - a failing job must not stop the worker
            log(f"run {tag} {job}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            lock.unlink(missing_ok=True)
        if job in RESTARTS and d is not None:
            try:
                try_vote(data, d, tag, a.device, log=log)
            except Exception as e:  # noqa: BLE001
                log(f"vote {tag}: FAILED {type(e).__name__}: {e}")
                traceback.print_exc()


def stage_votes(a) -> None:
    """Every top-k vote whose four restarts are done (the worker votes as soon as the last restart ends)."""
    torch.set_num_threads(a.threads)
    data = None
    for tag in sorted({tag for tag, _ in jobs_of(a)}):
        out = ROOT / "runs" / tag
        if (out / "nan_topk.json").exists() or not all((out / f"{r}.json").exists() for r in RESTARTS):
            continue
        ds = parse_tag(tag)[0]
        if data is None or data.ds != ds:
            data = Data(ds, a.device)
        try_vote(data, load_draw(data, tag), tag, a.device)


def done_of(tag: str, job: str) -> tuple[bool, bool]:
    """(finished, imported) for one instance and job."""
    f = ROOT / "runs" / tag / f"{job}.json"
    if not f.exists():
        return False, False
    return True, "provenance" in json.loads(f.read_text())


def stage_status(a) -> None:
    if a.job_list:
        todo = read_job_list(a.job_list)
        left = [(t, j) for t, j in todo if not done_of(t, j)[0]]
        votes = sorted({t for t, j in todo if j in RESTARTS and not (ROOT / "runs" / t / "nan_topk.json").exists()})
        print(f"{a.job_list}: {len(todo) - len(left)} of {len(todo)} jobs done; top-k votes missing for {len(votes)} instances")
        return
    jobs = list(JOBS) + ["nan_topk (vote)"]
    print(f"finished / imported, per job x dataset (of {len(a.Vs) * len(a.Rs) * len(a.seeds)} instances each)\n")
    print(f"  {'job':<16s}" + "".join(f"{ds[:13]:>15s}" for ds in a.datasets))
    for job in jobs:
        cells = []
        for ds in a.datasets:
            tags = [tag_of(ds, R, V, s) for V in a.Vs for R in a.Rs for s in a.seeds]
            states = [done_of(t, "nan_topk" if job.startswith("nan_topk (") else job) for t in tags]
            cells.append(f"{sum(f for f, _ in states):>5d} / {sum(i for _, i in states):<3d}")
        print(f"  {job:<16s}" + "".join(f"{c:>15s}" for c in cells))
    have = sum((ROOT / "draws" / f"{tag_of(*inst)}.npz").exists() for inst in grid(a))
    print(f"\ndraws: {have} of {len(grid(a))}; "
          f"locks held: {len(list((ROOT / 'runs').glob('*/*.lock')))}")


LAUNCHER = """#!/usr/bin/env bash
# Larger-V benchmark: GPU script {i:02d} of {N} -- {n_jobs} jobs, about {hours:.1f} H100-hours (estimate).
# Its longest job: {longest} (about {longest_min:.0f} H100-minutes). See scripts/bench_largev_jobs/README.md.
#
#   GPU=<device index> bash {out}/gpu_{i:02d}.sh
#
#   GPU      the CUDA device (default 0: the first one CUDA_VISIBLE_DEVICES exposes)
#   WORKERS  parallel workers on that GPU (default 2; gas-drift and coil2000 jobs take about 45 GB each)
#   PY       the python of the tabfm environment
# Needs results/bench_largev/draws/ (the shared draws). Finished jobs are skipped, so it can simply be started
# again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/{up}"
PY=${{PY:-{python}}}
GPU=${{GPU:-0}}
WORKERS=${{WORKERS:-2}}
LIST={out}/gpu_{i:02d}.txt
mkdir -p {logs}
export PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}
for w in $(seq 1 "$WORKERS"); do
  "$PY" -u scripts/bench_largev.py run --device "cuda:$GPU" --job-list "$LIST" --threads 4 \\
      > "{logs}/gpu_{i:02d}_w$w.log" 2>&1 &
  sleep 20
done
wait
# one worker alone retries whatever failed above (e.g. out of GPU memory), then any top-k vote left
"$PY" -u scripts/bench_largev.py run --device "cuda:$GPU" --job-list "$LIST" --threads 4 \\
    > "{logs}/gpu_{i:02d}_retry.log" 2>&1 || true
"$PY" -u scripts/bench_largev.py votes --device "cuda:$GPU" --job-list "$LIST" >> "{logs}/gpu_{i:02d}_retry.log" 2>&1 || true
"$PY" scripts/bench_largev.py status --job-list "$LIST"
grep -h "FAILED\\|SKIPPING" {logs}/gpu_{i:02d}_*.log || true
"""


def stage_plan(a) -> None:
    """Every unfinished job, packed longest first onto the least-loaded of N job lists; the four top-k
    restarts of an instance stay on one list (their vote needs all four)."""
    units = []  # (estimated H100-seconds, [(tag, job)])
    for ds, R, V, s in grid(a):
        tag = tag_of(ds, R, V, s)
        u = U[ds] * ((N_CTX[ds] - V) / N_CTX[ds]) ** 2
        left = [job for job in JOBS if not done_of(tag, job)[0]]
        for job in left:
            if job not in RESTARTS:
                units.append((UPDATES[job] * u, [(tag, job)]))
        rs = [job for job in left if job in RESTARTS]
        if rs:
            units.append((sum(UPDATES[r] for r in rs) * u, [(tag, r) for r in rs]))
    units.sort(key=lambda x: -x[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for unit in units:
        i = int(np.argmin(load))
        bins[i].append(unit)
        load[i] += unit[0]
    a.out.mkdir(parents=True, exist_ok=True)
    for old in list(a.out.glob("gpu_*.txt")) + list(a.out.glob("gpu_*.sh")):
        old.unlink()
    lines = ["| script | jobs | H100-hours (est.) | longest job | jobs per method |", "|---|---|---|---|---|"]
    for i, b in enumerate(bins):
        jobs = sorted(((sec / len(js), tag, job) for sec, js in b for tag, job in js), key=lambda t: (-t[0], t[1], t[2]))
        (a.out / f"gpu_{i:02d}.txt").write_text(
            f"# GPU script {i:02d}: {len(jobs)} jobs, about {load[i] / 3600:.1f} H100-hours; '<instance> <job>', longest first\n"
            + "".join(f"{tag} {job}\n" for _, tag, job in jobs))
        sh = a.out / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(jobs), hours=load[i] / 3600, longest=f"{jobs[0][1]} {jobs[0][2]}",
                                      longest_min=jobs[0][0] / 60, python=a.python, out=a.out.as_posix(),
                                      up="/".join([".."] * len(a.out.parts)), logs=f"logs/bench_largev/{a.out.name}"))
        os.chmod(sh, 0o755)
        per = {}
        for _, _, job in jobs:
            key = "nan_topk_r*" if job in RESTARTS else job
            per[key] = per.get(key, 0) + 1
        lines.append(f"| gpu_{i:02d}.sh | {len(jobs)} | {load[i] / 3600:.1f} | {jobs[0][1]} {jobs[0][2]} ({jobs[0][0] / 60:.0f} min) | "
                     + ", ".join(f"{k} {v}" for k, v in per.items()) + " |")
    total = sum(len(js) for _, js in units)
    lines += ["", f"{total} jobs, about {load.sum() / 3600:.0f} H100-hours in all, {load.max() / 3600:.1f} on the busiest script."]
    print("\n".join(lines))
    (a.out / "split.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "import-attn", "run", "votes", "status", "plan"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--Vs", nargs="+", type=int, default=list(VS))
    ap.add_argument("--Rs", nargs="+", type=float, default=list(RS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--methods", nargs="+", choices=JOBS, default=None)
    ap.add_argument("--job-list", type=Path, default=None, help="'<instance> <job>' lines (plan's gpu_XX.txt)")
    ap.add_argument("--shard", default=None, help="i/N: every N-th job of the grid, from the i-th")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scripts", type=int, default=8, help="plan: the number of job lists")
    ap.add_argument("--out", type=Path, default=Path("scripts/bench_largev_jobs"), help="plan: relative to the repository root")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    {"draws": stage_draws, "import-attn": stage_import_attn, "run": worker, "votes": stage_votes,
     "status": stage_status, "plan": stage_plan}[a.stage](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Which form of the soft attention-weight learner is best: parameterisation x start x update rule.

Soft throughout: the weights are learned soft and deployed soft (every poisoned row at its learned weight, all of
V added at weight 1; nothing is removed, no R% budget). Draws: the larger-V benchmark's (results/bench_largev/draws:
V = 500 clean rows, 80% fitted, 20% held out; R = 30% hub-flip-local; seed 0) on its 6 datasets.

Factors (18 configurations, plus log w with floor -2000):
  param    w        w itself in [0, 1], projected (w = 0 deletes the row exactly)
           sigmoid  w = sigmoid(t), log w = logsigmoid(t), t in [-1000, 20]
           log      log w in [-1000, 0]
           log2000  log w in [-2000, 0]: identical to log unless a run reaches -1000, so it is run only where the
                    log run did (otherwise the log run is its result)
  init     ones     w = 1 (sigmoid: t = 4, w = 0.982; a sigmoid at 1 has no gradient)
           vfilter  w = p(given label | the fitted V rows) / its max over the rows, the same w in every param
                    (log w = the attention study's V-filter start at temperature 1; sigmoid: t = logit w, at most 4)
  opt      adam     Adam (betas 0.9, 0.999), projected after every step
           pgd      lr * g / max|g| (attn_learn's pgd-linf: the largest move is lr), projected
           sign     lr * sign(g) (L-inf steepest descent, the textbook L-inf PGD), projected
Each update: the CE of the 400 fitted V rows with the weighted poisoned rows as the context, the gradient by autograd
through log w (attn_learn's Param / Stepper / gradient path). The held-out rows (100) are scored with the weighted
poisoned rows after every update up to 10, then every 5; a run stops once the held-out log loss has not improved by
1e-3 for 30 updates (at most 200), and the update with the lowest held-out log loss is deployed (ties: the earlier).
Test: soft + V (the headline) and soft; the update with the highest held-out accuracy is also scored.

Learning rate, one per configuration: a grid of 3 (x4 apart, LR_GRID) on the 4 cheap datasets (heloc,
students_dropout, ecommerce_shipping, is_this_a_good_customer); the rate with the best mean rank of the held-out log
loss is chosen (an edge winner extends the grid by x4, at most twice per side) and then run on gas-drift and coil2000.
No test number is used to choose anything.

    python scripts/attn_form_study.py work --device cuda:2 --worker 0     # workers share the queue (lock files)
    python scripts/attn_form_study.py status
    python scripts/attn_form_study.py smoke --device cuda:2               # 3 updates per configuration, own root
--only 'log*/*/*,sigmoid/*/sign' restricts a worker (and status) to those configurations (param/init/update, shell
patterns), so that machines without a shared disk can split the study (scripts/attn_form_jobs/README.md).
"""
from __future__ import annotations

import argparse
import fnmatch
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

ROOT = Path(os.environ.get("ATTN_FORM_ROOT", "results/attn_form"))
SMALL = ("heloc", "students_dropout", "ecommerce_shipping", "is_this_a_good_customer")
BIG = ("gas-drift", "coil2000_insurance_policies")
R, V, SEED = 30.0, 500, 0
PARAMS = ("w", "sigmoid", "log", "log2000")
INITS = ("ones", "vfilter")
OPTS = ("adam", "pgd", "sign")
FLOOR = {"w": 0.0, "sigmoid": -1000.0, "log": -1000.0, "log2000": -2000.0,
         "sigmoid_free": -math.inf}  # sigmoid_free: sigmoid with t unbounded (scripts/sigmoid_study.py)
T0, T_MAX = 4.0, 20.0
LR_GRID = {("w", "adam"): (0.0025, 0.01, 0.04), ("w", "pgd"): (0.0125, 0.05, 0.2), ("w", "sign"): (0.0025, 0.01, 0.04),
           ("sigmoid", "adam"): (0.05, 0.2, 0.8), ("sigmoid", "pgd"): (1.0, 4.0, 16.0),
           ("sigmoid", "sign"): (0.05, 0.2, 0.8),
           ("log", "adam"): (0.025, 0.1, 0.4), ("log", "pgd"): (1.0, 4.0, 16.0), ("log", "sign"): (0.025, 0.1, 0.4)}
MAX_EXT = 2  # grid extensions per side
STOP = dict(max_steps=200, dense=10, every=5, patience=30, tol=1e-3)
BIG_SLOTS = 1  # gas-drift / coil2000 jobs at once per GPU (37 / 26 GiB each); the GPU is time-shared anyway


ONLY: list[str] | None = None  # --only: the configurations this process works on (shell patterns on param/init/opt)


def configs() -> list[tuple[str, str, str]]:
    cs = [(p, i, o) for p in PARAMS for i in INITS for o in OPTS]
    if ONLY:
        cs = [c for c in cs if any(fnmatch.fnmatchcase("/".join(c), pat) for pat in ONLY)]
    return cs


def grid(param: str, opt: str) -> tuple[float, ...]:
    return LR_GRID[("log" if param == "log2000" else param, opt)]


def job_name(param: str, init: str, opt: str, lr: float) -> str:
    return f"{param}_{init}_{opt}_lr{lr:g}"


def tag_of(ds: str) -> str:
    return f"{ds}_R{R:g}_V{V}_s{SEED}"


def result_path(ds: str, name: str) -> Path:
    return ROOT / "runs" / tag_of(ds) / f"{name}.json"


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def claim(lock: Path) -> bool:
    """Atomically take a lock; one left by a dead process on this host is taken over."""
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


# ---------------------------------------------------------------- results and the learning-rate choice

_cache: dict[Path, dict] = {}


def load(ds: str, param: str, init: str, opt: str, lr: float) -> dict | None:
    """A finished run (or a failed one: {'failed': ...}); log2000 is the log run unless that one reached -1000."""
    if param == "log2000":
        base = load(ds, "log", init, opt, lr)
        if base is None or base.get("failed") or not base["floor_hit"]:
            return base
    path = result_path(ds, job_name(param, init, opt, lr))
    if path in _cache:
        return _cache[path]
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    _cache[path] = r
    return r


def hold_ce(r: dict | None) -> float:
    return math.inf if r is None or r.get("failed") else r["sel"]["ce"]["hold"]["ce"]


def lr_state(param: str, init: str, opt: str) -> tuple[str, list[float], float | None]:
    """('pending', the rates still needed on the small datasets, None) or ('done', the rates tried, the chosen one):
    the best mean rank of the held-out log loss over the small datasets (ties: the lower mean held-out log loss
    relative to the start); an edge winner extends the grid by x4, at most MAX_EXT times per side."""
    lrs = list(grid(param, opt))
    lo = hi = 0
    while True:
        res = {lr: [load(ds, param, init, opt, lr) for ds in SMALL] for lr in lrs}
        if any(r is None for rs in res.values() for r in rs):
            return "pending", lrs, None
        ranks = {lr: 0.0 for lr in lrs}
        rel = {lr: 0.0 for lr in lrs}
        for i, ds in enumerate(SMALL):
            ces = np.array([hold_ce(res[lr][i]) for lr in lrs])
            order = ces.argsort(kind="stable")
            rk = np.empty(len(lrs))
            rk[order] = np.arange(1, len(lrs) + 1)
            for v in np.unique(ces):  # ties share their mean rank
                rk[ces == v] = rk[ces == v].mean()
            for j, lr in enumerate(lrs):
                ranks[lr] += rk[j] / len(SMALL)
                r0 = res[lr][i]
                rel[lr] += (ces[j] / max(r0["evals"][0]["hold_ce"], 1e-12) if np.isfinite(ces[j]) else 1e9) / len(SMALL)
        best = min(lrs, key=lambda lr: (ranks[lr], rel[lr]))
        if best == lrs[0] and lo < MAX_EXT:
            lrs.insert(0, lrs[0] / 4)
            lo += 1
            continue
        if best == lrs[-1] and hi < MAX_EXT:
            lrs.append(lrs[-1] * 4)
            hi += 1
            continue
        return "done", lrs, best


def all_jobs() -> tuple[list[dict], bool]:
    """Every job that can run now (not finished), in priority order (Adam and PGD before sign; within each,
    gas-drift / coil2000 first), and whether anything is still waiting on a learning-rate choice."""
    tiers, waiting = {False: ([], []), True: ([], [])}, False
    for param, init, opt in configs():
        big, small = tiers[opt == "sign"]
        state, lrs, best = lr_state(param, init, opt)
        if state == "pending":
            waiting = True
        for lr in lrs:
            for ds in SMALL:
                small.append(dict(ds=ds, param=param, init=init, opt=opt, lr=lr))
        if state == "done":
            for ds in BIG:
                big.append(dict(ds=ds, param=param, init=init, opt=opt, lr=best))
    ready, seen = [], set()
    for j in [j for big, small in tiers.values() for j in big + small]:
        if j["param"] == "log2000":  # needs the log run first: queue that one (it may lie outside the log grid)
            base = load(j["ds"], "log", j["init"], j["opt"], j["lr"])
            if base is None:
                waiting = True
                j = j | {"param": "log"}
            elif base.get("failed") or not base["floor_hit"]:
                continue
        key = tuple(j.values())
        if key not in seen and load(j["ds"], j["param"], j["init"], j["opt"], j["lr"]) is None:
            seen.add(key)
            ready.append(j)
    return ready, waiting


# ---------------------------------------------------------------- one run

def init_raw(param: str, init: str, d: dict, clf, device) -> torch.Tensor:
    """The starting raw parameter: the same w in every parameterisation (sigmoid capped at sigmoid(T0)). Sigmoid only:
    vf_raw (sigmoid(t) = p(given label | V), no division by its max; p kept in [1e-12, 1 - 1e-6]), vf_hard (the k rows
    the V-filter finds least plausible at t = -30, the rest at T0) and random<seed> (t ~ N(0, 1))."""
    from tabfm_experiments.row_weights import evaluate_weighted
    n = d["n"]
    sig = param in ("sigmoid", "sigmoid_free")
    if init == "ones":
        return torch.full((n,), T0 if sig else {"w": 1.0}.get(param, 0.0), device=device)
    if init.startswith("random"):
        if not sig:
            raise ValueError("the random start is for the sigmoid forms")
        g = torch.Generator().manual_seed(int(init[len("random"):]))
        return torch.randn(n, generator=g).to(device)
    p = evaluate_weighted(clf, d["Xf"], d["yf"], d["Xc"], d["yp"], torch.zeros(len(d["yf"]), device=device),
                          need_grad=False)[2]
    s = torch.as_tensor(np.log(np.clip(p[np.arange(n), d["yp"]], 1e-12, None)), dtype=torch.float64, device=device)
    b = (s - s.max()).clamp(-1000.0, 0.0)  # log w at the start
    if init == "vf_raw":  # sigmoid only: sigmoid(t) = p(given label | V) itself, not divided by its max
        if not sig:
            raise ValueError("the raw V-filter start is for the sigmoid forms")
        q = torch.as_tensor(p[np.arange(n), d["yp"]], dtype=torch.float64, device=device).clamp(1e-12, 1 - 1e-6)
        return (torch.log(q) - torch.log1p(-q)).float()
    if init == "vf_hard":
        if not sig:
            raise ValueError("the hard V-filter start is for the sigmoid forms")
        raw = torch.full((n,), T0, dtype=torch.float64, device=device)
        raw[torch.as_tensor(np.argsort(s.cpu().numpy(), kind="stable")[:d["k"]], device=device)] = -30.0
        return raw.float()
    if param == "w":
        raw = b.exp()
    elif sig:  # t = logit(e^b) = b - log(1 - e^b)
        raw = (b - torch.log(-torch.expm1(b.clamp(max=-1e-12)))).clamp(max=T0)
    else:
        raw = b
    return raw.float()


def train(job: dict, data, d: dict, stop: dict, log=print, track_test: bool = False) -> tuple[dict, dict]:
    """track_test: also score the test set at every evaluation (never used to choose; for curves)."""
    import attn_learn as A
    from sklearn.metrics import roc_auc_score
    from tabfm_experiments.row_weights import evaluate_weighted
    clf, dev = data.clf, data.device
    kind = {"log2000": "log", "sigmoid_free": "sigmoid"}.get(job["param"], job["param"])
    floor = FLOOR[job["param"]]
    c = A.DEFAULTS | dict(param=kind, opt={"pgd": "pgd-linf"}.get(job["opt"], job["opt"]), lr=job["lr"], floor=floor,
                          t_max=math.inf if job["param"] == "sigmoid_free" else T_MAX, t0=T0)
    p = A.Param(kind, init_raw(job["param"], job["init"], d, clf, dev), c, d["k"])
    p.project()
    st = A.Stepper(p, c)
    pois = d["pois"]
    pois_t = torch.as_tensor(pois, device=dev)
    zeros_v = torch.zeros(len(d["yv"]), device=dev)
    Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([d["yp"], d["yv"]])
    floor_hit = False

    def test(lw: torch.Tensor) -> dict:
        soft_v = A.metrics(evaluate_weighted(clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw, zeros_v]),
                                             need_grad=False)[2], d["yt"])
        soft = A.metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xt"], d["yt"], lw, need_grad=False)[2], d["yt"])
        return {"soft_v": soft_v, "soft": soft}

    def weight_stats(lw: torch.Tensor) -> dict:
        lwc = lw.clamp(min=-1e4)
        key = p.raw.detach().cpu().numpy()
        out = {"auroc": float(roc_auc_score(pois, -key)) if np.ptp(key) > 0 else 0.5,
               "logw_med_pois": float(lwc[pois_t].median()), "logw_med_clean": float(lwc[~pois_t].median()),
               "w_share_pois": float(lw.exp()[pois_t].sum() / lw.exp().sum().clamp_min(1e-30))}
        for th in (10, 100, 1000):  # rows at or below log w = -th: poisoned / clean
            out[f"n_le{th}_pois"] = int((lw[pois_t] <= -th).sum())
            out[f"n_le{th}_clean"] = int((lw[~pois_t] <= -th).sum())
        return out

    evals, snaps, raws, t0 = [], {}, {}, time.time()
    best_stop, it_best_stop, stopped = math.inf, 0, "max_steps"
    for it in range(stop["max_steps"] + 1):
        if it < stop["dense"] or it % stop["every"] == 0 or it == stop["max_steps"]:
            lw = p.log_w().detach()
            hold = A.metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xh"], d["yh"], lw, need_grad=False)[2], d["yh"])
            e = {"it": it, "hold_ce": hold["ce"], "hold_acc": hold["acc"], "hold_auc": hold["auc"],
                 "sec": round(time.time() - t0, 1)} | weight_stats(lw)
            if track_test:
                e["test"] = test(lw)
            evals.append(e)
            snaps[it] = lw.clone()
            raws[it] = p.raw.detach().clone()
            if it % 50 == 0:
                log(f"    it {it:3d}  hold ce {hold['ce']:.4f} acc {hold['acc']:.3f} | auroc {e['auroc']:.3f}  log w med "
                    f"pois {e['logw_med_pois']:.1f} clean {e['logw_med_clean']:.1f}  ({e['sec']:.0f} s)")
            if not math.isfinite(hold["ce"]) or not torch.isfinite(p.raw).all():
                stopped = "nonfinite"
                break
            if hold["ce"] < best_stop - stop["tol"]:
                best_stop, it_best_stop = hold["ce"], it
            elif it - it_best_stop >= stop["patience"]:
                stopped = "patience"
                break
        if it == stop["max_steps"]:
            break
        lw = p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        train_ce, g, _ = evaluate_weighted(clf, d["Xc"], d["yp"], d["Xf"], d["yf"], lw_d, need_grad=True)
        if evals and evals[-1]["it"] == it:  # the fitted rows' log loss at the weights just evaluated (the objective)
            evals[-1]["train_ce"] = float(train_ce)
        p.raw.grad = None
        lw.backward(g)
        st.step(torch.nan_to_num(p.raw.grad.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0))
        if kind in ("log", "sigmoid") and bool((p.raw.detach() <= floor + 1e-3).any()):
            floor_hit = True

    def pick(metric: str) -> int:
        ok = [e for e in evals if math.isfinite(e["hold_ce"])]
        if metric == "ce":
            return min(ok, key=lambda e: (e["hold_ce"], e["it"]))["it"]
        return max(ok, key=lambda e: (np.nan_to_num(e["hold_acc"], nan=-1.0), -e["hold_ce"], -e["it"]))["it"]

    sel, tests = {}, {}
    for metric in ("ce", "acc"):
        it = pick(metric)
        if it not in tests:
            tests[it] = test(snaps[it])
        e = next(x for x in evals if x["it"] == it)
        sel[metric] = {"it": it, "hold": {"ce": e["hold_ce"], "acc": e["hold_acc"], "auc": e["hold_auc"]},
                       "test": tests[it]} | {k: e[k] for k in e if k not in ("it", "hold_ce", "hold_acc", "hold_auc", "sec")}
    start = tests[0] if 0 in tests else test(snaps[0])
    res = {"job": job, "instance": tag_of(job["ds"]), "n": d["n"], "k": d["k"], "K": d["K"], "stopped": stopped,
           "last_it": evals[-1]["it"], "floor_hit": floor_hit, "seconds": round(time.time() - t0, 1), "sel": sel,
           "start": start, "evals": evals, "stop": stop, "host": socket.gethostname(), "device": dev}
    it_ce = sel["ce"]["it"]
    arrays = {"raw_sel": raws[it_ce].cpu().numpy().astype(np.float32), "raw_start": raws[0].cpu().numpy().astype(np.float32),
              "logw_sel": snaps[it_ce].clamp(min=-1e4).cpu().numpy().astype(np.float32),
              "logw_start": snaps[0].clamp(min=-1e4).cpu().numpy().astype(np.float32), "pois": pois}
    return res, arrays


# ---------------------------------------------------------------- workers

class Datasets:
    """Each dataset's data, draw and TabPFN on one device, loaded on first use and kept."""

    def __init__(self, device: str):
        self.device, self.items = device, {}

    def get(self, ds: str):
        if ds not in self.items:
            import bench_largev as BL
            data = BL.Data(ds, self.device)
            self.items[ds] = (data, BL.load_draw(data, tag_of(ds)))
        return self.items[ds]


def take_slot(device: str) -> Path | None:
    (ROOT / "slots").mkdir(parents=True, exist_ok=True)
    gpu = device.replace(":", "")
    for s in range(BIG_SLOTS):
        lock = ROOT / "slots" / f"{socket.gethostname()}_{gpu}_slot{s}.lock"
        if claim(lock):
            return lock
    return None


def run_one(job: dict, sets: Datasets, stop: dict, log=print) -> None:
    path = result_path(job["ds"], job_name(job["param"], job["init"], job["opt"], job["lr"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    data, d = sets.get(job["ds"])
    t0 = time.time()
    log(f"{tag_of(job['ds'])} {path.stem}: start (n {d['n']}, k {d['k']})")
    try:
        res, arrays = train(job, data, d, stop, log=log)
    except torch.OutOfMemoryError:  # not the job's fault: leave it for a later retry
        log(f"{tag_of(job['ds'])} {path.stem}: out of GPU memory, released for a retry")
        torch.cuda.empty_cache()
        time.sleep(60)
        return
    except Exception as e:  # noqa: BLE001 - recorded as failed so the queue moves on
        log(f"{tag_of(job['ds'])} {path.stem}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        write_json(path, {"job": job, "failed": f"{type(e).__name__}: {e}", "host": socket.gethostname()})
        torch.cuda.empty_cache()
        return
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write_json(path, res)
    s = res["sel"]["ce"]
    log(f"{tag_of(job['ds'])} {path.stem}: it {s['it']} of {res['last_it']} ({res['stopped']}) hold ce {s['hold']['ce']:.4f}"
        f" | test soft+V auc {s['test']['soft_v']['auc']:.4f} acc {s['test']['soft_v']['acc']:.4f} ce "
        f"{s['test']['soft_v']['ce']:.4f} | auroc {s['auroc']:.3f} floor {res['floor_hit']} | {time.time() - t0:.0f} s")


def work(device: str, worker: int, small_only: bool) -> None:
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} w{worker}] {m}", flush=True)  # noqa: E731
    torch.set_num_threads(4)
    sets = Datasets(device)
    while True:
        _cache.clear()
        ready, waiting = all_jobs()
        picked = None
        for j in ready:
            path = result_path(j["ds"], job_name(j["param"], j["init"], j["opt"], j["lr"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            slot = None
            if j["ds"] in BIG:
                if small_only:
                    continue
                slot = take_slot(device)
                if slot is None:
                    continue
            if claim(path.with_suffix(".lock")):
                picked = (j, path, slot)
                break
            if slot is not None:
                slot.unlink(missing_ok=True)
        if picked is None:
            if not ready and not waiting:
                log("nothing left: done")
                return
            time.sleep(20)
            continue
        j, path, slot = picked
        try:
            run_one(j, sets, STOP, log=log)
        finally:
            torch.cuda.empty_cache()  # a gas-drift run leaves ~37 GiB cached; the GPU is shared by 3 workers
            path.with_suffix(".lock").unlink(missing_ok=True)
            if slot is not None:
                slot.unlink(missing_ok=True)


def status() -> None:
    _cache.clear()
    ready, waiting = all_jobs()
    done = failed = 0
    for f in (ROOT / "runs").glob("*/*.json"):
        r = json.loads(f.read_text())
        failed += bool(r.get("failed"))
        done += not r.get("failed")
    running = len(list((ROOT / "runs").glob("*/*.lock")))
    print(f"finished {done}, failed {failed}, running {running}, ready to run {len(ready) - running}, "
          f"waiting on a learning-rate choice: {waiting}")
    for param, init, opt in configs():
        state, lrs, best = lr_state(param, init, opt)
        big = [load(ds, param, init, opt, best) is not None for ds in BIG] if best is not None else []
        print(f"  {param:8s} {init:8s} {opt:5s} grid {','.join(f'{x:g}' for x in lrs):28s} {state:8s} "
              f"lr {best if best is not None else '-'}  big {sum(big)}/{len(BIG) if big else 0}")


def smoke(device: str) -> None:
    """3 updates of every configuration on is_this_a_good_customer (the cheapest dataset), in a scratch root."""
    global ROOT
    ROOT = Path(os.environ.get("ATTN_FORM_SMOKE_ROOT", "/tmp/attn_form_smoke"))
    sets = Datasets(device)
    stop = dict(STOP, max_steps=3, dense=10)
    for param, init, opt in configs():
        run_one(dict(ds="is_this_a_good_customer", param=param, init=init, opt=opt, lr=grid(param, opt)[1]), sets, stop)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("work", "status", "smoke"))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--small-only", action="store_true", help="never take a gas-drift / coil2000 job")
    ap.add_argument("--only", default="", help="comma-separated shell patterns on param/init/opt, e.g. 'log*/*/*'")
    a = ap.parse_args(argv)
    global ONLY
    ONLY = [x.strip() for x in a.only.split(",") if x.strip()] or None
    if a.stage == "work":
        work(a.device, a.worker, a.small_only)
    elif a.stage == "status":
        status()
    else:
        smoke(a.device)


if __name__ == "__main__":
    main()

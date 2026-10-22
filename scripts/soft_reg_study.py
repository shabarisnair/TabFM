#!/usr/bin/env python
"""Can the best soft attention-weight form train long on ALL of V (no held-out part, no early stopping) and beat its
V-filter start, given the right learning rate or regularisation?

Instances (R = 30%; development draws, seed 1, so that choosing among the arms does not tune anything on the
benchmarks' own seed-0 instances):
  large_<ds>_R30_V{50,100}_s1   V = 50 / 100 clean rows (results/bench_largev/draws)
  tiny_<ds>_R30_c5_s1          V = 5 rows per class (bench_tinyv.build_draw, seed 1; ROOT/draws)
on 3 datasets chosen as the informative ones (R = 30% means over V = 50, 100 and 5 per class, ROC-AUC): heloc
(balanced binary, noisy labels), students_dropout (multi-class) and is_this_a_good_customer (imbalanced; the cheapest).
All three have room above the V-filter (oracle - V-filter 0.06-0.14) and a harmful poison (oracle - poisoned + V
0.12-0.16). Left out: gas-drift (the V-filter is within 0.013 of the oracle), ecommerce_shipping (the poison does no
harm: oracle = poisoned + V), coil2000 (room, but an update costs more than the other three together, and erratic).

Two forms, both started from the V-filter (w0 = p(given label | all of V) / its max) and trained by PGD-L-inf steps
(lr * g / max|g|, projected); the loss is the log loss of V's rows with the weighted poisoned rows as the context;
STEPS updates, no held-out part:
  w_    w in [0, 1] (attn_form_study's best form), lr 0.05 (its held-out choice 0.8, cut 16x for 300 updates). A row at
        w = 0 is deleted exactly and stays deleted (no gradient reaches it; with a pull toward the start this is kept
        explicitly, since the pull would otherwise revive it every update to a tiny weight)
  log_  log w in [-1000, 0] (the better of log w and sigmoid in attn_form_study; its V-filter / PGD configuration is
        the better one deployed soft + V), lr 0.0625 (its held-out choice 1.0, cut by the same 16x). Weights never
        reach exactly 0, so nothing is permanent
Arms (ARMS; in TIERS order of priority), each for both forms:
  <form>_lr<x>   no regularisation: the form's control
  ..._mb<f>      each update's loss on a class-stratified random f of V (mini-batch; at least one row per class)
  ..._l1x<t>     proximal L1 toward the start: after each step the deviation w - w0 is soft-thresholded by t times the
                 median push of that step (lr * |g| / max|g| over the rows still above 0), so only rows pushed harder than
                 t x the typical row move (sparse changes; scale-free, since that spread differs by 10x between instances)
  ..._l2_<g>     proximal L2 toward the start: after each step the deviation shrinks by the factor 1 / (1 + g); g is set
                 so that the hardest-pushed row can still reach the form's span (equilibrium deviation lr / g): 1 in w
                 (g = 0.05), 20 nats in log w (g = 0.003)
                 (for both: a row at w = 0 stays deleted, as without regularisation, since no gradient reaches it; a pull
                 would otherwise revive it every update to a tiny weight, where TabPFN is distorted)
  ..._knn<k>     the gradient (in log w) averaged over each row and its k nearest neighbours with the same given label
                 (features z-scored over the poisoned rows + V): neighbourhoods move together, far fewer effective
                 degrees of freedom than one free weight per row
  ..._pseudo<f>  pseudo-queries: per update also the log loss of 200 rows drawn from the f most plausible poisoned rows
                 (by the start score), each held out of the context for that forward, at weight 1 (more labels to fit)
At every checkpoint in CKPTS: test metrics deployed soft + V (all of V added at weight 1; the headline, as in the
benchmarks) and soft alone, the log loss of V, the rows at w = 0 and their precision, the weights' AUROC against the
poison, the mean |w - w0| and the rows moved by more than 0.1; the test probabilities (soft + V) and the weights are
saved for post-hoc analyses. refs, once per instance: V only, poisoned + V, oracle + V, the V-filter hard + V (its
k = R% lowest rows removed; the benchmarks' V-filter) and the start itself, soft + V and soft.

    python scripts/soft_reg_study.py draws                                   # once, here: the tiny-V seed-1 draws
    python scripts/soft_reg_study.py work --device cuda:0 --job-list scripts/soft_reg_jobs/gpu_00.txt
    python scripts/soft_reg_study.py plan [--scripts 3]                      # -> scripts/soft_reg_jobs/
    python scripts/soft_reg_study.py status
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
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("SOFT_REG_ROOT", "results/soft_reg"))
JOBS_DIR = Path("scripts/soft_reg_jobs")
DATASETS = ("heloc", "students_dropout", "is_this_a_good_customer")
SEED, R = 1, 30.0
LARGE_VS, TINY_C = (50, 100), 5
STEPS = 300
CKPTS = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300)
FORMS = {"w": dict(lr=0.05, l2=0.05), "log": dict(lr=0.0625, l2=0.003)}  # per form: learning rate, L2 strength
REGS = {"": {}, "_mb0.5": dict(mb=0.5), "_mb0.25": dict(mb=0.25), "_l1x5": dict(l1=5.0), "_l2": {"l2": None},
        "_knn10": dict(knn=10), "_pseudo0.3": dict(pseudo=0.3)}


def arm_name(form: str, reg: str) -> str:
    f = FORMS[form]
    return f"{form}_lr{f['lr']:g}" + (f"_l2_{f['l2']:g}" if reg == "_l2" else reg)


ARMS = {arm_name(form, reg): {"form": form, "lr": FORMS[form]["lr"]}
        | {k: (FORMS[form]["l2"] if k == "l2" else v) for k, v in cfg.items()}
        for form in FORMS for reg, cfg in REGS.items()}
TIERS = tuple(tuple(arm_name(form, reg) for reg in tier for form in FORMS)
              for tier in (("", "_mb0.5", "_l1x5", "_l2"), ("_knn10", "_mb0.25"), ("_pseudo0.3",)))
PSEUDO_BATCH = 200


def instances() -> list[str]:
    return ([f"large_{ds}_R{R:g}_V{V}_s{SEED}" for ds in DATASETS for V in LARGE_VS]
            + [f"tiny_{ds}_R{R:g}_c{TINY_C}_s{SEED}" for ds in DATASETS])


def split_tag(tag: str) -> tuple[str, str]:
    kind, rest = tag.split("_", 1)
    return kind, rest


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- instances

def load_instance(tag: str, device: str) -> dict:
    """The draw's tensors and TabPFN, in one form for both kinds."""
    kind, rest = split_tag(tag)
    if kind == "large":
        import bench_largev as BL
        ds = BL.parse_tag(rest)[0]
        data = BL.Data(ds, device)
        d = BL.load_draw(data, rest)
        inst = dict(clf=data.clf, K=d["K"], n=d["n"], k=d["k"], Xc=d["Xc"], yp=np.asarray(d["yp"]).astype(int), pois=d["pois"],
                    Xv=d["Xv"], yv=np.asarray(d["yv"]).astype(int), Xt=d["Xt"], yt=np.asarray(d["yt"]).astype(int))
    else:
        import bench_tinyv as B
        ds = B.parse_tag(rest)[0]
        bd = B.BenchData(ds, device)
        z = np.load(ROOT / "draws" / f"{rest}.npz")
        pois = np.zeros(len(z["ctx_rows"]), dtype=bool)
        pois[z["pois_rows"]] = True
        inst = dict(clf=bd.clf, K=bd.K, n=len(z["ctx_rows"]), k=int(z["k"]), Xc=bd.X[bd.idx(z["ctx_rows"])],
                    yp=z["y_given"].astype(int), pois=pois, Xv=bd.X[bd.idx(z["v_rows"])], yv=np.asarray(bd.y[z["v_rows"]]).astype(int),
                    Xt=bd.Xt, yt=np.asarray(bd.yt).astype(int))
    inst |= {"tag": tag, "kind": kind, "ds": ds, "device": device}
    return inst


def start_scores(inst: dict) -> np.ndarray:
    """log p(given label | all of V as the context) per poisoned row: the V-filter's score."""
    from tabfm_experiments.row_weights import evaluate_weighted
    p = evaluate_weighted(inst["clf"], inst["Xv"], inst["yv"], inst["Xc"], inst["yp"],
                          torch.zeros(len(inst["yv"]), device=inst["device"]), need_grad=False)[2]
    return np.log(np.clip(p[np.arange(inst["n"]), inst["yp"]], 1e-12, None))


def knn_index(inst: dict, k: int) -> torch.Tensor:
    """[n, k] indices of each poisoned row's k nearest neighbours with the same given label (z-scored features over the
    poisoned rows + V); a row of a class with fewer rows is padded with itself."""
    from sklearn.neighbors import NearestNeighbors
    A = torch.cat([inst["Xc"], inst["Xv"]]).double().cpu().numpy()
    mu, sd = A.mean(0), A.std(0)
    sd[sd < 1e-8] = 1.0
    Z = (inst["Xc"].double().cpu().numpy() - mu) / sd
    n, yp = inst["n"], inst["yp"]
    idx = np.tile(np.arange(n)[:, None], (1, k))
    for cl in np.unique(yp):
        rows = np.flatnonzero(yp == cl)
        kk = min(k, len(rows) - 1)
        if kk < 1:
            continue
        nb = NearestNeighbors(n_neighbors=kk + 1).fit(Z[rows]).kneighbors(Z[rows], return_distance=False)
        idx[rows, :kk] = rows[nb[:, 1:]]
    return torch.as_tensor(idx, dtype=torch.long, device=inst["device"])


# ---------------------------------------------------------------- refs and training

def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    import attn_learn as A
    return A.metrics(p, y)


def run_refs(inst: dict) -> dict:
    from tabfm_experiments.row_weights import evaluate_weighted
    clf, dev, n, k = inst["clf"], inst["device"], inst["n"], inst["k"]
    zv = torch.zeros(len(inst["yv"]), device=dev)
    Xcv, ycv = torch.cat([inst["Xc"], inst["Xv"]]), np.concatenate([inst["yp"], inst["yv"]])
    s = start_scores(inst)
    b = torch.as_tensor(np.clip(s - s.max(), -1000.0, 0.0), dtype=torch.float32, device=dev)

    def test(X, y, lw):
        return metrics(evaluate_weighted(clf, X, y, inst["Xt"], inst["yt"], lw, need_grad=False)[2], inst["yt"])

    def deleted(mask):
        return torch.where(torch.as_tensor(mask, device=dev), torch.full((n,), -math.inf, device=dev), torch.zeros(n, device=dev))

    vf = np.zeros(n, dtype=bool)
    vf[np.argsort(s, kind="stable")[:k]] = True
    rows = {"v_only": test(inst["Xv"], inst["yv"], zv),
            "poisoned_plus_v": test(Xcv, ycv, torch.cat([torch.zeros(n, device=dev), zv])),
            "oracle_plus_v": test(Xcv, ycv, torch.cat([deleted(inst["pois"]), zv])),
            "vfilter_hard_v": test(Xcv, ycv, torch.cat([deleted(vf), zv])),
            "start_soft_v": test(Xcv, ycv, torch.cat([b, zv])),
            "start_soft": test(inst["Xc"], inst["yp"], b)}
    return {"instance": inst["tag"], "rows": rows, "vfilter_prec": float(inst["pois"][vf].mean()), "n": n, "k": k,
            "n_v": int(len(inst["yv"])), "K": inst["K"]}


def train(inst: dict, arm: str, steps: int = STEPS, log=print) -> tuple[dict, dict]:
    import attn_learn as A
    import gate_context as G
    from sklearn.metrics import roc_auc_score
    from tabfm_experiments.row_weights import evaluate_weighted
    cfg = ARMS[arm]
    form = cfg["form"]
    clf, dev, n, K = inst["clf"], inst["device"], inst["n"], inst["K"]
    pois, yv = inst["pois"], inst["yv"]
    c = A.DEFAULTS | {"param": form, "opt": "pgd-linf", "lr": cfg["lr"], "floor": 0.0 if form == "w" else -1000.0,
                      "seed": SEED}
    s = start_scores(inst)
    b = np.clip(s - s.max(), -1000.0, 0.0)  # log w at the start (the same weights for both forms)
    raw0 = torch.as_tensor(np.exp(b) if form == "w" else b, dtype=torch.float32, device=dev)
    p = A.Param(form, raw0, c, inst["k"])
    p.project()
    w0 = p.raw.detach().clone()  # the start, in the form's own parameter
    w0_w = p.log_w().detach().exp()  # ... and as weights
    st = A.Stepper(p, c)
    seed = int(hashlib.md5(f"{inst['tag']}|{arm}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    nb = knn_index(inst, cfg["knn"]) if cfg.get("knn") else None
    pseudo = np.argsort(-s, kind="stable")[:int(round(cfg["pseudo"] * n))] if cfg.get("pseudo") else None
    zv = torch.zeros(len(yv), device=dev)
    Xcv, ycv = torch.cat([inst["Xc"], inst["Xv"]]), np.concatenate([inst["yp"], yv])
    pois_t = torch.as_tensor(pois, device=dev)
    curve, probs, ws, t0 = [], [], [], time.time()
    for it in range(steps + 1):
        if it in CKPTS or it == steps:
            lw = p.log_w().detach()
            zero = torch.isinf(lw)
            low = lw <= -10  # near-deletion (w <= 4.5e-5), w = 0 included
            pv = evaluate_weighted(clf, Xcv, ycv, inst["Xt"], inst["yt"], torch.cat([lw, zv]), need_grad=False)[2]
            ps = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xt"], inst["yt"], lw, need_grad=False)[2]
            ce_v = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xv"], yv, lw, need_grad=False)[0]
            key = p.raw.detach().cpu().numpy()
            dev_w = (lw.exp() - w0_w).abs()  # in w, for both forms
            e = {"it": it, "train_ce": ce_v, "test_v": metrics(pv, inst["yt"]), "test": metrics(ps, inst["yt"]),
                 "n_zero": int(zero.sum()), "prec_zero": float(pois_t[zero].float().mean()) if zero.any() else None,
                 "n_le10": int(low.sum()), "prec_le10": float(pois_t[low].float().mean()) if low.any() else None,
                 "auroc": float(roc_auc_score(pois, -key)) if np.ptp(key) > 0 else 0.5,
                 "dev_mean": float(dev_w.mean()), "n_moved": int((dev_w > 0.1).sum()), "sec": round(time.time() - t0, 1)}
            curve.append(e)
            probs.append(pv.astype(np.float32))
            ws.append(lw.clamp(min=-1e4).cpu().numpy().astype(np.float32))
            if it % 50 == 0:
                log(f"    {arm} it {it:3d}: test+V auc {e['test_v']['auc']:.4f} acc {e['test_v']['acc']:.4f} ce "
                    f"{e['test_v']['ce']:.4f} | V ce {ce_v:.4f} | w=0 {e['n_zero']}, log w<=-10 {e['n_le10']} | "
                    f"moved {e['n_moved']}  ({e['sec']:.0f} s)")
        if it == steps:
            break
        q = np.arange(len(yv))
        if cfg.get("mb"):
            q = np.asarray(G.draw_validation(yv, max(K, int(round(cfg["mb"] * len(yv)))), int(rng.integers(1 << 30))))
        lw = p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        _, g, _ = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xv"][torch.as_tensor(q, device=dev)], yv[q], lw_d,
                                    need_grad=True)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if pseudo is not None:  # the most plausible poisoned rows as extra queries, each out of the context for it
            qb = np.sort(rng.choice(pseudo, min(PSEUDO_BATCH, len(pseudo)), replace=False))
            qbt = torch.as_tensor(qb, device=dev)
            lw_p = lw_d.detach().clone()
            lw_p[qbt] = -math.inf
            lw_p.requires_grad_(True)
            _, g_p, _ = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xc"][qbt], inst["yp"][qb], lw_p, need_grad=True)
            g_p = torch.nan_to_num(g_p, nan=0.0, posinf=0.0, neginf=0.0)
            g_p[qbt] = 0.0
            g = g + g_p
        if nb is not None:  # a neighbourhood moves together
            g = (g + g[nb].sum(1)) / (1 + nb.shape[1])
        p.raw.grad = None
        lw.backward(g)
        g_raw = torch.nan_to_num(p.raw.grad.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        st.step(g_raw)
        if cfg.get("l1") or cfg.get("l2"):
            with torch.no_grad():  # proximal steps toward the start; in the w form, rows at w = 0 stay deleted
                zero = p.raw == 0 if form == "w" else torch.zeros_like(p.raw, dtype=torch.bool)
                if cfg.get("l1"):
                    alive = ~zero
                    push = cfg["lr"] * g_raw.abs() / g_raw.abs().max().clamp_min(1e-30)
                    mu = cfg["l1"] * float(push[alive].median()) if alive.any() else 0.0
                    dv = p.raw - w0
                    p.raw.copy_(w0 + dv.sign() * (dv.abs() - mu).clamp_min(0.0))
                if cfg.get("l2"):
                    p.raw.copy_(w0 + (p.raw - w0) / (1.0 + cfg["l2"]))
                p.raw[zero] = 0.0
            p.project()
    res = {"arm": arm, "instance": inst["tag"], "config": cfg | {"steps": steps, "ckpts": [e["it"] for e in curve]},
           "curve": curve, "seconds": round(time.time() - t0, 1), "host": socket.gethostname()}
    arrays = {"its": np.array([e["it"] for e in curve]), "probs_v": np.stack(probs), "log_w": np.stack(ws),
              "log_w0": np.clip(b, -1e4, 0).astype(np.float32), "pois": pois, "yt": inst["yt"]}
    return res, arrays


# ---------------------------------------------------------------- stages

def stage_draws(a) -> None:
    """The tiny-V seed-1 draws (bench_tinyv's construction), built once here; the job machines get them copied."""
    import bench_tinyv as B
    (ROOT / "draws").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        rest = B.tag_of(ds, R, TINY_C, SEED)
        path = ROOT / "draws" / f"{rest}.npz"
        if path.exists():
            continue
        bd = B.BenchData(ds, a.device)
        np.savez(path, **B.build_draw(bd, R, TINY_C, SEED))
        print("built", path, flush=True)


def read_job_list(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tag, job = line.split()
            assert job == "refs" or job in ARMS, f"unknown job {job!r}"
            out.append((tag, job))
    return out


def work(a) -> None:
    import bench_largev as BL
    torch.set_num_threads(4)
    todo = read_job_list(a.job_list) if a.job_list else [(t, j) for t in instances() for j in ("refs", *ARMS)]
    import tabpfn
    try:
        from tabpfn.preprocessing.steps.context_row_weights import context_row_weights_installed  # noqa: F401
    except ImportError:
        raise SystemExit(f"tabpfn from {Path(tabpfn.__file__).parent} lacks the row-weight hooks (use this repo's TabPFN/src)")
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} {a.device}] {m}", flush=True)  # noqa: E731
    inst = None
    for tag, job in todo:
        out = ROOT / "runs" / tag / f"{job}.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        lock = out.with_suffix(".lock")
        if not BL.claim(lock):
            continue
        t0 = time.time()
        try:
            if inst is None or inst["tag"] != tag:
                inst = None
                torch.cuda.empty_cache()
                inst = load_instance(tag, a.device)
            log(f"{tag} {job}: start (n {inst['n']}, k {inst['k']}, V {len(inst['yv'])})")
            if job == "refs":
                write_json(out, run_refs(inst))
            else:
                res, arrays = train(inst, job, a.steps, log)
                np.savez_compressed(out.with_suffix(".npz"), **arrays)
                write_json(out, res)
            log(f"{tag} {job}: done in {time.time() - t0:.0f} s")
        except torch.OutOfMemoryError:  # not the job's fault: left for the retry pass
            log(f"{tag} {job}: out of GPU memory, left for a retry")
            inst = None
            torch.cuda.empty_cache()
            time.sleep(30)
        except Exception as e:  # noqa: BLE001
            log(f"{tag} {job}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            lock.unlink(missing_ok=True)


def status(a) -> None:
    todo = read_job_list(a.job_list) if a.job_list else [(t, j) for t in instances() for j in ("refs", *ARMS)]
    done = sum((ROOT / "runs" / t / f"{j}.json").exists() for t, j in todo)
    print(f"{done} of {len(todo)} jobs done")
    for job in ("refs", *ARMS):
        tags = [t for t, j in todo if j == job]
        if tags:
            print(f"  {job:20s} {sum((ROOT / 'runs' / t / f'{job}.json').exists() for t in tags)}/{len(tags)}")


LAUNCHER = """#!/bin/bash
# Soft-form regularisation study (scripts/soft_reg_study.py), script {i:02d} of {N}: {n_jobs} jobs, about {hours:.1f} GPU-hours
# (H100), in priority order (the core arms first). Run from a clone of this repository with results/ copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/soft_reg/.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/soft_reg
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > logs/soft_reg/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/soft_reg/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/soft_reg_study.py work --device "cuda:$GPU" --job-list {out}/gpu_{i:02d}.txt >> logs/soft_reg/gpu_{i:02d}.log 2>&1
done
$PY scripts/soft_reg_study.py status --job-list {out}/gpu_{i:02d}.txt
grep -h "FAILED" logs/soft_reg/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""


def plan(a) -> None:
    """refs first, then the arms tier by tier; within a tier the jobs go longest first onto the least-loaded list."""
    import bench_largev as BL
    import bench_tinyv as B

    def per_update(tag):  # H100-seconds per update (bench_largev's cost model, context size squared)
        kind, rest = split_tag(tag)
        ds = rest.rsplit("_R", 1)[0]
        n = BL.N_CTX[ds] - (BL.parse_tag(rest)[2] if kind == "large" else 0)
        return BL.U[ds] * (n / BL.N_CTX[ds]) ** 2

    load = np.zeros(a.scripts)
    bins = [[] for _ in range(a.scripts)]
    tiers = [[(4 * per_update(t), t, "refs") for t in instances()]]
    for tier in TIERS:
        tiers.append([((a.steps + 3 * len(CKPTS)) * per_update(t) * (2 if ARMS[arm].get("pseudo") else 1), t, arm)
                      for arm in tier for t in instances()])
    for tier in tiers:
        for cost, tag, job in sorted(tier, key=lambda x: -x[0]):
            i = int(np.argmin(load))
            bins[i].append((tag, job))
            load[i] += cost
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(JOBS_DIR.glob("gpu_*.sh")) + list(JOBS_DIR.glob("gpu_*.txt")):
        old.unlink()
    lines = ["| script | jobs | GPU-hours (est., H100) |", "|---|---|---|"]
    for i, b in enumerate(bins):
        (JOBS_DIR / f"gpu_{i:02d}.txt").write_text("".join(f"{t} {j}\n" for t, j in b))
        sh = JOBS_DIR / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=JOBS_DIR.as_posix(),
                                      up="/".join([".."] * len(JOBS_DIR.parts))))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.1f} |")
    files = []
    for ds in DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
        files += [f"{BL.ROOT}/draws/{BL.tag_of(ds, R, V, SEED)}.npz" for V in LARGE_VS]
        files += [f"{ROOT}/draws/{B.tag_of(ds, R, TINY_C, SEED)}.npz"]

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    (JOBS_DIR / "inputs.sha256").write_text("".join(f"{sha(f)}  {f}\n" for f in files))
    lines += ["", f"{sum(len(b) for b in bins)} jobs ({len(instances())} instances x (refs + {len(ARMS)} arms)), about "
              f"{load.sum() / 3600:.0f} H100-hours, {load.max() / 3600:.1f} on the busiest script; {a.steps} updates per arm."]
    print("\n".join(lines))
    (JOBS_DIR / "split.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("draws", "work", "plan", "status"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--job-list", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=STEPS, help="updates per arm (testing: fewer)")
    ap.add_argument("--scripts", type=int, default=3)
    a = ap.parse_args(argv)
    {"draws": stage_draws, "work": work, "plan": plan, "status": status}[a.stage](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Regularisation for M1 (top-k attention weights) trained on ALL of V: can it beat its start, which is exactly the
V-filter (hard + V), if trained long with a regulariser? The M1 counterpart of scripts/soft_reg_study.py.

Instances (R = 30%; development draws, seed 1; ROOT/draws holds them all):
  large_<ds>_R30_V{50,100}_s1   V = 50 / 100 clean rows (bench_largev's construction)
  tiny_<ds>_R30_c5_s1          V = 5 rows per class (bench_tinyv's construction)
on the soft study's 3 datasets (heloc, students_dropout, is_this_a_good_customer; the same instances, copied),
gas-drift (6 classes, 12-22% each; 128 features, the most expensive) and ecommerce_shipping (41/59; its poison does no
harm, the V-filter's removals do); the last two's draws copied from bench_largev and the soft study.

M1: one free score per poisoned row, started from the V-filter score s = log p(given label | all of V); the forward
pass removes the k = R% lowest-scored rows; the gradient (the log loss of V with the poisoned rows as the context) is
taken at the relaxed point (removed rows at w = 0.05, kept at 0.95) and moves the scores by PGD-L-inf steps
(lr * g / max|g|), lr 0.2 (tinyv_learned.ATTN_CONFIGS["topk"]). A removed row keeps a gradient, so it can come back.
STEPS updates, no held-out part. Deployed hard + V (the k lowest removed, all of V added): update 0 is the V-filter.

Arms (ARMS; TIERS order of priority), all at lr 0.2:
  m1             no regularisation: the control
  m1_mb<f>       each update's loss on a class-stratified random f of V
  m1_l1x5        proximal L1 toward the start scores: deviations soft-thresholded by 5x the median push of the step
  m1_l2_0.01     proximal L2 toward the start scores: the deviation shrinks by 1 / 1.01 per update (the hardest-pushed
                 row can still move about 20 nats, as for soft_reg's log form)
  m1_knn10       the gradient averaged over each row and its 10 nearest same-label neighbours
  m1_pseudo0.3   pseudo-queries: per update also 200 of the 30% most plausible poisoned rows (start score) as queries,
                 each held out of the context for its forward, at weight 1
At every checkpoint in CKPTS: test metrics hard + V (headline) and hard alone, V's log loss at the relaxed point (the
objective) and with the hard context, the precision of the removed rows, their overlap with the V-filter's, the
scores' AUROC against the poison; the test probabilities (hard + V) and the removed sets are saved.

    python scripts/m1_reg_study.py draws --device cuda:0        # once, here
    python scripts/m1_reg_study.py work --device cuda:0 --job-list scripts/m1_reg_jobs/gpu_00.txt
    python scripts/m1_reg_study.py plan [--scripts 9]           # -> scripts/m1_reg_jobs/
    python scripts/m1_reg_study.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import soft_reg_study as SR  # noqa: E402

ROOT = Path(os.environ.get("M1_REG_ROOT", "results/m1_reg"))
JOBS_DIR = Path("scripts/m1_reg_jobs")
# dataset: (clean train context, held-out test set)
DATA = {"heloc": ("datasets/heloc/splits/selected/natural/context_5000.csv", "datasets/heloc/splits/test_attack_1000.csv"),
        "students_dropout": ("datasets/students_dropout/splits/context_3000.csv",
                             "datasets/students_dropout/splits/test_attack_500.csv"),
        "is_this_a_good_customer": ("datasets/is_this_a_good_customer/splits/train.csv",
                                    "datasets/is_this_a_good_customer/splits/test.csv"),
        "gas-drift": ("datasets/gas-drift/splits/context_5000.csv", "datasets/gas-drift/splits/test_attack_1000.csv"),
        "ecommerce_shipping": ("datasets/ecommerce_shipping/splits/selected/natural/context_5000.csv",
                               "datasets/ecommerce_shipping/splits/test_attack_1000.csv")}
SEED, R = 1, 30.0
LARGE_VS, TINY_C = (50, 100), 5
STEPS = 300
CKPTS = SR.CKPTS
LR, PROBE = 0.2, 0.05
ARMS = {"m1": {}, "m1_mb0.5": dict(mb=0.5), "m1_mb0.25": dict(mb=0.25), "m1_l1x5": dict(l1=5.0),
        "m1_l2_0.01": dict(l2=0.01), "m1_knn10": dict(knn=10), "m1_pseudo0.3": dict(pseudo=0.3)}
TIERS = (("m1", "m1_mb0.5", "m1_l1x5", "m1_l2_0.01"), ("m1_knn10", "m1_mb0.25"), ("m1_pseudo0.3",))
# H100-seconds per update at the benchmarks' context sizes (bench_largev.U)
COST = {"heloc": 1.07, "students_dropout": 0.79, "is_this_a_good_customer": 0.14, "gas-drift": 4.8, "ecommerce_shipping": 0.64}


def register() -> None:
    """Make the datasets known to the draw builders (attn_learn / bench_tinyv), in this process only."""
    import attn_learn as A
    import bench_tinyv as B
    for ds, (ctx, test) in DATA.items():
        A.CONTEXT.setdefault(ds, ctx)
        A.TEST.setdefault(ds, test)
        B.DATASETS.setdefault(ds, (ctx, test))


def instances() -> list[str]:
    return ([f"large_{ds}_R{R:g}_V{V}_s{SEED}" for ds in DATA for V in LARGE_VS]
            + [f"tiny_{ds}_R{R:g}_c{TINY_C}_s{SEED}" for ds in DATA])


def draw_path(tag: str) -> Path:
    return ROOT / "draws" / f"{tag}.npz"


def load_instance(tag: str, device: str) -> dict:
    """The draw (ROOT/draws, both kinds in bench_tinyv's form: v_rows, ctx_rows, y_given, pois_rows, k) as tensors."""
    import bench_tinyv as B
    register()
    kind, rest = SR.split_tag(tag)
    ds = rest.rsplit("_R", 1)[0]
    bd = B.BenchData(ds, device)
    z = np.load(draw_path(tag))
    pois = np.zeros(len(z["ctx_rows"]), dtype=bool)
    pois[z["pois_rows"]] = True
    return dict(clf=bd.clf, K=bd.K, n=len(z["ctx_rows"]), k=int(z["k"]), Xc=bd.X[bd.idx(z["ctx_rows"])],
                yp=z["y_given"].astype(int), pois=pois, Xv=bd.X[bd.idx(z["v_rows"])], yv=np.asarray(bd.y[z["v_rows"]]).astype(int),
                Xt=bd.Xt, yt=np.asarray(bd.yt).astype(int), tag=tag, kind=kind, ds=ds, device=device)


def train(inst: dict, arm: str, steps: int = STEPS, log=print) -> tuple[dict, dict]:
    import attn_learn as A
    import gate_context as G
    from sklearn.metrics import roc_auc_score
    from tabfm_experiments.row_weights import evaluate_weighted
    cfg = ARMS[arm]
    clf, dev, n, K, k = inst["clf"], inst["device"], inst["n"], inst["K"], inst["k"]
    pois, yv = inst["pois"], inst["yv"]
    c = A.DEFAULTS | {"param": "topk", "opt": "pgd-linf", "lr": LR, "probe": PROBE, "seed": SEED}
    s = SR.start_scores(inst)
    p = A.Param("topk", torch.as_tensor(s, dtype=torch.float32, device=dev), c, k)
    raw0 = p.raw.detach().clone()
    st = A.Stepper(p, c)
    removed0 = p.deleted().cpu().numpy()
    rng = np.random.default_rng(int(hashlib.md5(f"{inst['tag']}|{arm}".encode()).hexdigest()[:8], 16))
    nb = SR.knn_index(inst, cfg["knn"]) if cfg.get("knn") else None
    pseudo = np.argsort(-s, kind="stable")[:int(round(cfg["pseudo"] * n))] if cfg.get("pseudo") else None
    zv = torch.zeros(len(yv), device=dev)
    Xcv, ycv = torch.cat([inst["Xc"], inst["Xv"]]), np.concatenate([inst["yp"], yv])
    curve, probs, removed_sets, t0 = [], [], [], time.time()
    prev = removed0.copy()
    swaps = 0
    for it in range(steps + 1):
        if it in CKPTS or it == steps:
            removed = p.deleted().cpu().numpy()
            lw_h = torch.where(torch.as_tensor(removed, device=dev), torch.full((n,), -math.inf, device=dev),
                               torch.zeros(n, device=dev))
            pv = evaluate_weighted(clf, Xcv, ycv, inst["Xt"], inst["yt"], torch.cat([lw_h, zv]), need_grad=False)[2]
            ph = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xt"], inst["yt"], lw_h, need_grad=False)[2]
            ce_probe = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xv"], yv, p.probe_log_w(), need_grad=False)[0]
            ce_hard = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xv"], yv, lw_h, need_grad=False)[0]
            key = p.raw.detach().cpu().numpy()
            e = {"it": it, "train_ce": ce_probe, "train_ce_hard": ce_hard, "test_v": SR.metrics(pv, inst["yt"]),
                 "test": SR.metrics(ph, inst["yt"]), "prec": float(pois[removed].mean()),
                 "overlap_vfilter": float((removed & removed0).sum() / k), "swaps": swaps,
                 "auroc": float(roc_auc_score(pois, -key)) if np.ptp(key) > 0 else 0.5,
                 "dev_mean": float((p.raw.detach() - raw0).abs().mean()), "sec": round(time.time() - t0, 1)}
            curve.append(e)
            probs.append(pv.astype(np.float32))
            removed_sets.append(removed)
            if it % 50 == 0:
                log(f"    {arm} it {it:3d}: test+V auc {e['test_v']['auc']:.4f} acc {e['test_v']['acc']:.4f} ce "
                    f"{e['test_v']['ce']:.4f} | V ce {ce_probe:.4f} | prec {e['prec']:.3f} | shared with V-filter "
                    f"{e['overlap_vfilter']:.2f}  ({e['sec']:.0f} s)")
        if it == steps:
            break
        q = np.arange(len(yv))
        if cfg.get("mb"):
            q = np.asarray(G.draw_validation(yv, max(K, int(round(cfg["mb"] * len(yv)))), int(rng.integers(1 << 30))))
        lw_d = p.probe_log_w().detach().clone().requires_grad_(True)
        _, g, _ = evaluate_weighted(clf, inst["Xc"], inst["yp"], inst["Xv"][torch.as_tensor(q, device=dev)], yv[q], lw_d,
                                    need_grad=True)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if pseudo is not None:  # the most plausible poisoned rows as extra queries, each out of the context for it
            qb = np.sort(rng.choice(pseudo, min(SR.PSEUDO_BATCH, len(pseudo)), replace=False))
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
        st.step(g)  # straight-through: the scores move like log-weights
        if cfg.get("l1") or cfg.get("l2"):
            with torch.no_grad():  # proximal steps toward the start scores
                if cfg.get("l1"):
                    push = LR * g.abs() / g.abs().max().clamp_min(1e-30)
                    mu = cfg["l1"] * float(push.median())
                    dv = p.raw - raw0
                    p.raw.copy_(raw0 + dv.sign() * (dv.abs() - mu).clamp_min(0.0))
                if cfg.get("l2"):
                    p.raw.copy_(raw0 + (p.raw - raw0) / (1.0 + cfg["l2"]))
        now = p.deleted().cpu().numpy()
        swaps += int((prev & ~now).sum())
        prev = now
    res = {"arm": arm, "instance": inst["tag"], "config": cfg | {"lr": LR, "probe": PROBE, "steps": steps},
           "curve": curve, "seconds": round(time.time() - t0, 1), "host": socket.gethostname()}
    arrays = {"its": np.array([e["it"] for e in curve]), "probs_v": np.stack(probs), "removed": np.stack(removed_sets),
              "pois": pois, "yt": inst["yt"]}
    return res, arrays


# ---------------------------------------------------------------- stages

def stage_draws(a) -> None:
    """All instances' draws in ROOT/draws: the soft study's (copied, so the instances are the same), the rest built
    by bench_largev's / bench_tinyv's constructions."""
    import bench_largev as BL
    import bench_tinyv as B
    register()
    (ROOT / "draws").mkdir(parents=True, exist_ok=True)
    for tag in instances():
        out = draw_path(tag)
        if out.exists():
            continue
        kind, rest = SR.split_tag(tag)
        ds = rest.rsplit("_R", 1)[0]
        src = (BL.ROOT / "draws" / f"{rest}.npz") if kind == "large" else (SR.ROOT / "draws" / f"{rest}.npz")
        if src.exists():
            z = dict(np.load(src))
        elif kind == "large":
            z = BL.build_draw(ds, R, BL.parse_tag(rest)[2], SEED, a.device)
        else:
            z = B.build_draw(B.BenchData(ds, a.device), R, TINY_C, SEED)
        np.savez(out, v_rows=z["v_rows"] if "v_rows" in z else z["val_rows"], ctx_rows=z["ctx_rows"], y_given=z["y_given"],
                 pois_rows=z["pois_rows"], k=z["k"])
        print(("copied " if src.exists() else "built ") + str(out), flush=True)


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
                SR.write_json(out, SR.run_refs(inst))
            else:
                res, arrays = train(inst, job, a.steps, log)
                np.savez_compressed(out.with_suffix(".npz"), **arrays)
                SR.write_json(out, res)
            log(f"{tag} {job}: done in {time.time() - t0:.0f} s")
        except torch.OutOfMemoryError:
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
    print(f"{sum((ROOT / 'runs' / t / f'{j}.json').exists() for t, j in todo)} of {len(todo)} jobs done")
    for job in ("refs", *ARMS):
        tags = [t for t, j in todo if j == job]
        if tags:
            print(f"  {job:14s} {sum((ROOT / 'runs' / t / f'{job}.json').exists() for t in tags)}/{len(tags)}")


def plan(a) -> None:
    """All jobs go longest first onto the least-loaded list (gas-drift's few long jobs balance only this way); each list
    then runs in priority order: refs, then the arms tier by tier."""
    def per_update(tag):
        kind, rest = SR.split_tag(tag)
        ds = rest.rsplit("_R", 1)[0]
        n = len(np.load(draw_path(tag))["ctx_rows"])
        n_ref = {"heloc": 4500, "gas-drift": 4500, "ecommerce_shipping": 4500, "students_dropout": 2500,
                 "is_this_a_good_customer": 900}[ds]
        return COST[ds] * (n / n_ref) ** 2

    load = np.zeros(a.scripts)
    bins = [[] for _ in range(a.scripts)]
    jobs = [(0, 4 * per_update(t), t, "refs") for t in instances()]
    for ti, tier in enumerate(TIERS, 1):  # a checkpoint (4 passes without gradients) costs about one update
        jobs += [(ti, (a.steps + len(CKPTS)) * per_update(t) * (2 if ARMS[arm].get("pseudo") else 1), t, arm)
                 for arm in tier for t in instances()]
    for ti, cost, tag, job in sorted(jobs, key=lambda x: -x[1]):
        i = int(np.argmin(load))
        bins[i].append((ti, -cost, tag, job))
        load[i] += cost
    bins = [[(tag, job) for _, _, tag, job in sorted(b)] for b in bins]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(JOBS_DIR.glob("gpu_*.sh")) + list(JOBS_DIR.glob("gpu_*.txt")):
        old.unlink()
    launcher = (SR.LAUNCHER.replace("Soft-form regularisation study (scripts/soft_reg_study.py)",
                                    "M1 regularisation study (scripts/m1_reg_study.py)")
                .replace("scripts/soft_reg_study.py", "scripts/m1_reg_study.py").replace("logs/soft_reg", "logs/m1_reg"))
    lines = ["| script | jobs | GPU-hours (est., H100) |", "|---|---|---|"]
    for i, b in enumerate(bins):
        (JOBS_DIR / f"gpu_{i:02d}.txt").write_text("".join(f"{t} {j}\n" for t, j in b))
        sh = JOBS_DIR / f"gpu_{i:02d}.sh"
        sh.write_text(launcher.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=JOBS_DIR.as_posix(),
                                      up="/".join([".."] * len(JOBS_DIR.parts))))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.1f} |")
    files = []
    for ds, (ctx, test) in DATA.items():
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
    files += [str(draw_path(t)) for t in instances()]
    (JOBS_DIR / "inputs.sha256").write_text("".join(f"{SR_sha(f)}  {f}\n" for f in files))
    lines += ["", f"{sum(len(b) for b in bins)} jobs ({len(instances())} instances x (refs + {len(ARMS)} arms)), about "
              f"{load.sum() / 3600:.1f} H100-hours, {load.max() / 3600:.1f} on the busiest script; {a.steps} updates per arm."]
    print("\n".join(lines))
    (JOBS_DIR / "split.md").write_text("\n".join(lines) + "\n")


def SR_sha(path) -> str:  # noqa: N802
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("draws", "work", "plan", "status"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--job-list", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--scripts", type=int, default=9)
    a = ap.parse_args(argv)
    {"draws": stage_draws, "work": work, "plan": plan, "status": status}[a.stage](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Sigmoid soft attention weights with t unbounded, trained by Adam, at V = 50, 100 and 5 rows per class (R = 30%):
which start and learning rate, and how do they compare with the best soft form and the form study's sigmoid?

w = sigmoid(t) per poisoned row, t any real number, never clipped (attn_form_study's param "sigmoid_free"); log w =
logsigmoid(t) enters the attention, w the label mean and the z-norms. The loss is the log loss of the fitted 80% of V
with the weighted poisoned rows as the context (attn_form_study.train). Starts:
  ones      t = 4 for every row (w = 0.982)
  vf_soft   sigmoid(t) = the V-filter's soft weight p(given label | fitted V) itself (p kept in [1e-12, 1 - 1e-6])
  vf_hard   the k = R% rows the V-filter finds least plausible at t = -30, the rest at t = 4
  random    t ~ N(0, 1), seeds 0-4: `random` is seed 0; `random_ens` averages the five runs' weights sigmoid(t), each
            at its chosen update
Adam (betas 0.9, 0.999) at lr 0.01, 0.1, 5 and 10. The held-out 20% of V is scored after every update up to 10, then
every 5; a run stops 50 updates after its held-out log loss last improved by 1e-3 (at most 300), and the update with
the lowest held-out log loss is deployed: soft + V (the headline: every poisoned row at its weight, all of V at 1) and
soft (the poisoned rows alone), on the test set.

Instances (seed 0, the 6 datasets): the larger-V benchmark's draws at V = 50 and 100 (their 80/20 split of V) and the
tiny-V benchmark's at 5 rows per class (1 row per class held out: gate_context.split_holdout). Reference at 5 rows per
class: the best soft form (w / V-filter / PGD lr 0.8: bench_largev's attn_wsoft, the form study's stopping rule) with
the same holdout; at V = 50 and 100 the larger-V benchmark already has it.

    python scripts/sigmoid_study.py work --device cuda:0 --job-list scripts/sigmoid_jobs/gpu_00.txt
    python scripts/sigmoid_study.py plan [--scripts 9]           # -> scripts/sigmoid_jobs/
    python scripts/sigmoid_study.py status
    python scripts/sigmoid_study.py report                       # -> results/tables/sigmoid_study/report.md
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

ROOT = Path(os.environ.get("SIGMOID_STUDY_ROOT", "results/sigmoid_study"))
JOBS_DIR = Path("scripts/sigmoid_jobs")
DATASETS = ("gas-drift", "students_dropout", "heloc", "ecommerce_shipping", "is_this_a_good_customer",
            "coil2000_insurance_policies")
REGIMES = ("V50", "V100", "c5")
R, SEED, HOLDOUT = 30.0, 0, 0.2
INITS = ("ones", "vf_soft", "vf_hard", "random")
INIT_ARG = {"ones": "ones", "vf_soft": "vf_raw", "vf_hard": "vf_hard"}  # attn_form_study.init_raw's names
SEEDS = (0, 1, 2, 3, 4)
LRS = (0.01, 0.1, 5.0, 10.0)
STOP = dict(max_steps=300, dense=10, every=5, patience=50, tol=1e-3)
# expected updates per run (planning only: early stopping makes runs shorter than 300)
EXPECTED_UPDATES = 120


def tag_of(ds: str, regime: str) -> str:
    return f"{ds}_R{R:g}_{regime}_s{SEED}"


def instances() -> list[tuple[str, str]]:
    return [(ds, rg) for ds in DATASETS for rg in REGIMES]


def job_names(regime: str) -> list[str]:
    return [f"{init}_lr{lr:g}" for init in INITS for lr in LRS] + (["best_soft"] if regime == "c5" else [])


def load_instance(ds: str, regime: str, device: str):
    """(data with .clf, draw in bench_largev.load_draw's form)."""
    import bench_largev as BL
    if regime.startswith("V"):
        data = BL.Data(ds, device)
        return data, BL.load_draw(data, BL.tag_of(ds, R, int(regime[1:]), SEED))
    import bench_tinyv as B
    import gate_context as G
    bd = B.BenchData(ds, device)
    t = B.load_draw(bd, B.tag_of(ds, R, int(regime[1:]), SEED), SEED)
    fit, hold = G.split_holdout(np.asarray(t["yv"]), HOLDOUT, SEED)
    yv = np.asarray(t["yv"]).astype(int)
    d = dict(ds=ds, R=R, V=len(yv), seed=SEED, K=bd.K, n=t["n"], k=t["k"], Xc=t["Xc"], y_clean=t["y_true"],
             yp=np.asarray(t["y_given"]).astype(int), pois=t["pois"], Xf=t["Xv"][bd.idx(fit)], yf=yv[fit],
             Xh=t["Xv"][bd.idx(hold)], yh=yv[hold], Xv=t["Xv"], yv=yv, Xt=bd.Xt, yt=np.asarray(bd.yt).astype(int))
    return bd, d


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def save(out_dir: Path, name: str, res: dict, arrays: dict, tag: str) -> None:
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    write_json(out_dir / f"{name}.json", res | {"instance": tag, "run": name})


def ensemble(data, d: dict, raws: list[np.ndarray]) -> tuple[dict, dict]:
    """The random starts' weights sigmoid(t) averaged (each run at its chosen update), deployed like a run's chosen
    update."""
    import attn_learn as A
    from sklearn.metrics import roc_auc_score
    from tabfm_experiments.row_weights import evaluate_weighted, log_weights
    clf, dev = data.clf, data.device
    w = torch.sigmoid(torch.as_tensor(np.stack(raws), dtype=torch.float64)).mean(0)
    lw = log_weights(w).float().to(dev)  # a weight that underflows to 0 deletes its row
    zv = torch.zeros(len(d["yv"]), device=dev)
    Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([d["yp"], d["yv"]])
    hold = A.metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xh"], d["yh"], lw, need_grad=False)[2], d["yh"])
    test = {"soft_v": A.metrics(evaluate_weighted(clf, Xcv, ycv, d["Xt"], d["yt"], torch.cat([lw, zv]), need_grad=False)[2],
                                d["yt"]),
            "soft": A.metrics(evaluate_weighted(clf, d["Xc"], d["yp"], d["Xt"], d["yt"], lw, need_grad=False)[2], d["yt"])}
    res = {"sel": {"ce": {"it": None, "hold": {"ce": hold["ce"], "acc": hold["acc"], "auc": hold["auc"]}, "test": test,
                          "auroc": float(roc_auc_score(d["pois"], -w.numpy()))}},
           "n_runs": len(raws), "stopped": "ensemble"}
    return res, {"w_sel": w.numpy().astype(np.float32), "pois": d["pois"]}


def run_job(data, d: dict, tag: str, job: str, log=print) -> None:
    import attn_form_study as S
    import bench_largev as BL
    out_dir = ROOT / "runs" / tag
    if job == "best_soft":
        res, arrays = S.train({"ds": d["ds"]} | BL.WSOFT, data, d, S.STOP, log=log)
        save(out_dir, job, res | {"config": BL.WSOFT | {"stop": S.STOP}}, arrays, tag)
        return
    init, lr = job.rsplit("_lr", 1)
    lr = float(lr)
    spec = lambda init_arg: {"ds": d["ds"], "param": "sigmoid_free", "init": init_arg, "opt": "adam", "lr": lr}  # noqa: E731
    if init != "random":
        res, arrays = S.train(spec(INIT_ARG[init]), data, d, STOP, log=log)
        save(out_dir, job, res | {"config": spec(INIT_ARG[init]) | {"stop": STOP}}, arrays, tag)
        return
    raws = []
    for s in SEEDS:  # each seed's run is saved on its own, so an interrupted job resumes after the last finished seed
        name = f"random_s{s}_lr{lr:g}"
        if not (out_dir / f"{name}.json").exists():
            res, arrays = S.train(spec(f"random{s}"), data, d, STOP, log=log)
            save(out_dir, name, res | {"config": spec(f"random{s}") | {"stop": STOP}}, arrays, tag)
        raws.append(np.load(out_dir / f"{name}.npz")["raw_sel"])
    res, arrays = ensemble(data, d, raws)
    save(out_dir, f"random_ens_lr{lr:g}", res | {"config": {"seeds": list(SEEDS), "lr": lr,
                                                             "average": "sigmoid(t) at each run's chosen update"}},
         arrays, tag)


def done_marker(tag: str, job: str) -> Path:
    name = job if not job.startswith("random_") else "random_ens_lr" + job.rsplit("_lr", 1)[1]
    return ROOT / "runs" / tag / f"{name}.json"


def read_job_list(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tag, job = line.split()
            out.append((tag, job))
    return out


def all_jobs() -> list[tuple[str, str]]:
    return [(tag_of(ds, rg), job) for ds, rg in instances() for job in job_names(rg)]


def work(a) -> None:
    import bench_tinyv as B
    torch.set_num_threads(4)
    todo = read_job_list(a.job_list) if a.job_list else all_jobs()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} {a.device}] {m}", flush=True)  # noqa: E731
    loaded = (None, None, None)
    for tag, job in todo:
        if done_marker(tag, job).exists():
            continue
        out_dir = ROOT / "runs" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        lock = out_dir / f"{job}.lock"
        if not B.claim(lock):
            continue
        t0 = time.time()
        try:
            ds, rest = tag.split("_R", 1)
            regime = rest.split("_")[1]
            if loaded[0] != tag:
                loaded = (None, None, None)
                torch.cuda.empty_cache()
                loaded = (tag, *load_instance(ds, regime, a.device))
            log(f"{tag} {job}: start (n {loaded[2]['n']}, k {loaded[2]['k']}, V fitted {len(loaded[2]['yf'])}, held out "
                f"{len(loaded[2]['yh'])})")
            run_job(loaded[1], loaded[2], tag, job, log=log)
            log(f"{tag} {job}: done in {time.time() - t0:.0f} s")
        except torch.OutOfMemoryError:
            log(f"{tag} {job}: out of GPU memory, left for a retry")
            loaded = (None, None, None)
            torch.cuda.empty_cache()
            time.sleep(30)
        except Exception as e:  # noqa: BLE001 - a failing job must not stop the worker
            log(f"{tag} {job}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            lock.unlink(missing_ok=True)


def status(a) -> None:
    todo = read_job_list(a.job_list) if a.job_list else all_jobs()
    done = [j for j in todo if done_marker(*j).exists()]
    print(f"{len(done)} of {len(todo)} jobs done")
    for rg in REGIMES:
        mine = [j for j in todo if f"_{rg}_" in j[0]]
        if mine:
            print(f"  {rg:5s} {sum(done_marker(*j).exists() for j in mine)}/{len(mine)}")


# ---------------------------------------------------------------- planning

LAUNCHER = """#!/bin/bash
# Sigmoid soft attention weights (scripts/sigmoid_study.py), script {i:02d} of {N}: {n_jobs} jobs, about {hours:.1f} GPU-hours
# (H100). Run from a clone of this repository with results/ and datasets/ copied over:
#   bash {out}/gpu_{i:02d}.sh [GPU]        (PY=/path/to/python to use another python than conda env tabfm)
# It stops before running anything unless the inputs, the model checkpoint, the code and a numeric preflight match
# the main machine (scripts/attn_form_jobs/check_remote.py). Rerunning resumes; logs in logs/sigmoid_jobs/.
GPU=${{1:-0}}
cd "$(dirname "$0")/{up}" || exit 1
PY=${{PY:-"conda run -n tabfm --no-capture-output python"}}
export PYTHONPATH="$PWD/TabPFN/src${{PYTHONPATH:+:$PYTHONPATH}}"  # the repo's TabPFN, with the row-weight hooks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore
mkdir -p logs/sigmoid_jobs
echo "script {i:02d} on GPU $GPU: checking (about a minute)"
$PY scripts/attn_form_jobs/check_remote.py --device "cuda:$GPU" --manifest {out}/inputs.sha256 \\
  > logs/sigmoid_jobs/gpu_{i:02d}_check.log 2>&1
status=$?
grep -E "^  (inputs|model|numbers|PROBLEM|all checks|checks failed)" logs/sigmoid_jobs/gpu_{i:02d}_check.log
[ $status -eq 0 ] || exit 1
for pass in 1 2; do  # the second pass retries anything that failed (e.g. out of GPU memory)
  $PY -u scripts/sigmoid_study.py work --device "cuda:$GPU" --job-list {out}/gpu_{i:02d}.txt >> logs/sigmoid_jobs/gpu_{i:02d}.log 2>&1
done
$PY scripts/sigmoid_study.py status --job-list {out}/gpu_{i:02d}.txt
grep -h "FAILED" logs/sigmoid_jobs/gpu_{i:02d}.log || echo "script {i:02d}: no failures"
"""


def cost(tag: str, job: str) -> float:
    """H100-seconds: runs x expected updates x the benchmark's per-update cost (bench_largev.U), and the evaluations."""
    import bench_largev as BL
    ds = tag.split("_R", 1)[0]
    runs = len(SEEDS) if job.startswith("random_") else 1
    if job == "best_soft":
        return 60 * 1.2 * BL.U[ds] + 30
    return runs * EXPECTED_UPDATES * 1.2 * BL.U[ds] + 30


def plan(a) -> None:
    from plan_soft_m1 import sha256
    import bench_largev as BL
    import bench_tinyv as B
    units = sorted(((cost(t, j), t, j) for t, j in all_jobs() if not done_marker(t, j).exists()), key=lambda u: -u[0])
    bins, load = [[] for _ in range(a.scripts)], np.zeros(a.scripts)
    for u in units:
        i = int(np.argmin(load))
        bins[i].append(u)
        load[i] += u[0]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(JOBS_DIR.glob("gpu_*.sh")) + list(JOBS_DIR.glob("gpu_*.txt")):
        old.unlink()
    lines = ["| script | jobs | GPU-hours (est., H100) |", "|---|---|---|"]
    for i, b in enumerate(bins):
        b = sorted(b, key=lambda u: (u[1], u[2]))  # by instance, so each instance is loaded once
        (JOBS_DIR / f"gpu_{i:02d}.txt").write_text("".join(f"{t} {j}\n" for _, t, j in b))
        sh = JOBS_DIR / f"gpu_{i:02d}.sh"
        sh.write_text(LAUNCHER.format(i=i, N=a.scripts, n_jobs=len(b), hours=load[i] / 3600, out=JOBS_DIR.as_posix(),
                                      up="/".join([".."] * len(JOBS_DIR.parts))))
        os.chmod(sh, 0o755)
        lines.append(f"| gpu_{i:02d}.sh | {len(b)} | {load[i] / 3600:.1f} |")
    files = []
    for ds in DATASETS:
        ctx, test = B.DATASETS[ds]
        files += [ctx, test, f"datasets/{ds}/splits/split_info.json"]
        files += [f"{BL.ROOT}/draws/{BL.tag_of(ds, R, V, SEED)}.npz" for V in (50, 100)]
        files += [f"{B.ROOT}/draws/{B.tag_of(ds, R, 5, SEED)}.npz"]
    (JOBS_DIR / "inputs.sha256").write_text("".join(f"{sha256(Path(f))}  {f}\n" for f in files))
    lines += ["", f"{len(units)} jobs, about {load.sum() / 3600:.0f} H100-hours in all ({EXPECTED_UPDATES} updates per run "
              f"assumed: early stopping), {load.max() / 3600:.1f} on the busiest script; on a GPU that another busy job "
              "shares, two to four times that."]
    print("\n".join(lines))
    (JOBS_DIR / "split.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- report

METRICS = (("auc", "ROC-AUC"), ("acc", "accuracy"), ("ce", "log loss"))


def load_results() -> dict:
    out = {}
    for f in sorted((ROOT / "runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        out[(f.parent.name, f.stem)] = r
    return out


def references(ds: str, regime: str) -> dict:
    """name -> {"soft_v" or "plus_v": test metrics}: the benchmarks' references and best soft form on this instance."""
    import bench_largev as BL
    import bench_tinyv as B
    rows = {}
    if regime.startswith("V"):
        d = BL.ROOT / "runs" / BL.tag_of(ds, R, int(regime[1:]), SEED)
        files = {"refs": d / "refs.json", "attn_wsoft": d / "attn_wsoft.json"}
    else:
        d = B.ROOT / "runs" / B.tag_of(ds, R, 5, SEED)
        files = {"refs": d / "refs.json", "vfilter": d / "vfilter.json", "tiny_wsoft": d / "attn_wsoft_lr0.2_it4.json"}
    for f in files.values():
        if f.exists():
            for e in json.loads(f.read_text())["rows"]:
                rows[e["name"]] = e["test"]
    return rows


def report(a) -> None:
    res = load_results()
    out_md = Path("results/tables/sigmoid_study")
    out_md.mkdir(parents=True, exist_ok=True)
    md = ["# Sigmoid soft attention weights, t unbounded, Adam (R = 30%)", "",
          "Mean over the datasets with the run finished; test set, deployed at the update with the lowest held-out log "
          "loss. soft + V: every poisoned row at its weight, all of V at 1 (the headline); soft: the poisoned rows "
          "alone. upd: the median chosen update. Each cell: ROC-AUC / accuracy / log loss.", ""]
    names = [f"{init}_lr{lr:g}" for init in ("ones", "vf_soft", "vf_hard") for lr in LRS] + \
            [f"random_s0_lr{lr:g}" for lr in LRS] + [f"random_ens_lr{lr:g}" for lr in LRS]
    fmt = lambda xs: " / ".join(f"{np.mean([x[m] for x in xs]):.3f}" for m, _ in METRICS) if xs else "."  # noqa: E731
    for regime in REGIMES:
        tags = [tag_of(ds, regime) for ds in DATASETS]
        md += [f"## {regime}", "", "| configuration | datasets | soft + V | soft | upd |", "|---|---|---|---|---|"]
        for name in names + (["best_soft"] if regime == "c5" else []):
            rs = [res[(t, name)] for t in tags if (t, name) in res]
            its = [r["sel"]["ce"]["it"] for r in rs if r["sel"]["ce"]["it"] is not None]
            md.append(f"| {name.replace('_s0', '')} | {len(rs)} | {fmt([r['sel']['ce']['test']['soft_v'] for r in rs])} | "
                      f"{fmt([r['sel']['ce']['test']['soft'] for r in rs])} | {int(np.median(its)) if its else '.'} |")
        refs = [references(ds, regime) for ds in DATASETS]
        md += ["", "| reference | soft + V or + V | without V |", "|---|---|---|"]
        for label, with_v, without_v in (("V only", "v_only", None), ("poisoned + V", "poisoned_plus_v", "poisoned"),
                                        ("V-filter (hard)", "vfilter", "vfilter_nov"), ("oracle + V", "oracle_plus_v", "oracle"),
                                        ("best soft form (w / V-filter / PGD, holdout)", "attn_wsoft", "attn_wsoft_nov"),
                                        ("tiny-V best soft (lr 0.2, 4 updates, all of V)", "attn_wsoft_lr0.2_it4", None)):
            xs = [r[with_v] for r in refs if with_v in r]
            ys = [r[without_v] for r in refs if without_v and without_v in r]
            if xs or ys:
                md.append(f"| {label} | {fmt(xs)} | {fmt(ys)} |")
        md.append("")
    # context: the form study's sigmoid (t in [-1000, 20], its chosen learning rates) at V = 500, same 6 datasets
    md += ["## Context: the form study's sigmoid at V = 500 (results/attn_form; t clipped to [-1000, 20])", "",
           "| configuration | datasets | soft + V | soft |", "|---|---|---|---|"]
    for cfg in ("ones_adam_lr0.2", "ones_pgd_lr1", "ones_sign_lr12.8", "vfilter_adam_lr3.2", "vfilter_pgd_lr16",
                "vfilter_sign_lr0.8"):
        rs = [json.loads(f.read_text()) for f in sorted(Path("results/attn_form/runs").glob(f"*/sigmoid_{cfg}.json"))]
        md.append(f"| sigmoid / {cfg.replace('_', ' / ', 1).replace('_lr', ', lr ')} | {len(rs)} | "
                  f"{fmt([r['sel']['ce']['test']['soft_v'] for r in rs])} | {fmt([r['sel']['ce']['test']['soft'] for r in rs])} |")
    (out_md / "report.md").write_text("\n".join(md) + "\n")
    print(f"-> {out_md / 'report.md'} ({len(res)} result files)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("work", "plan", "status", "report"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--job-list", type=Path, default=None)
    ap.add_argument("--scripts", type=int, default=9)
    a = ap.parse_args(argv)
    {"work": work, "plan": plan, "status": status, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()

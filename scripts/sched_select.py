#!/usr/bin/env python
"""Label-free (V-only) per-draw selection among the schedules of scripts/schedule_study.py.

For every finished run of the listed schedules on a draw, the final trusted set T (the accepted
context rows + V) is scored without test labels:
  vscore   log loss of V's rows under TabPFN with T minus V as the context (V is clean and was not
           chosen by the growth; at c = 1 it has one row per class, so this is noisy)
  cv2      two-fold cross-validated log loss of TabPFN on T (rows of each half predicted from the
           other half; a fixed random split per draw)
  cvacc    the same two folds' error rate
Scores go to <draw>/select.json (a schedule already scored is skipped); --report then picks, per
draw, the schedule with the lowest score and compares its test metric with each fixed schedule.

    python scripts/sched_select.py --gpu 1 --specs oneshot fixed:pct1 fixed:pct10 fixed:frac0.1 fixed:frac0.1:cprior
    python scripts/sched_select.py --report --specs ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("results/sched/runs")
AUC = {"coil2000_insurance_policies"}


def ce(q: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean())


def score(a) -> None:
    import schedule_study as S
    device = f"cuda:{a.gpu}"
    data = None
    for draw in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        out = draw / "select.json"
        done = json.loads(out.read_text()) if out.exists() else {}
        todo = [sp for sp in a.specs if sp not in done and (draw / f"{S.safe(sp)}.npz").exists()
                and (draw / f"{S.safe(sp)}.json").exists()]
        if not todo:
            continue
        r0 = json.loads((draw / f"{S.safe(todo[0])}.json").read_text())["draw"]
        if data is None or data.ds != r0["ds"]:
            data = None
            torch.cuda.empty_cache()
            data = S.Data(r0["ds"], device)
        d = S.get_draw(data, r0["R"], r0["c"], r0["seed"])
        n = d["n"]
        for sp in todo:
            pos = np.load(draw / f"{S.safe(sp)}.npz")["pos"]
            rows = np.flatnonzero(pos < n)
            X_T = torch.cat([d["Xc"][data.idx(rows)], d["Xv"]])
            y_T = np.concatenate([d["y_given"][rows], d["yv"]])
            # V predicted from the accepted rows alone
            qv = data.probs(d["Xc"][data.idx(rows)], d["y_given"][rows], d["Xv"], d["yv"])
            # two folds of T (every class of T in both folds where it can be)
            rng = np.random.default_rng(int(r0["seed"]) + 31)
            fold = np.zeros(len(y_T), dtype=int)
            for cl in np.unique(y_T):
                idx = rng.permutation(np.flatnonzero(y_T == cl))
                fold[idx[: len(idx) // 2]] = 1
            losses, errs = [], []
            for f in (0, 1):
                tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                q = data.probs(X_T[data.idx(tr)], y_T[tr], X_T[data.idx(te)], y_T[te])
                losses.append(ce(q, y_T[te]) * len(te))
                errs.append(float((q.argmax(1) != y_T[te]).sum()))
            done[sp] = {"vscore": ce(qv, d["yv"]), "vacc": float((qv.argmax(1) == d["yv"]).mean()),
                        "cv2": float(np.sum(losses) / len(y_T)), "cverr": float(np.sum(errs) / len(y_T))}
        out.write_text(json.dumps(done))
        print(draw.name, {sp: round(v["vscore"], 3) for sp, v in done.items()}, flush=True)


def report(a) -> None:
    rows = []
    for draw in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        f = draw / "select.json"
        if not f.exists():
            continue
        sel = json.loads(f.read_text())
        if not all(sp in sel for sp in a.specs):
            continue
        test = {}
        for sp in a.specs:
            r = json.loads((draw / f"{sp.replace(':', '_')}.json").read_text())
            m = "auc" if r["draw"]["ds"] in AUC else "acc"
            test[sp] = 100 * r["test"][m]
        rows.append((r["draw"], sel, test))
    print(f"{len(rows)} draws with every schedule scored")
    crit = {"vscore (V log loss, lower)": lambda s: s["vscore"], "V accuracy under T (higher)": lambda s: -s["vacc"],
            "cv2 (T 2-fold log loss, lower)": lambda s: s["cv2"], "cverr (T 2-fold error, lower)": lambda s: s["cverr"]}
    best = np.mean([max(t.values()) for _, _, t in rows])
    print(f"{'rule':34s} {'all':>7s} {'c1':>7s} {'c5':>7s}   (mean test metric; best per draw in hindsight {best:.2f})")
    def line(name, pick):
        v = {cc: np.mean([t[pick(s, t)] for dd, s, t in rows if cc is None or dd["c"] == cc]) for cc in (None, 1, 5)}
        print(f"{name:34s} {v[None]:7.2f} {v[1]:7.2f} {v[5]:7.2f}")
    for sp in a.specs:
        line(f"fixed {sp}", lambda s, t, sp=sp: sp)
    for nm, key in crit.items():
        line(f"select by {nm}", lambda s, t, key=key: min(a.specs, key=lambda sp: (key(s[sp]), a.specs.index(sp))))
    # rank correlation between each criterion and the test metric, within draws
    from scipy.stats import spearmanr
    for nm, key in crit.items():
        rho = [spearmanr([-key(s[sp]) for sp in a.specs], [t[sp] for sp in a.specs]).correlation for _, s, t in rows]
        rho = [x for x in rho if np.isfinite(x)]
        print(f"within-draw Spearman of {nm:32s} with the test metric: {np.mean(rho):+.3f} (n {len(rho)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    report(a) if a.report else score(a)


if __name__ == "__main__":
    main()

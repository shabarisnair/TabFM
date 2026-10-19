#!/usr/bin/env python
"""Report of the schedule study (scripts/schedule_study.py; results/sched/runs).

Primary metric: test accuracy (%), except coil2000, where every context predicts the majority
class, so test ROC-AUC (%) is used (--metric auc|ce|acc|bacc uses one metric for every dataset; for
ce lower is better). Cells: (dataset, R, clean rows per class c), each the mean over its seeds.

  - per cell, every schedule (and ensemble) next to V only, poisoned + V, oracle + V;
  - robustness: per schedule, the mean over cells, the regret against the best schedule of each
    cell (best minus it; mean and worst cell), the worst (dataset, R) group (regret averaged over
    c), the mean rank, and per-draw wins / losses against the V-filter (oneshot) and fixed:frac0.1.
Only cells where every listed schedule finished on every seed are compared.

    python scripts/report_sched.py [--metric auc] [--min-seeds 3] [--only spec1 spec2 ...]
        -> results/tables/sched/report[_<metric>].md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/sched")
BINARY_AUC = {"coil2000_insurance_policies"}
SHORT = {"gas-drift": "gas", "url_unique": "url", "heloc": "heloc", "students_dropout": "students",
         "credit-approval": "credit", "coil2000_insurance_policies": "coil"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default=None, choices=["acc", "auc", "ce", "bacc"])
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="compare only these schedules (and ensembles)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="only draws of these seeds")
    ap.add_argument("--cs", nargs="*", type=int, default=None, help="only draws with these clean rows per class")
    ap.add_argument("--out", default=None, help="report file name (default report[_<metric>].md)")
    a = ap.parse_args()
    vals = defaultdict(dict)   # (ds, R, c, seed) -> {schedule: metric}
    refs = {}
    prec = defaultdict(dict)

    def metric_of(ds):
        return a.metric or ("auc" if ds in BINARY_AUC else "acc")

    for draw in sorted((ROOT / "runs").iterdir()):
        if not draw.is_dir():
            continue
        for f in draw.glob("*.json"):
            if f.name in ("references.json", "posthoc.json", "select.json"):
                continue
            r = json.loads(f.read_text())
            if "draw" not in r:  # vstats.json and other per-draw side files
                continue
            dr = r["draw"]
            if (a.seeds is not None and int(dr["seed"]) not in a.seeds) or (a.cs is not None and int(dr["c"]) not in a.cs):
                continue
            key = (dr["ds"], float(dr["R"]), int(dr["c"]), int(dr["seed"]))
            m = metric_of(dr["ds"])
            vals[key][r["spec"]] = r["test"][m] * (1 if m == "ce" else 100)
            prec[key][r["spec"]] = r["prec"]
        if (draw / "posthoc.json").exists() and vals:
            ph = json.loads((draw / "posthoc.json").read_text())
            some = next((k for k in vals if f"{k[0]}_R{k[1]:g}_c{k[2]}_s{k[3]}" == draw.name), None)
            if some is not None:
                m = metric_of(some[0])
                for nm, r in ph.items():
                    vals[some][f"ens:{nm}"] = r["test"][m] * (1 if m == "ce" else 100)
                    prec[some][f"ens:{nm}"] = r["prec"]
        if (draw / "references.json").exists():
            refs[draw.name] = json.loads((draw / "references.json").read_text())
    lower = a.metric == "ce"
    specs = sorted({s for v in vals.values() for s in v})
    if a.only:
        specs = [s for s in specs if s in a.only or s.split("ens:")[-1] in a.only or s in [f"ens:{x}" for x in a.only]]
    # cells with every schedule on every seed
    cells = defaultdict(list)
    for key in vals:
        cells[key[:3]].append(key)
    complete = {cell: keys for cell, keys in cells.items()
                if len(keys) >= a.min_seeds and all(all(s in vals[k] for s in specs) for k in keys)}
    cell_mean = {cell: {s: float(np.mean([vals[k][s] for k in keys])) for s in specs} for cell, keys in complete.items()}
    better = (lambda x, y: x < y) if lower else (lambda x, y: x > y)
    best = {cell: (min if lower else max)(cm.values()) for cell, cm in cell_mean.items()}
    regret = {cell: {s: (cm[s] - best[cell]) if lower else (best[cell] - cm[s]) for s in specs} for cell, cm in cell_mean.items()}
    L = [f"# Schedule study ({len(complete)} complete cells; metric: "
         + (a.metric or "accuracy, coil2000 ROC-AUC") + (", lower is better" if lower else "") + ")", ""]
    # robustness summary
    groups = defaultdict(list)
    for cell in complete:
        groups[cell[:2]].append(cell)
    rows = []
    for s in specs:
        mean = np.mean([cell_mean[c][s] for c in complete])
        reg = [regret[c][s] for c in complete]
        greg = {g: np.mean([regret[c][s] for c in cs]) for g, cs in groups.items()}
        worst_g = max(greg, key=greg.get)
        ranks = []
        for c in complete:
            order = sorted(specs, key=lambda x: cell_mean[c][x], reverse=not lower)
            ranks.append(order.index(s) + 1)
        def wl(ref):
            w = l_ = 0
            for c, keys in complete.items():
                for k in keys:
                    if ref in vals[k]:
                        w += better(vals[k][s], vals[k][ref])
                        l_ += better(vals[k][ref], vals[k][s])
            return f"{w}-{l_}"
        pr = np.mean([prec[k][s] for c, keys in complete.items() for k in keys if s in prec[k]])
        rows.append((np.mean(reg), s, mean, np.mean(reg), np.max(reg), f"{SHORT[worst_g[0]]} R{worst_g[1]:g} ({greg[worst_g]:.1f})",
                     np.mean(ranks), wl("oneshot"), wl("fixed:frac0.1"), pr))
    rows.sort()
    L += ["## Robustness (all complete cells)", "",
          "mean: over cells; regret: best schedule of the cell minus this one (points; log loss: this one minus the best); "
          "worst group: the (dataset, R) with the largest regret averaged over c; draws: better-worse against the "
          "V-filter (oneshot) and geometric 10% (fixed:frac0.1); prec: precision of the rows left out.", "",
          "| schedule | mean | mean regret | worst cell regret | worst group | mean rank | draws vs V-filter | draws vs frac0.1 | prec |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, s, mean, mr, xr, wg, rk, w1, w2, pr in rows:
        L.append(f"| {s} | {mean:.2f} | {mr:.2f} | {xr:.2f} | {wg} | {rk:.1f} | {w1} | {w2} | {pr:.3f} |")
    # per clean-set size
    cs_ = sorted({c[2] for c in complete})
    L += ["", "## By clean rows per class c (mean / mean regret over the cells of that c)", "",
          "| schedule | " + " | ".join(f"c{c} mean | c{c} regret" for c in cs_) + " |", "|---|" + "---|---|" * len(cs_)]
    for _, s, *_ in rows:
        L.append(f"| {s} | " + " | ".join(f"{np.mean([cell_mean[c][s] for c in complete if c[2] == cc]):.2f} | "
                                         f"{np.mean([regret[c][s] for c in complete if c[2] == cc]):.2f}" for cc in cs_) + " |")
    # leave one dataset out: the schedule chosen on the other datasets, scored on the held-out one
    dss_ = sorted({c[0] for c in complete})
    if len(dss_) > 2:
        L += ["", "## Leave one dataset out", "",
              "For each held-out dataset, the schedule with the lowest mean regret (or the lowest worst-group regret) "
              "over the cells of the other datasets, and its mean regret on the held-out dataset.", "",
              "| held out | chosen (mean regret) | its regret | chosen (minimax) | its regret | best on it | its regret |",
              "|---|---|---|---|---|---|---|"]
        lo = {"mean": [], "minimax": []}
        for d in dss_:
            other = [c for c in complete if c[0] != d]
            mine = [c for c in complete if c[0] == d]
            if not mine:
                continue
            ch_mean = min(specs, key=lambda s: np.mean([regret[c][s] for c in other]))
            g_other = defaultdict(list)
            for c in other:
                g_other[c[:2]].append(c)
            ch_mm = min(specs, key=lambda s: max(np.mean([regret[c][s] for c in cs]) for cs in g_other.values()))
            r_of = lambda s: np.mean([regret[c][s] for c in mine])
            best_here = min(specs, key=r_of)
            lo["mean"].append(r_of(ch_mean))
            lo["minimax"].append(r_of(ch_mm))
            L.append(f"| {SHORT[d]} | {ch_mean} | {r_of(ch_mean):.2f} | {ch_mm} | {r_of(ch_mm):.2f} | {best_here} | {r_of(best_here):.2f} |")
        L.append(f"| **mean over held-out datasets** | | {np.mean(lo['mean']):.2f} | | {np.mean(lo['minimax']):.2f} | | |")
    # per dataset means
    L += ["", "## Per dataset (mean over its complete cells)", "",
          "| schedule | " + " | ".join(SHORT[d] for d in sorted({c[0] for c in complete})) + " |",
          "|---|" + "---|" * len({c[0] for c in complete})]
    dss = sorted({c[0] for c in complete})
    for _, s, *_ in rows:
        L.append(f"| {s} | " + " | ".join(f"{np.mean([cell_mean[c][s] for c in complete if c[0] == d]):.1f}" for d in dss) + " |")
    refrow = {}
    for nm in ("v_only", "poisoned_plus_v", "oracle_plus_v"):
        per = []
        for d in dss:
            xs = []
            for c, keys in complete.items():
                if c[0] != d:
                    continue
                for k in keys:
                    rr = refs.get(f"{k[0]}_R{k[1]:g}_c{k[2]}_s{k[3]}")
                    if rr:
                        m = metric_of(d)
                        xs.append(rr[nm][m] * (1 if m == "ce" else 100))
            per.append(f"{np.mean(xs):.1f}" if xs else "")
        refrow[nm] = per
        L.append(f"| *{nm}* | " + " | ".join(per) + " |")
    # per cell tables
    L += ["", "## Per cell (mean over seeds)", ""]
    for d in dss:
        cs = sorted(c for c in complete if c[0] == d)
        L += [f"### {d}", "", "| schedule | " + " | ".join(f"R{c[1]:g} c{c[2]}" for c in cs) + " |", "|---|" + "---|" * len(cs)]
        for _, s, *_ in rows:
            L.append(f"| {s} | " + " | ".join(("**" if regret[c][s] < 1e-9 else "") + f"{cell_mean[c][s]:.1f}"
                                              + ("**" if regret[c][s] < 1e-9 else "") for c in cs) + " |")
        for nm in ("v_only", "oracle_plus_v"):
            xs = []
            for c in cs:
                v = [refs[f"{k[0]}_R{k[1]:g}_c{k[2]}_s{k[3]}"][nm][metric_of(d)] * (1 if metric_of(d) == "ce" else 100)
                     for k in complete[c] if f"{k[0]}_R{k[1]:g}_c{k[2]}_s{k[3]}" in refs]
                xs.append(f"{np.mean(v):.1f}" if v else "")
            L.append(f"| *{nm}* | " + " | ".join(xs) + " |")
        L.append("")
    out = Path("results/tables/sched")
    out.mkdir(parents=True, exist_ok=True)
    name = a.out or f"report{'_' + a.metric if a.metric else ''}.md"
    (out / name).write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(rows) + 8]))
    print(out / name)


if __name__ == "__main__":
    main()

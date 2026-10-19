#!/usr/bin/env python
"""Tables and figures of the tiny-clean-set benchmark (scripts/bench_small_v.py).

Per (dataset, R, c) cell, the mean over the evaluation seeds of the test accuracy of TabPFN with
the cleaned context, for the references, every baseline in the mode and config its tuning picked,
ours with the candidate its tuning picked (per cell, like the baselines), ours with one fixed recipe
(the candidate with the best mean tuning accuracy over every tuning draw) and the strongest baseline
of the cell (the best test accuracy among the tuned baselines: generous to the baselines). Also the
precision of the removed rows (each method's tuned removal config), the modes and configs the tuning
chose, and per draw a paired bootstrap over test rows (2000 resamples, 95%) of ours against every
other method.

    python scripts/report_small_v.py   # -> results/tables/small_v/*.md|csv, results/plots/small_v/*.png
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_small_v import CS, DATASETS, EVAL_SEEDS, ROOT, RS, TUNE_SEEDS, read_rows, selected_key  # noqa: E402

OUT_T = Path(os.environ.get("SMALL_V_TABLES", "results/tables/small_v"))
OUT_P = Path(os.environ.get("SMALL_V_PLOTS", "results/plots/small_v"))
REFS = (("v_only", "V only"), ("poisoned_plus_v", "V + poisoned"), ("vfilter", "V-filter"))
BASELINES = (("knn_sanitize", "kNN sanitisation"), ("label_spread", "label spreading"), ("knn_shapley", "KNN-Shapley"),
             ("data_oob", "Data-OOB"), ("cleanlab", "cleanlab"), ("simifeat", "SimiFeat"), ("glc", "GLC"),
             ("mwnet", "MW-Net"), ("emlc", "EMLC"), ("ieg", "IEG"))
DS_COLORS = {"gas-drift": "#2a78d6", "url_unique": "#eb6834", "heloc": "#1baf7a", "students_dropout": "#eda100",
             "credit-approval": "#e87ba4"}
INK, MUTED, GRID, SURFACE = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"
MODE_ABBR = {"remove": "rm", "relabel": "rl", "relabel_k": "rlk", "flagged": "fl", "native": "nat"}


def boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    d = a.astype(float) - b.astype(float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_P.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "selected.json").read_text())
    ev = read_rows(sorted((ROOT / "eval").glob("*.csv")))
    t1 = read_rows(sorted((ROOT / "tune1").glob("*__ours.csv")))

    # ours, one fixed recipe: the best mean tuning accuracy over every tuning draw
    tune_ours = defaultdict(list)
    for r in t1:
        if r["method"] == "ours":
            tune_ours[r["cfg"]].append(float(r["acc"]))
    fixed = max(tune_ours, key=lambda v: np.mean(tune_ours[v]))

    acc, prec, correct = {}, {}, {}  # (ds, R, c, seed, entry) -> value; entry = method or "ours|<variant>" or "<m>|<mode>"
    for r in ev:
        key = (r["dataset"], float(r["R"]), int(r["c"]), int(r["seed"]))
        entry = r["method"] if r["method"] in dict(REFS) or r["method"] == "oracle_plus_v" else \
            f"ours|{r['cfg']}" if r["method"] == "ours" else f"{r['method']}|{r['mode']}"
        acc[key + (entry,)] = 100 * float(r["acc"])
        if r["prec"] != "":
            prec[key + (entry,)] = float(r["prec"])
    for f in sorted((ROOT / "eval").glob("*_correct.npz")):
        tag = f.name.split("__")[0]
        ds = next(d for d in DATASETS if tag.startswith(d + "_"))
        rest = tag[len(ds) + 1:]
        R, c, s = float(rest.split("_")[0][1:]), int(rest.split("_")[1][1:]), int(rest.split("_")[2][1:])
        z = np.load(f)
        for k in z.files:
            correct[(ds, R, c, s, k)] = z[k]

    def entry_of(ds, R, c, method):  # the tuned entry of a method in a cell
        g = sel[selected_key(ds, R, c)]
        if method == "ours":
            return f"ours|{g['ours']['modes']['remove']}"
        if method == "ours_fixed":
            return f"ours|{fixed}"
        if method in g:
            return f"{method}|{g[method]['best']}"
        return method

    def corr_key(entry):
        return entry.replace("|", "__").replace(":", "_")

    cols = [m for m, _ in REFS] + [m for m, _ in BASELINES] + ["best_baseline", "ours", "ours_fixed", "oracle_plus_v"]
    labels = dict(REFS) | dict(BASELINES) | {"best_baseline": "strongest baseline", "ours": "ours (tuned)",
                                             "ours_fixed": f"ours ({fixed.replace('grow:', '')})",
                                             "oracle_plus_v": "oracle + V"}
    cells, table, ptable = [], {}, {}
    for ds in DATASETS:
        for R in RS:
            for c in CS:
                if selected_key(ds, R, c) not in sel:
                    continue
                row, prow = {}, {}
                for m in cols:
                    if m == "best_baseline":
                        continue
                    vals = [acc.get((ds, R, c, s, entry_of(ds, R, c, m))) for s in EVAL_SEEDS]
                    if any(v is None for v in vals):
                        continue
                    row[m] = float(np.mean(vals))
                    ent = entry_of(ds, R, c, m)
                    if m in dict(BASELINES):  # precision of the tuned removal config
                        ent = f"{m}|remove"
                    p = [prec.get((ds, R, c, s, ent)) for s in EVAL_SEEDS]
                    if all(v is not None for v in p):
                        prow[m] = float(np.mean(p))
                bl = [row[m] for m, _ in BASELINES if m in row]
                if bl:
                    row["best_baseline"] = max(bl)
                cells.append((ds, R, c))
                table[(ds, R, c)], ptable[(ds, R, c)] = row, prow

    # ---- main tables
    L = ["# Tiny clean set: greedy growth vs label-noise baselines (hub-flip-local)", "",
         f"V = c clean rows per class; R% of the rest flipped; mean test accuracy (%) over seeds {list(EVAL_SEEDS)} "
         "of TabPFN with the cleaned context (V + kept / relabelled rows; native: the method's own classifier). Every "
         "baseline in the mode and config picked on separate tuning draws (seeds 100-102, 20% of the context file held "
         "out as their test set; cheap shortlist per mode, then TabPFN accuracy). Ours (tuned): the step schedule picked the "
         f"same way; ours ({fixed.replace('grow:', '')}): one fixed recipe, the best over every tuning draw. Strongest "
         "baseline: the best test accuracy among the ten tuned baselines in that cell (picked on the test set). "
         "Bold: best non-oracle entry.", ""]
    hdr = ["dataset", "R", "c"] + [labels[m] for m in cols]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    csv_rows = []
    for cell in cells:
        row = table[cell]
        best = max(v for m, v in row.items() if m not in ("oracle_plus_v", "best_baseline"))
        out = [cell[0], f"{cell[1]:g}", str(cell[2])]
        for m in cols:
            v = row.get(m)
            out.append("" if v is None else (f"**{v:.1f}**" if abs(v - best) < 1e-9 and m != "oracle_plus_v" else f"{v:.1f}"))
        L.append("| " + " | ".join(out) + " |")
        csv_rows.append({"dataset": cell[0], "R": cell[1], "c": cell[2]} | {m: row.get(m, "") for m in cols})
    L += ["", "## Mean over the cells", ""]
    L.append("| " + " | ".join(labels[m] for m in cols) + " |")
    L.append("|" + "---|" * len(cols))
    L.append("| " + " | ".join(f"{np.mean([table[c][m] for c in cells if m in table[c]]):.1f}" for m in cols) + " |")

    # ---- ours against every method: cells won / lost and draws significantly better / worse
    L += ["", "## Ours (tuned) against every other entry", "",
          "Cells: mean accuracy over the seeds higher-lower. Draws: per evaluation draw, the 95% paired bootstrap over "
          "test rows; significantly higher-lower.", "", "| against | mean gap (points) | cells | draws |", "|---|---|---|---|"]
    for m in [x for x in cols if x not in ("ours", "best_baseline")]:
        gaps = [table[c]["ours"] - table[c][m] for c in cells if "ours" in table[c] and m in table[c]]
        hi, lo = sum(g > 1e-9 for g in gaps), sum(g < -1e-9 for g in gaps)
        s_hi = s_lo = 0
        for (ds, R, c) in cells:
            for s in EVAL_SEEDS:
                a = correct.get((ds, R, c, s, corr_key(entry_of(ds, R, c, "ours"))))
                b = correct.get((ds, R, c, s, corr_key(entry_of(ds, R, c, m))))
                if a is None or b is None:
                    continue
                lo_ci, hi_ci = boot_ci(a, b)
                s_hi += lo_ci > 0
                s_lo += hi_ci < 0
        L.append(f"| {labels[m]} | {np.mean(gaps):+.2f} | {hi}-{lo} | {s_hi}-{s_lo} |")

    # ---- precision
    pcols = ["vfilter"] + [m for m, _ in BASELINES] + ["ours", "ours_fixed"]
    L += ["", "## Precision of the removed rows (each method's tuned removal config)", ""]
    L.append("| dataset | R | c | " + " | ".join(labels[m] for m in pcols) + " |")
    L.append("|" + "---|" * (len(pcols) + 3))
    for cell in cells:
        L.append(f"| {cell[0]} | {cell[1]:g} | {cell[2]} | " + " | ".join(
            f"{ptable[cell][m]:.2f}" if m in ptable[cell] else "" for m in pcols) + " |")
    L.append("| mean | | | " + " | ".join(f"{np.mean([ptable[c][m] for c in cells if m in ptable[c]]):.2f}" for m in pcols) + " |")

    # ---- every candidate of ours
    from bench_small_v import OURS
    L += ["", "## Every step schedule of ours (test accuracy; * = picked by the tuning)", ""]
    L.append("| dataset | R | c | " + " | ".join(v.replace("grow:", "") for v in OURS) + " |")
    L.append("|" + "---|" * (len(OURS) + 3))
    for (ds, R, c) in cells:
        picked = sel[selected_key(ds, R, c)].get("ours", {}).get("modes", {}).get("remove")
        vals = []
        for v in OURS:
            a = [acc.get((ds, R, c, s, f"ours|{v}")) for s in EVAL_SEEDS]
            vals.append("" if any(x is None for x in a) else f"{np.mean(a):.1f}" + ("*" if v == picked else ""))
        L.append(f"| {ds} | {R:g} | {c} | " + " | ".join(vals) + " |")
    means = []
    for v in OURS:
        a = [acc.get((ds, R, c, s, f"ours|{v}")) for (ds, R, c) in cells for s in EVAL_SEEDS]
        means.append(f"{np.mean([x for x in a if x is not None]):.1f}" if any(x is not None for x in a) else "")
    L.append("| mean | | | " + " | ".join(means) + " |")

    # ---- what the tuning chose
    L += ["", "## What the tuning chose", "",
          "Per baseline: how often each mode was picked over the cells (rm remove, rl relabel, rlk relabel the k most "
          "suspicious, fl cleanlab's own issue set, nat the method's own classifier), and the most frequent config.", "",
          "| method | modes picked | most frequent config (cells) |", "|---|---|---|"]
    for m, name in BASELINES + (("ours", "ours"),):
        modes, cfgs = Counter(), Counter()
        for cell in cells:
            g = sel[selected_key(*cell)].get(m)
            if g:
                modes[g["best"]] += 1
                cfgs[g["modes"][g["best"]]] += 1
        top = cfgs.most_common(1)
        L.append(f"| {name} | " + ", ".join(f"{MODE_ABBR.get(k, k)} {v}" for k, v in modes.most_common()) +
                 f" | {top[0][0] if top else ''} ({top[0][1] if top else 0}) |")
    text = "\n".join(L) + "\n"
    (OUT_T / "summary.md").write_text(text)
    with open(OUT_T / "accuracy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "R", "c"] + cols)
        w.writeheader()
        w.writerows(csv_rows)
    print(text)

    # ---- figure 1: every method's gap to ours, one dot per cell
    others = [m for m in cols if m not in ("ours", "best_baseline", "ours_fixed")]
    fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    ax.axhline(0, color="#8a8985", lw=1.2, zorder=1)
    rng = np.random.default_rng(0)
    for i, m in enumerate(others):
        ks = [c for c in cells if m in table[c] and "ours" in table[c]]
        d = np.array([table[c][m] - table[c]["ours"] for c in ks])
        for ds, color in DS_COLORS.items():
            j = [t for t, c in enumerate(ks) if c[0] == ds]
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(j)), d[j], s=24, color=color, edgecolor=SURFACE, lw=0.8, zorder=3,
                       label=ds if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [d.mean()] * 2, color=INK, lw=2.4, zorder=4)
        ax.annotate(f"{d.mean():+.1f}", (i, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(others)), [labels[m] for m in others], rotation=30, ha="right", fontsize=9, color=INK)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.set_ylabel("test accuracy minus ours (tuned), points", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.legend(loc="outside lower center", fontsize=9, frameon=False, labelcolor=INK, ncol=5)
    ax.set_title(f"Every method against ours with a tiny clean set ({len(cells)} cells: 5 datasets x R 20/30% x c 1/3/5 "
                 "clean rows per class; mean of 3 draws each; bar = mean)", loc="left", fontsize=10.5, color=INK)
    fig.savefig(OUT_P / "gap_to_ours.png", dpi=150, facecolor=SURFACE)

    # ---- figure 2: accuracy against c, per dataset and R
    series = (("v_only", "V only", "#eda100", "o"), ("vfilter", "V-filter", "#2a78d6", "o"),
              ("best_baseline", "strongest baseline (picked on test)", "#6250d6", "s"),
              ("ours", "ours (tuned)", "#eb6834", "D"), ("oracle_plus_v", "oracle + V", "#8a8985", None))
    fig, axes = plt.subplots(len(RS), len(DATASETS), figsize=(19, 7.6), constrained_layout=True, squeeze=False)
    for i, R in enumerate(RS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for m, name, color, marker in series:
                xs = [c for c in CS if (ds, R, c) in table and m in table[(ds, R, c)]]
                ys = [table[(ds, R, c)][m] for c in xs]
                ax.plot(xs, ys, color=color, lw=1.8, marker=marker, ms=5, ls="--" if m == "oracle_plus_v" else "-",
                        label=name if (i, j) == (0, 0) else None)
            ax.set_xticks(list(CS))
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            ax.set_title(f"{ds}, R = {R:g}%", loc="left", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("test accuracy (%)", color=INK)
            if i == len(RS) - 1:
                ax.set_xlabel("clean rows per class (c)", color=INK)
    fig.legend(loc="outside lower center", ncol=5, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Accuracy with a tiny clean set (mean of 3 draws)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT_P / "accuracy_vs_c.png", dpi=150, facecolor=SURFACE)
    print(OUT_T / "summary.md", OUT_P / "gap_to_ours.png", OUT_P / "accuracy_vs_c.png")


if __name__ == "__main__":
    main()

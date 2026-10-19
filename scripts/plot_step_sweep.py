#!/usr/bin/env python
"""How many rows should the greedy move into the trusted set per step? (night 2, E16)

Reads every iterative-filter run (results/iterative_filter/e*/rows_*.csv) of the fixed-step
greedy (grow:m<m>, plain CE) and the V-filter (oneshot = one step of n - k rows) on the
hub-flip-local, R = 20%, stratified draws, and draws per dataset test accuracy (top) and
precision of the removed rows (bottom) against the rows moved per step (log scale; the V-filter
at m = n - k, the rightmost point), one line per V: the mean over the validation seeds run at that
m (band: their range), oracle + V dashed. Also writes the table of every (dataset, V)'s means, the
seeds behind them and its best m (results/tables/iterative_filter/step_sweep.md).

    python scripts/plot_step_sweep.py   # -> results/plots/iterative_filter/step_sweep.png
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

DATASETS = ("gas-drift", "url_unique", "credit-approval", "students_dropout", "heloc")
V_COLOR = {50: "#2a78d6", 100: "#6250d6", 200: "#eb6834", 500: "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"


def load() -> dict:
    rows = {}
    for f in sorted(Path("results/iterative_filter").glob("e*/rows_*.csv")):
        for r in csv.DictReader(open(f)):
            if (r["attack"], r["form"], float(r["R"])) != ("hub-flip-local", "stratified", 20.0):
                continue
            key = (r["dataset"], int(r["V"]), int(r["seed"]))
            n, k = int(r["n"]), int(r["k"])
            mm = re.fullmatch(r"grow:m(\d+)", r["variant"])
            m = n - k if r["variant"] == "oneshot" else int(mm.group(1)) if mm else None
            if r["variant"] == "oracle_plus_v":
                rows.setdefault(key, {})["oracle"] = 100 * float(r["acc"])
            if m is None:
                continue
            rows.setdefault(key, {}).setdefault(min(m, n - k), (100 * float(r["acc"]), float(r["prec"]), r["variant"]))
            rows[key]["all"] = n - k
    return rows


def main() -> None:
    per_seed = load()
    draws = {}  # (dataset, V) -> {m: (mean acc, mean prec, min acc, max acc, seeds)}, plus oracle and all
    for (ds, V) in sorted({(k[0], k[1]) for k, r in per_seed.items() if sum(isinstance(m, int) for m in r if m != "all") >= 5}):
        runs = [r for k, r in per_seed.items() if k[:2] == (ds, V) and sum(isinstance(m, int) for m in r if m != "all") >= 5]
        allm = runs[0]["all"]
        ms = sorted({min(m, allm) for r in runs for m in r if isinstance(m, int)})
        d = {"all": allm, "oracle": np.mean([r["oracle"] for r in runs if "oracle" in r])}
        for m in ms:
            vals = [r[m] for r in runs if m in r]
            a = [v[0] for v in vals]
            d[m] = (float(np.mean(a)), float(np.mean([v[1] for v in vals])), min(a), max(a), len(vals))
        draws[(ds, V)] = d
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(19, 7.6), constrained_layout=True)
    L = ["# Rows moved into the trusted set per step (greedy grow:m<m>; m = all is the V-filter)", "",
         "Test accuracy % (precision of the removed rows), mean over the validation seeds run (count in brackets when "
         "more than one). hub-flip-local, R = 20%, stratified V.", ""]
    for col, ds in enumerate(DATASETS):
        for key in sorted(k for k in draws if k[0] == ds):
            r = draws[key]
            ms = sorted(m for m in r if isinstance(m, int))
            acc = [r[m][0] for m in ms]
            prec = [r[m][1] for m in ms]
            color = V_COLOR.get(key[1], "#8a8985")
            axes[0, col].fill_between(ms, [r[m][2] for m in ms], [r[m][3] for m in ms], color=color, alpha=0.12, lw=0)
            axes[0, col].plot(ms, acc, color=color, lw=1.8, marker="o", ms=5, mec=SURFACE, label=f"V = {key[1]}")
            axes[1, col].plot(ms, prec, color=color, lw=1.8, marker="o", ms=5, mec=SURFACE, label=f"V = {key[1]}")
            axes[0, col].axhline(r["oracle"], color=color, lw=1, ls="--", alpha=0.6)
            best_a, best_p = ms[int(np.argmax(acc))], ms[int(np.argmax(prec))]
            lab = lambda m: "all" if m == r["all"] else str(m)  # noqa: E731
            L.append(f"- {ds} V{key[1]}: " + ", ".join(f"m={lab(m)} {r[m][0]:.1f} ({r[m][1]:.2f}){'' if r[m][4] == 1 else f' [{r[m][4]} seeds]'}" for m in ms)
                     + f" | best accuracy at m = {lab(best_a)}, best precision at m = {lab(best_p)} | oracle + V {r['oracle']:.1f}")
        for row, ylabel in enumerate(("test accuracy (%)", "precision of the removed rows")):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK)
        axes[0, col].set_title(ds, loc="left", fontsize=10.5, color=INK)
        axes[1, col].set_xlabel("rows moved per step (log; rightmost = all, the V-filter)", color=INK, fontsize=9)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    order = np.argsort([int(lb.split("= ")[1]) for lb in labels])
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center", ncol=len(labels),
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Greedy growth: rows moved into the trusted set per step (hub-flip-local, R = 20%; mean over validation seeds, "
                 "band: their range; dashed: oracle + V)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    out = Path("results/plots/iterative_filter/step_sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    Path("results/tables/iterative_filter/step_sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()

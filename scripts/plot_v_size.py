#!/usr/bin/env python
"""How small can the clean set be? Test accuracy and precision against V (night 2, E17).

Reads every iterative-filter run (results/iterative_filter/e*/rows_*.csv) on hub-flip-local,
R = 20%, stratified draws and plots, per dataset, against V (log scale): the V clean rows alone,
the V-filter (oneshot), the greedy with 1% steps (grow:pct1), geometric growth (grow:frac0.1,
grow:frac0.02: one row at a time while T is small), geometric growth with the prior toward the
context's labels (grow:frac0.1:cprior), the exact one-row greedy (grow:m1) and oracle + V, each the
mean over the validation seeds run at that V. Precision of the removed rows below. Writes the
table of means (seeds in brackets) to results/tables/iterative_filter/v_size.md.

    python scripts/plot_v_size.py   # -> results/plots/iterative_filter/v_size.png
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

DATASETS = ("gas-drift", "url_unique", "credit-approval", "students_dropout", "heloc")
SERIES = (("v_only", "V clean rows alone", "#eda100", "o"), ("oneshot", "V-filter", "#2a78d6", "o"),
          ("grow:pct1", "greedy, 1% of n per step", "#eb6834", "s"), ("grow:frac0.1", "geometric growth, 10% of |T|", "#1baf7a", "^"),
          ("grow:frac0.02", "geometric growth, 2% of |T|", "#6250d6", "v"),
          ("grow:frac0.1:cprior", "geometric 10% + prior toward the context's labels", "#e87ba4", "D"),
          ("grow:m1", "exact one-row greedy", "#008300", "x"))
INK, MUTED, GRID, SURFACE = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"


def main() -> None:
    vals = defaultdict(list)  # (dataset, V, variant) -> [(acc, prec)] over seeds
    seen = set()
    for f in sorted(Path("results/iterative_filter").glob("e*/rows_*.csv")):
        for r in csv.DictReader(open(f)):
            if (r["attack"], r["form"], float(r["R"])) != ("hub-flip-local", "stratified", 20.0):
                continue
            key = (r["dataset"], int(r["V"]), int(r["seed"]), r["variant"])
            if key in seen:
                continue
            seen.add(key)
            vals[(r["dataset"], int(r["V"]), r["variant"])].append((100 * float(r["acc"]), float(r["prec"]) if r["prec"] else np.nan))
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(19, 7.8), constrained_layout=True)
    L = ["# How small can the clean set be? (hub-flip-local, R = 20%, stratified V)", "",
         "Test accuracy % (precision of the removed rows), mean over the validation seeds run [count]. V = 1 is impossible "
         "(TabPFN cannot fit a one-row context); the smallest V is one row per class.", ""]
    for col, ds in enumerate(DATASETS):
        Vs_all = sorted({V for (d, V, _) in vals if d == ds})
        orc = [np.mean([a for a, _ in vals[(ds, V, "oracle_plus_v")]]) for V in Vs_all if (ds, V, "oracle_plus_v") in vals]
        if orc:
            axes[0, col].axhline(np.mean(orc), color="#8a8985", lw=1.2, ls="--")
        for key, label, color, marker in SERIES:
            Vs = [V for V in Vs_all if (ds, V, key) in vals]
            if not Vs:
                continue
            acc = [np.mean([a for a, _ in vals[(ds, V, key)]]) for V in Vs]
            axes[0, col].plot(Vs, acc, color=color, lw=1.8, marker=marker, ms=5, label=label)
            if key != "v_only":
                prec = [np.nanmean([p for _, p in vals[(ds, V, key)]]) for V in Vs]
                axes[1, col].plot(Vs, prec, color=color, lw=1.8, marker=marker, ms=5, label=label)
        L.append(f"## {ds}")
        L.append("| V | " + " | ".join(lbl for _, lbl, _, _ in SERIES) + " | oracle + V |")
        L.append("|---|" + "---|" * (len(SERIES) + 1))
        for V in Vs_all:
            cells = []
            for key, _, _, _ in SERIES + (("oracle_plus_v", "", "", ""),):
                v = vals.get((ds, V, key))
                if not v:
                    cells.append("")
                    continue
                a, p = np.mean([x for x, _ in v]), np.nanmean([y for _, y in v]) if key not in ("v_only", "oracle_plus_v") else None
                cells.append(f"{a:.1f}" + (f" ({p:.2f})" if p is not None else "") + (f" [{len(v)}]" if len(v) > 1 else ""))
            L.append(f"| {V} | " + " | ".join(cells) + " |")
        L.append("")
        for row, ylabel in enumerate(("test accuracy (%)", "precision of the removed rows")):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 5, 10, 20, 50, 100, 200, 500]))
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
        axes[1, col].set_xlabel("V (clean rows, log)", color=INK)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("How small can the clean set be? (hub-flip-local, R = 20%; mean over validation seeds; dashed: oracle + V)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    out = Path("results/plots/iterative_filter/v_size.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    Path("results/tables/iterative_filter/v_size.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(out)


if __name__ == "__main__":
    main()

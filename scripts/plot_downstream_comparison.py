#!/usr/bin/env python
"""Grouped bars: TabPFNv2 ROC-AUC on the held-out test set, per method and condition.

The reference lines are what matter more than the bar heights:
  solid grey  -- DOING NOTHING (the poisoned context). A method only helps above this.
  dashed      -- the oracle, i.e. perfect removal.
A bar between the two is a partial repair; a bar below the solid line means the
"defence" made the model worse than leaving the poison in.

Colour: only the four METHODS get bars, in the slots validated all-pairs in
plot_method_comparison.py. pure-m is a control rather than a method, so it is drawn as a
reference mark alongside the other two instead of competing as a fifth bar -- which also
avoids handing a neutral grey a categorical slot it would fail (chroma 0.008, lightness
0.782, contrast 1.94:1). At the full budget it coincides with the oracle by
construction, since "m random clean rows" and "all clean rows" are then the same set.

    python scripts/plot_downstream_comparison.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
COLOUR = {"k-fold": "#2a78d6", "sift": "#eb6834", "meta-sift": "#1baf7a",
          "iclad": "#4a3aa7"}
LABEL = {"k-fold": "k-fold", "sift": "sift (ours)", "meta-sift": "Meta-Sift",
         "iclad": "ICLAD"}
SERIES = ["k-fold", "sift", "meta-sift", "iclad"]
DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
SH = {"coil2000_insurance_policies": "coil2000", "credit-approval": "credit-appr."}
ATTACKS = [("label-flip-random", "random label flips"), ("hub-flip", "hub label flips")]


def figure(rows, full_budget, metric, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    width, handles = 0.2, []
    for ax, (atk, alabel) in zip(axes, ATTACKS):
        ax.set_facecolor(SURFACE)
        xs = np.arange(len(DATASETS))
        for j, meth in enumerate(SERIES):
            vals = []
            for ds in DATASETS:
                r = next((x for x in rows if x["dataset"] == ds and x["attack"] == atk
                          and x["context"] == meth
                          and (x["is_full_budget"] == "True") == full_budget), None)
                vals.append(float(r[metric]) if r else np.nan)
            off = (j - (len(SERIES) - 1) / 2) * width
            b = ax.bar(xs + off, vals, width * 0.9, color=COLOUR[meth], label=LABEL[meth],
                       edgecolor=SURFACE, linewidth=1.1, zorder=3)
            if ax is axes[0]:
                handles.append(b)
        # per-dataset reference marks: doing nothing, and perfect removal
        for i, ds in enumerate(DATASETS):
            g = lambda c: float(next(x for x in rows if x["dataset"] == ds and x["attack"] == atk  # noqa: E731
                                     and x["context"] == c
                                     and (x["is_full_budget"] == "True") == full_budget)[metric])
            lo, hi = i - 0.46, i + 0.46
            ax.plot([lo, hi], [g("poisoned")] * 2, color=INK, lw=1.6, zorder=6,
                    label="no defence" if (i == 0 and ax is axes[0]) else None)
            ax.plot([lo, hi], [g("oracle")] * 2, color=INK_3, lw=1.4, ls=(0, (3, 2)), zorder=6,
                    label="oracle (perfect removal)" if (i == 0 and ax is axes[0]) else None)
            pm = g("pure-m")
            if abs(pm - g("oracle")) > 1e-9:   # at the full budget they coincide
                ax.plot([lo, hi], [pm] * 2, color=INK_3, lw=1.4, ls=(0, (1, 1.6)), zorder=6,
                        label="pure-m (same size, all clean)"
                        if (i == 0 and ax is axes[0]) else None)
        ax.set_title(alabel, fontsize=10.5, color=INK, pad=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([SH.get(d, d) for d in DATASETS], fontsize=8.5, color=INK_2)
        ax.set_ylim(0.2, 1.02)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("ROC-AUC on the held-out test set", fontsize=9.5, color=INK_2)
    fig.suptitle(title, x=0.007, y=0.975, ha="left", fontsize=13, color=INK)
    fig.text(0.007, 0.905,
             "R=30% · solid line = doing nothing (poisoned context) · dashed = oracle "
             "(perfect removal) · a bar below the solid line made things worse",
             ha="left", fontsize=8.5, color=INK_2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=7, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=1.4, columnspacing=1.6,
               handletextpad=0.5)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.80, bottom=0.175, wspace=0.05)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/tables/comparison/downstream_comparison.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/comparison"))
    ap.add_argument("--metric", default="roc_auc")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.csv)))
    small = sorted({int(r["size"]) for r in rows if r["is_full_budget"] == "False"})[0]
    print("wrote", figure(rows, False, a.metric,
                          f"Downstream {a.metric.upper()} — top-{small} kept",
                          a.out / f"downstream_top{small}_{a.metric}.png"))
    print("wrote", figure(rows, True, a.metric,
                          f"Downstream {a.metric.upper()} — full budget (n − k kept)",
                          a.out / f"downstream_full_budget_{a.metric}.png"))


if __name__ == "__main__":
    main()

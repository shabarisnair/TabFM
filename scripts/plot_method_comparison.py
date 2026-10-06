#!/usr/bin/env python
"""Grouped bars: detection precision of the four methods, per dataset and attack.

One figure per subset size, two panels (random label-flip | hub-flip), five dataset
groups, four bars per group. The dashed rule is chance (1-R = 0.70): keeping rows at
random gives that, so only the part of a bar above the rule is evidence of skill.

Colour: categorical slots blue / orange / aqua / violet, one per method. Validated with
scripts/validate_palette.py on the ALL-PAIRS pairlist, which is the right one for
grouped bars since every pair sits adjacent inside a group: worst CVD dE 9.2 (deutan,
target >=8), worst normal-vision dE 16.3 (floor >=15). Violet also clears 3:1 contrast,
so no series depends on a label to be found.

    python scripts/plot_method_comparison.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"
COLOUR = {"k-fold": "#2a78d6", "sift": "#eb6834",
          "meta-sift": "#1baf7a", "iclad": "#4a3aa7"}
LABEL = {"k-fold": "k-fold", "sift": "sift (ours, TabPFN)",
         "meta-sift": "Meta-Sift", "iclad": "ICLAD"}
METHODS = ["k-fold", "sift", "meta-sift", "iclad"]
DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
SH = {"coil2000_insurance_policies": "coil2000", "credit-approval": "credit-appr."}
ATTACKS = [("label-flip-random", "random label flips"), ("hub-flip", "hub label flips")]


def figure(rows, full_budget: bool, title: str, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    width = 0.2
    handles = []

    for ax, (atk, alabel) in zip(axes, ATTACKS):
        ax.set_facecolor(SURFACE)
        xs = np.arange(len(DATASETS))
        for j, meth in enumerate(METHODS):
            vals, errs = [], []
            for ds in DATASETS:
                r = next((x for x in rows if x["dataset"] == ds and x["attack"] == atk
                          and x["method"] == meth
                          and (x["is_full_budget"] == "True") == full_budget), None)
                vals.append(float(r["purity"]) if r else np.nan)
                errs.append(float(r["purity_std"]) if r else 0.0)
            off = (j - (len(METHODS) - 1) / 2) * width
            b = ax.bar(xs + off, vals, width * 0.92, color=COLOUR[meth], label=LABEL[meth],
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)
            # k-fold is a mean over its five fold draws; the others are single runs.
            if any(e > 0 for e in errs):
                ax.errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor=INK_2,
                            elinewidth=1, capsize=2, zorder=4)
            if ax is axes[0]:
                handles.append(b)

        chance = float(next(x for x in rows if x["attack"] == atk)["chance"])
        ax.axhline(chance, color=INK_3, linewidth=1.3, linestyle=(0, (4, 3)), zorder=5)
        if ax is axes[1]:
            # Inside the axes, not beyond the right spine, where it would be clipped.
            ax.annotate(f"chance = {chance:.2f}", xy=(0.995, chance),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", fontsize=8, color=INK_3,
                        ha="right", va="bottom")
        ax.set_title(alabel, fontsize=10.5, color=INK, pad=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([SH.get(d, d) for d in DATASETS], fontsize=8.5, color=INK_2)
        ax.set_ylim(0, 1.0)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].set_ylabel("purity of the kept subset", fontsize=9.5, color=INK_2)
    fig.suptitle(title, x=0.007, y=0.975, ha="left", fontsize=13, color=INK)
    fig.text(0.007, 0.905,
             "R=30% contamination · identical poisoned rows for all four methods · "
             "above the dashed line = better than keeping rows at random",
             ha="left", fontsize=8.5, color=INK_2)
    fig.legend(handles=[h for h in handles], loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=9, labelcolor=INK_2, handlelength=1.4,
               columnspacing=2.2, handletextpad=0.6)
    fig.subplots_adjust(left=0.065, right=0.955, top=0.80, bottom=0.175, wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/tables/comparison/detection_comparison.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/comparison"))
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.csv)))
    small = sorted({int(r["size"]) for r in rows if r["is_full_budget"] == "False"})[0]
    print("wrote", figure(rows, False, f"Detection precision — top-{small} kept",
                          a.out / f"detection_top{small}.png"))
    print("wrote", figure(rows, True, "Detection precision — full budget (n − k kept)",
                          a.out / "detection_full_budget.png"))


if __name__ == "__main__":
    main()

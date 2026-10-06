#!/usr/bin/env python
"""Small multiples: the k-fold defence against perfect removal and against no corruption.

One figure per (attack, metric). Five panels, one per dataset, each showing the metric
against the corruption rate R for three contexts, plus the uncorrupted ceiling as a
reference rule:

    poisoned   corrupted, undefended          -- the floor
    defended   corrupted, our R% removed      -- the method
    oracle     corrupted, the REAL rows removed -- perfect removal
    clean      never corrupted                -- dashed rule (constant in R)

Panels keep independent y-scales: the comparison of interest is within a dataset, and
a shared scale would flatten url_unique's CE (~0.09) against heloc's (~0.65). R is
shared and identical across panels.

Colour: categorical slots 1-3 of the validated palette. Validated with
scripts/validate_palette.py (a faithful port of the skill's validator -- node is not
installed here) on the ALL-PAIRS pairlist, which is the one small multiples require:
worst CVD dE 9.2 (deutan, target >=8), worst normal-vision dE 24.0 (floor >=15). Aqua
sits at 2.74:1 on this surface, below the 3:1 contrast gate, so identity never rests on
colour: every series carries its own marker, a figure-level legend, and direct labels in
the first panel.

    python scripts/plot_defense.py
    python scripts/plot_defense.py --metrics ce accuracy
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"

# Slot 1 blue / slot 2 orange / slot 3 aqua, in a fixed order by role.
SERIES = [("poisoned", "#2a78d6", "o", "poisoned (no defence)"),
          ("defended", "#eb6834", "s", "defended (k-fold)"),
          ("oracle", "#1baf7a", "D", "oracle (perfect removal)")]

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies",
            "heloc", "lcld_v2"]
LABEL = {"coil2000_insurance_policies": "coil2000", "credit-approval": "credit-approval",
         "url_unique": "url_unique", "heloc": "heloc", "lcld_v2": "lcld_v2"}
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
TITLE = {"ce": "Cross-entropy", "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
         "balanced_accuracy": "Balanced accuracy", "mcc": "Matthews correlation (MCC)"}
BETTER = {"ce": "lower is better"}
ATTACK_TITLE = {"label-flip-random": "random label flips",
                "hub-flip": "hub label flips (k-NN density)"}


def load(path: Path) -> dict:
    out: dict = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["mean"] in ("", None):
            continue
        key = (r["dataset"], r["attack"], r["metric"], r["context"])
        out[key][int(r["row_percent"])] = (float(r["mean"]), float(r["std"] or 0.0))
    return out


def figure(data: dict, attack: str, metric: str, datasets: list[str], out: Path,
           band: bool) -> Path:
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.05 * len(datasets), 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    if len(datasets) == 1:
        axes = [axes]

    handles: list = []
    for col, (ds, ax) in enumerate(zip(datasets, axes)):
        ax.set_facecolor(SURFACE)
        rs = sorted({r for ctx, _, _, _ in SERIES
                     for r in data.get((ds, attack, metric, ctx), {})})
        if not rs:
            ax.set_visible(False)
            continue

        clean = data.get((ds, attack, metric, "clean"), {})
        if clean:
            lvl = clean[rs[0]][0]
            # The rule goes in the legend, not inline: at low R the clean level sits
            # right on top of the oracle line, so an inline label lands on the oracle's.
            ax.axhline(lvl, color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

        for ctx, colour, marker, legend in SERIES:
            cells = data.get((ds, attack, metric, ctx), {})
            if not cells:
                continue
            present = [r for r in rs if r in cells]
            ys = [cells[r][0] for r in present]
            sd = [cells[r][1] for r in present]
            if band:
                ax.fill_between(present, [y - s for y, s in zip(ys, sd)],
                                [y + s for y, s in zip(ys, sd)],
                                color=colour, alpha=0.14, linewidth=0, zorder=2)
            ln, = ax.plot(present, ys, color=colour, linewidth=2, marker=marker,
                          markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                          label=legend, zorder=3, solid_capstyle="round")
            if col == 0:
                handles.append(ln)
                # Direct labels in the first panel only: repeating them in all five
                # would collide with the marks at this panel width.
                ax.annotate(ctx, xy=(present[-1], ys[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7.5, color=colour,
                            ha="left", va="center")

        ax.set_title(LABEL[ds], fontsize=10, color=INK, pad=6)
        ax.set_xticks(rs)
        ax.set_xlim(rs[0] - 3, rs[-1] + 3)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    note = f"  ·  {BETTER[metric]}" if metric in BETTER else ""
    fig.suptitle(TITLE[metric], x=0.006, y=0.975, ha="left", fontsize=13, color=INK)
    fig.text(0.006, 0.906,
             f"TabPFNv2 · {ATTACK_TITLE.get(attack, attack)} · k-fold defence · "
             f"bands = ±1 std across cells{note}",
             ha="left", fontsize=8.5, color=INK_2)
    fig.text(0.5, 0.055, "R  —  % of context rows corrupted (known to the defender)",
             ha="center", fontsize=9.5, color=INK_2)
    if handles:
        handles.append(Line2D([], [], color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)),
                              label="clean (never corrupted)"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005),
                   ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK_2,
                   handlelength=1.8, columnspacing=2.0, handletextpad=0.5)

    fig.subplots_adjust(left=0.045, right=0.988, top=0.80, bottom=0.185, wspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("results/tables/defense_kfold/defense_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/defense_kfold"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--attacks", nargs="+", default=["label-flip-random", "hub-flip"])
    ap.add_argument("--no-band", action="store_true", help="hide the ±1 std ribbons")
    a = ap.parse_args(argv)

    data = load(a.csv)
    for attack in a.attacks:
        for m in a.metrics:
            p = figure(data, attack, m, a.datasets, a.out / attack / f"{m}.png",
                       band=not a.no_band)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Figures for the gate-valsplit sweep, from results/tables/gate_valsplit/runs.csv
(written by analyze_gate_valsplit.py).

  accuracy_vs_V.png   test accuracy against the validation size V; one panel per
                      dataset (rows) and R (columns), each on its own y-scale.
                      Lines: gating with the 20% validation-test holdout choosing the
                      iteration; gating fitted on all V rows (round 1's rule); the V clean
                      rows alone as the context; the holdout runs' best test checkpoint
                      (a ceiling that peeks at the test set). Grey references: the
                      poisoned context and the oracle gate.
  gain_vs_headroom.png
                      gain of gating (holdout rule) over the clean rows alone against the
                      headroom the clean rows leave (oracle gate - clean rows), one panel
                      per dataset, colour = R, bars = 95% paired bootstrap interval.

Colour: blue / orange / aqua, validated with validate_palette.py on the all-pairs
pairlist. Five datasets in five hues failed the normal-vision floor (orange vs yellow,
orange vs pink), so the second figure facets by dataset instead. Aqua sits at 2.74:1
contrast; the numbers are in results/tables/gate_valsplit/.

Both figures show one validation seed (--seed, default 0). A third,

  gain_vs_V_all_seeds.png
                      the mean gain of gating over the clean rows alone across all
                      validation draws (results/tables/gate_valsplit/seeds.csv), against V,
                      with its 95% interval over test rows; one panel per dataset, colour =
                      R, solid = the 20% holdout rule, dashed = fitted on all V rows,

uses every seed.

    python scripts/plot_gate_valsplit.py [--seed 1]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DATASETS = [("gas-drift", "gas-drift (6 classes)"), ("url_unique", "url_unique (binary)"),
            ("heloc", "heloc (binary)"), ("students_dropout", "students_dropout (3 classes)"),
            ("credit-approval", "credit-approval (binary, 196 test rows)")]
TABLE = Path("results/tables/gate_valsplit/runs.csv")
OUT = Path("results/figures/gate_valsplit")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def load(seed: int):
    rows = [r for r in csv.DictReader(open(TABLE)) if int(r.get("seed", 0)) == seed]
    if not rows:
        raise SystemExit(f"no runs with validation seed {seed} in {TABLE}")
    for r in rows:
        for k, v in r.items():
            if k not in ("dataset", "strategy"):
                r[k] = float(v) if v not in ("", None) else float("nan")
    return {(r["dataset"], int(r["R"]), int(r["V"]), r["strategy"]): r for r in rows}


def accuracy_vs_v(by) -> Path:
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10.5, 15.5), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    for i, (ds, title) in enumerate(DATASETS):
        for j, R in enumerate((10, 20)):
            ax = axes[i, j]
            style(ax)
            Vs = sorted({k[2] for k in by if k[0] == ds and k[1] == R})
            h = [by[(ds, R, V, "holdout")] for V in Vs]
            f = [by[(ds, R, V, "full")] for V in Vs]
            x = list(range(len(Vs)))
            ax.plot(x, [r["poisoned_acc"] for r in h], color=INK_3, lw=1.1, ls=(0, (4, 3)), zorder=2)
            ax.plot(x, [r["oracle_gate_acc"] for r in h], color=INK_3, lw=1.1, ls=(0, (1, 2)), zorder=2)
            ax.plot(x, [r["test_acc_oracle_acc"] for r in h], color=BLUE, lw=1.1, ls=(0, (1, 2)), zorder=3)
            for rows, key, col, mk in ((f, "chosen_acc", ORANGE, "s"), (h, "clean_rows_only_acc", AQUA, "D"),
                                       (h, "chosen_acc", BLUE, "o")):
                ax.plot(x, [r[key] for r in rows], color=col, lw=2, marker=mk, ms=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
            yp, yo = h[-1]["poisoned_acc"], h[-1]["oracle_gate_acc"]
            lo, hi = ax.get_ylim()
            gap = 0.05 * (hi - lo)  # keep the two end labels apart where the lines meet
            if abs(yo - yp) < gap:
                mid = (yo + yp) / 2
                yo, yp = (mid + gap / 2, mid - gap / 2) if yo >= yp else (mid - gap / 2, mid + gap / 2)
            for y, txt in ((yp, "poisoned"), (yo, "oracle")):
                ax.annotate(txt, xy=(len(Vs) - 1, y), xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=INK_3, va="center", ha="left", annotation_clip=False)
            ax.set_xticks(x)
            ax.set_xticklabels([str(V) for V in Vs])
            ax.set_xlim(-0.3, len(Vs) - 1 + 0.75)
            ax.set_title(f"{title}   R={R}%", fontsize=9.5, color=INK, loc="left", pad=4)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("V  (clean validation rows)", fontsize=8.5, color=INK_2)
            if j == 0:
                ax.set_ylabel("test accuracy", fontsize=8.5, color=INK_2)
    fig.suptitle("Gating vs the clean rows alone, by validation size", x=0.01, y=0.997, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.976, "hub-flip-local poisoning; PGD, L-inf step, lr 0.05, 300 updates; one validation "
             "draw per V; each panel has its own y-scale", fontsize=8.5, color=INK_2, ha="left")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=2, ms=5, markeredgecolor=SURFACE,
                      label="gating, iteration chosen on the 20% validation-test holdout"),
               Line2D([], [], color=ORANGE, marker="s", lw=2, ms=5, markeredgecolor=SURFACE,
                      label="gating, fitted on all V rows (round 1's rule)"),
               Line2D([], [], color=AQUA, marker="D", lw=2, ms=5, markeredgecolor=SURFACE,
                      label="the V clean rows alone as the context"),
               Line2D([], [], color=BLUE, lw=1.1, ls=(0, (1, 2)),
                      label="gating, best test checkpoint (ceiling; peeks at the test set)"),
               Line2D([], [], color=INK_3, lw=1.1, ls=(0, (4, 3)), label="poisoned context, no gate"),
               Line2D([], [], color=INK_3, lw=1.1, ls=(0, (1, 2)), label="oracle gate (poisoned rows gated off)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=8.3, labelcolor=INK_2, handlelength=2.6, columnspacing=1.8)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.948, bottom=0.075, hspace=0.42, wspace=0.18)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "accuracy_vs_V.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def gain_vs_headroom(by) -> Path:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.3), dpi=170, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    xlim, ylim = (-0.02, 0.18), (-0.16, 0.17)
    for ax, (ds, title) in zip(axes, DATASETS):
        style(ax)
        ax.axhline(0, color=INK_3, lw=1.0, zorder=1)
        ax.plot((-0.02, 0.17), (-0.02, 0.17), color=INK_3, lw=1.0, ls=(0, (1, 2)), zorder=1)
        for R, col, mk in ((10, BLUE, "o"), (20, ORANGE, "s")):
            pts = [r for k, r in by.items() if k[0] == ds and k[1] == R and k[3] == "holdout"]
            for r in pts:
                xh = r["oracle_gate_acc"] - r["clean_rows_only_acc"]
                ax.plot([xh, xh], [r["gain_ci_lo"], r["gain_ci_hi"]], color=col, lw=1.2, alpha=0.45, zorder=2)
                ax.plot([xh], [r["gain_vs_clean_rows"]], color=col, marker=mk, ms=6, linestyle="none",
                        markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=4)
        ax.set_xlabel("headroom: oracle gate − clean rows alone", fontsize=8, color=INK_2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
    axes[0].set_ylabel("gain: gating (holdout rule) − clean rows alone", fontsize=8, color=INK_2)
    fig.suptitle("Gating only pays where the clean rows leave headroom", x=0.01, y=0.99, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.905, "test accuracy; one point per validation size V; bars: 95% paired bootstrap "
             "interval over test rows (the validation draw is fixed)", fontsize=8.5, color=INK_2, ha="left")
    handles = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=6, markeredgecolor=SURFACE, label="R = 10%"),
               Line2D([], [], color=ORANGE, marker="s", ls="none", ms=6, markeredgecolor=SURFACE, label="R = 20%"),
               Line2D([], [], color=INK_3, lw=1, ls=(0, (1, 2)), label="gain = headroom (all of it recovered)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.25, wspace=0.12)
    path = OUT / "gain_vs_headroom.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def gain_vs_v_seeds(table: Path) -> Path:
    rows = list(csv.DictReader(open(table)))
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(17, 4.4), dpi=170, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    series = [(10, "holdout", BLUE, "o", "-", -0.16), (10, "full", BLUE, "s", (0, (3, 2)), -0.05),
              (20, "holdout", ORANGE, "o", "-", 0.05), (20, "full", ORANGE, "s", (0, (3, 2)), 0.16)]
    for ax, (ds, title) in zip(axes, DATASETS):
        style(ax)
        ax.axhline(0, color=INK_3, lw=1.0, zorder=1)
        Vs = sorted({int(r["V"]) for r in rows if r["dataset"] == ds})
        for R, st, col, mk, ls, dx in series:
            pts = sorted(((int(r["V"]), r) for r in rows if r["dataset"] == ds and int(r["R"]) == R
                          and r["strategy"] == st), key=lambda t: t[0])
            x = [Vs.index(v) + dx for v, _ in pts]
            y = [100 * float(r["gain_mean"]) for _, r in pts]
            lo = [y_ - 100 * float(r["gain_lo"]) for y_, (_, r) in zip(y, pts)]
            hi = [100 * float(r["gain_hi"]) - y_ for y_, (_, r) in zip(y, pts)]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, lw=1.6, marker=mk, ms=5, capsize=0,
                        elinewidth=1.0, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        ax.set_xticks(range(len(Vs)))
        ax.set_xticklabels([str(v) for v in Vs])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=4)
        ax.set_xlabel("V  (clean validation rows)", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("gating − clean rows alone (accuracy points)", fontsize=8.5, color=INK_2)
    fig.suptitle("Where gating beats the clean rows alone", x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.905, "mean over 3 validation draws per point; bars: 95% interval over test rows "
             "(given the draws); above 0 = gating wins", fontsize=8.5, color=INK_2, ha="left")
    handles = [Line2D([], [], color=c, marker=m, ls=l, lw=1.6, ms=5, markeredgecolor=SURFACE,
                      label=f"R = {R}%, " + ("20% holdout chooses the iteration" if st == "holdout"
                                             else "fitted on all V rows"))
               for R, st, c, m, l, _ in series]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
               fontsize=8.3, labelcolor=INK_2, handlelength=2.6)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.25, wspace=0.08)
    path = Path("results/figures/gate_valsplit") / "gain_vs_V_all_seeds.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="validation seed to plot")
    a = ap.parse_args(argv)
    by = load(a.seed)
    global OUT
    OUT = OUT if a.seed == 0 else OUT / f"seed{a.seed}"
    seeds_table = TABLE.parent / "seeds.csv"
    if seeds_table.exists():
        print("wrote", gain_vs_v_seeds(seeds_table))
    for p in (accuracy_vs_v(by), gain_vs_headroom(by)):
        print("wrote", p)


if __name__ == "__main__":
    main()

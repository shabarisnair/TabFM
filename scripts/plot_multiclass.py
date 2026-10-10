#!/usr/bin/env python
"""Metric vs R for the multiclass label-perturbation grid: one figure per metric.

Two panels per figure (students_dropout, gas-drift), each with independent y-scales --
the clean baselines differ too much to share one (accuracy 0.816 vs 0.996). Three lines
per panel, one per attack: the mean over runs, with a +-1 std band for the two
stochastic attacks (10 runs each). hub-flip-local is deterministic, so it has no band.
hub-flip-random is dashed because it poisons the SAME rows as hub-flip-local and differs
only in the target labels.

The dashed grey rule is the metric's NO-SKILL level (see make_multiclass_tables.py):
the score of a model that ignores the features. R=0 is the clean baseline, shared by
all three attacks.

Colour: categorical slots blue / orange / aqua, validated with validate_palette.py on
the all-pairs pairlist (worst CVD dE 9.2, worst normal-vision dE 24.0). Aqua sits at
2.74:1 contrast, so every series also has its own marker and a direct label, and the
numbers are in results/tables/multiclass/.

    python scripts/plot_multiclass.py
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

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
STYLE = {  # attack: (colour, marker, linestyle, short label, legend label)
    "label-flip-bernoulli": ("#2a78d6", "o", "-", "Bernoulli",
                             "random flip (each row w.p. R, random other class)"),
    "hub-flip-local": ("#eb6834", "s", "-", "hub-local",
                       "hub flip, locally closest other class"),
    "hub-flip-random": ("#1baf7a", "D", (0, (4, 2)), "hub-random",
                        "hub flip, random other class (same rows)"),
}
DATASETS = [("students_dropout", "students_dropout  (3 classes)"),
            ("gas-drift", "gas-drift  (6 classes)")]
TITLE = {"ce": "Cross-entropy", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
         "mcc": "Matthews correlation (multiclass)", "roc_auc": "ROC-AUC (one-vs-rest, macro)"}
LOWER_BETTER = {"ce"}


def load(path: Path):
    data = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        data[(r["dataset"], r["attack"], r["metric"])][int(r["row_percent"])] = r
    return data


def end_labels(ax, items, frac=0.06):
    """Direct labels at the right edge, nudged apart so none collide. Ink, not series colour."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * frac
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - (hi - gap * 0.3)
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, text) in zip(ys, items):
        ax.annotate(text, xy=(1.015, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8, color=INK_2)


def figure(data, metric: str, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ds, dlabel) in zip(axes, DATASETS):
        ax.set_facecolor(SURFACE)
        ends, rs_all, ns = [], set(), None
        for atk, (col, mk, ls, short, _) in STYLE.items():
            cells = data.get((ds, atk, metric), {})
            if not cells:
                continue
            rs = sorted(cells)
            rs_all |= set(rs)
            ys = [float(cells[r]["poisoned_mean"]) for r in rs]
            sd = [float(cells[r]["poisoned_std"]) for r in rs]
            ns = float(cells[rs[0]]["no_skill"])
            if any(s > 0 for s in sd):
                ax.fill_between(rs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)],
                                color=col, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(rs, ys, color=col, linestyle=ls, linewidth=2, marker=mk, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3, solid_capstyle="round")
            ends.append((ys[-1], short))
        if ns is not None:
            ax.axhline(ns, color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"no skill {ns:.3g}", xy=(0.005, ns), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_3,
                        ha="left", va="bottom")
        # keep the no-skill rule in view: it is what tells you how far an attack got
        lo, hi = ax.get_ylim()
        if ns is not None:
            pad = 0.04 * (max(hi, ns) - min(lo, ns))
            ax.set_ylim(min(lo, ns) - pad, max(hi, ns) + pad)
        rs_sorted = sorted(rs_all)
        ax.set_xticks(rs_sorted)
        ax.set_xlim(rs_sorted[0] - 1.5, rs_sorted[-1] + 1.5)
        ax.set_xlabel("R  —  % of context rows perturbed", fontsize=9, color=INK_2)
        ax.set_title(dlabel, fontsize=10.5, color=INK, pad=6)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        end_labels(ax, ends)

    note = "lower is better" if metric in LOWER_BETTER else "higher is better"
    fig.suptitle(TITLE[metric], x=0.007, y=0.975, ha="left", fontsize=13, color=INK)
    fig.text(0.007, 0.905,
             f"TabPFNv2 · label perturbation of the context · mean ± 1 std over 10 runs "
             f"(hub-local is deterministic) · R=0 is the clean model · {note}",
             ha="left", fontsize=8.5, color=INK_2)
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, linewidth=2, markersize=5,
                      markeredgecolor=SURFACE, label=lab) for c, m, ls, _, lab in STYLE.values()]
    handles.append(Line2D([], [], color=INK_3, linestyle=(0, (4, 3)), linewidth=1.2,
                          label="no-skill level"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4,
               frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=2.2,
               columnspacing=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.055, right=0.925, top=0.80, bottom=0.215, wspace=0.27)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/tables/multiclass/multiclass_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/multiclass"))
    ap.add_argument("--metrics", nargs="+", default=list(TITLE))
    a = ap.parse_args(argv)
    data = load(a.csv)
    for m in a.metrics:
        print("wrote", figure(data, m, a.out / f"{m}.png"))


if __name__ == "__main__":
    main()

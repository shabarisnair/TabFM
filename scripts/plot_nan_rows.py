#!/usr/bin/env python
"""Accuracy and cross-entropy vs number of NaN rows added (nan_rows_experiment.py).

Two panels: accuracy (linear -- shows the collapse) and cross-entropy (log -- shows the
small degradations the accuracy axis flattens). x is log2 of the number of rows added,
with the clean context (0 rows) set apart at the left. The dashed grey rule is the
no-skill level: the test set's majority-class rate for accuracy, the entropy of its class
prior for cross-entropy.

Colour: blue / orange / aqua / magenta, validated with validate_palette.py on the
all-pairs pairlist (worst CVD dE 9.2, worst normal-vision dE 19.5). Aqua sits at 2.74:1
contrast, so every series also has its own marker and a direct label, and the numbers
are in table.txt next to the results.

    python scripts/plot_nan_rows.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
STYLE = {  # variant: (colour, marker, linestyle, short label, legend label)
    "pipeline": ("#eb6834", "o", "-", "pipeline",
                 "our pipeline, rows as specified (NaN features + NaN label)"),
    "pipeline-nanaware": ("#c4378f", "s", "-", "NaN-aware z-norm",
                          "pipeline with NaN-aware z-norm, same rows"),
    "pipeline-nanaware-labelled": ("#2a78d6", "D", "-", "NaN-aware, labelled",
                                   "pipeline with NaN-aware z-norm, NaN features + real label"),
    "standard-api-labelled": ("#1baf7a", "^", (0, (4, 2)), "standard API, labelled",
                              "standard TabPFN API, NaN features + real label (it rejects NaN labels)"),
}
ZERO_X = -1.6  # where the clean context (0 rows added) sits on the log2 axis


def xpos(n: int) -> float:
    return ZERO_X if n == 0 else math.log2(n)


def end_labels(ax, items, frac=0.07, log=False):
    """Direct labels at the right edge, nudged apart so none collide. Ink, not series colour."""
    tr = (math.log10, lambda v: 10 ** v) if log else (lambda v: v, lambda v: v)
    lo, hi = (tr[0](v) for v in ax.get_ylim())
    gap = (hi - lo) * frac
    items = sorted(((tr[0](y), t) for y, t in items), key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - (hi - gap * 0.3)
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, text) in zip(ys, items):
        ax.annotate(text, xy=(1.015, tr[1](y)), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8, color=INK_2)


def panel(ax, s, metric, no_skill, log=False):
    ax.set_facecolor(SURFACE)
    ns = s["n_nan"]
    ends = []
    for name, (col, mk, ls, short, _) in STYLE.items():
        pts = sorted((r["n_nan"], r[metric]) for r in s["rows"] if r["variant"] == name)
        if not pts:
            continue
        xs, ys = [xpos(n) for n, _ in pts], [v for _, v in pts]
        ax.plot(xs[1:], ys[1:], color=col, linestyle=ls, linewidth=2, marker=mk, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3, solid_capstyle="round")
        # the clean point stands apart: joined by a thin dotted segment, not the series line
        ax.plot(xs[:2], ys[:2], color=col, linestyle=(0, (1, 2)), linewidth=1.1, zorder=2)
        ax.plot(xs[:1], ys[:1], color=col, marker=mk, markersize=5, linestyle="none",
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
        ends.append((ys[-1], short))
    ax.axhline(no_skill, color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"no skill {no_skill:.3g}", xy=(0.995, no_skill), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_3,
                ha="right", va="bottom")
    if log:
        ax.set_yscale("log")
    ax.axvline((ZERO_X + 0) / 2, color=GRID, linewidth=1.0, zorder=0)
    ax.set_xticks([xpos(n) for n in ns])
    ax.set_xticklabels([str(n) for n in ns], fontsize=7.5)
    ax.set_xlim(ZERO_X - 0.6, xpos(ns[-1]) + 0.5)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel(f"NaN rows added to the {s['n_base']}-row context  (log scale; "
                  f"{ns[-1]} = half the context)", fontsize=8.5, color=INK_2)
    end_labels(ax, ends, log=log)


def table(s) -> str:
    cols = ["accuracy", "balanced_accuracy", "ce", "mcc", "roc_auc", "top_predicted_share"]
    head = {"accuracy": "acc", "balanced_accuracy": "bal.acc", "ce": "CE", "mcc": "MCC",
            "roc_auc": "AUC", "top_predicted_share": "top-class"}
    out = [f"{s['dataset']}: {s['n_base']}-row context, {s['n_features']} features, "
           f"{s['n_classes']} classes; test {sum(s['class_counts_test'])} rows; "
           f"no-skill accuracy {s['no_skill_accuracy']:.3f}",
           "top-class = share of test rows given the single most-predicted class",
           f"standard API with a NaN label: {s['standard_api_with_nan_label']}"]
    for name in s["variants"]:
        out.append(f"\n  {name}  --  {STYLE[name][4]}")
        hdr = f"    {'NaN rows':>8s} {'share':>6s}" + "".join(f"{head[c]:>10s}" for c in cols)
        out += [hdr, "    " + "-" * (len(hdr) - 4)]
        for r in sorted((r for r in s["rows"] if r["variant"] == name), key=lambda r: r["n_nan"]):
            out.append(f"    {r['n_nan']:>8d} {r['nan_share']:>6.1%}"
                       + "".join(f"{r[c]:>10.4f}" for c in cols))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gas-drift")
    ap.add_argument("--root", type=Path, default=Path("results/nan_rows"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/nan_rows"))
    a = ap.parse_args(argv)
    s = json.loads((a.root / a.dataset / "summary.json").read_text())

    text = table(s)
    print(text)
    (a.root / a.dataset / "table.txt").write_text(text + "\n")

    counts = s["class_counts_test"]
    prior = [c / sum(counts) for c in counts]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    panel(axes[0], s, "accuracy", s["no_skill_accuracy"])
    axes[0].set_title("Accuracy  (higher is better)", fontsize=10.5, color=INK, pad=6)
    panel(axes[1], s, "ce", -sum(p * math.log(p) for p in prior if p > 0), log=True)
    axes[1].set_title("Cross-entropy  (lower is better, log scale)", fontsize=10.5, color=INK, pad=6)

    fig.suptitle(f"Adding NaN rows to a TabPFNv2 context  ·  {s['dataset']}",
                 x=0.007, y=0.975, ha="left", fontsize=13, color=INK)
    fig.text(0.007, 0.912,
             f"{s['n_base']}-row stratified context, {s['n_classes']} classes, scored on "
             f"{sum(counts)} held-out test rows  ·  one deterministic fit per point  ·  "
             "the leftmost point is the clean context",
             ha="left", fontsize=8.5, color=INK_2)
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, linewidth=2, markersize=5,
                      markeredgecolor=SURFACE, label=lab) for c, m, ls, _, lab in STYLE.values()]
    handles.append(Line2D([], [], color=INK_3, linestyle=(0, (4, 3)), linewidth=1.2,
                          label="no-skill level"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=2.4,
               columnspacing=2.0, handletextpad=0.5)
    fig.subplots_adjust(left=0.05, right=0.885, top=0.81, bottom=0.27, wspace=0.36)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.dataset}_nan_rows.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path, "and", a.root / a.dataset / "table.txt")


if __name__ == "__main__":
    main()

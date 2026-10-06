#!/usr/bin/env python
"""Detection quality of the outlier filter: metric vs R, one line per dataset.

One figure per (attack, metric), five dataset lines each, written to
``<out>/<attack>/<metric>.png`` -- the same split the defence figures use. Colour and
marker are imported from plot_results.py so a dataset looks the same here as in every
other figure in the project.

Reference lines:
  precision / recall   the dashed line is CHANCE, k/n = R/100: what a filter that picks
                       rows at random would score. Precision is only evidence above it.
  accuracy             chance is 1 - 2(R/100)(1 - R/100). Note accuracy is a rescaling of
                       precision, not independent of it (see make_detection_tables.py).
  ranking AUC          0.5 = no signal. This is the budget-free measure.

    python scripts/plot_detection.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_results import COLOUR, LABEL, MARKER  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies",
            "heloc", "lcld_v2"]
ATTACKS = [("label-flip-random", "random label flips"),
           ("hub-flip", "hub label flips (k-NN density)")]
TITLE = {"precision": "Detection precision", "recall": "Detection recall",
         "accuracy": "Detection accuracy", "f1": "Detection F1",
         "detection_roc_auc": "Ranking AUC (budget-free)",
         "false_removal_rate": "False removal rate",
         "recall_by_true_class_0": "Recall on rows whose true label was 0",
         "recall_by_true_class_1": "Recall on rows whose true label was 1"}
NOTE = {"precision": "dashed = chance (a filter picking rows at random)",
        "recall": "dashed = chance",
        "f1": "dashed = chance",
        "accuracy": "dashed = chance  ·  NB: accuracy = 1 - 2(R/100)(1 - precision), not independent of precision",
        "detection_roc_auc": "dashed = 0.5, no signal",
        "false_removal_rate": "share of removed rows that were actually clean",
        "recall_by_true_class_0": "dashed = chance",
        "recall_by_true_class_1": "dashed = chance"}


def chance(metric: str, r: int) -> float | None:
    p = r / 100.0
    if metric in ("precision", "recall", "f1", "recall_by_true_class_0",
                  "recall_by_true_class_1"):
        return p
    if metric == "accuracy":
        return 1 - 2 * p * (1 - p)
    if metric == "detection_roc_auc":
        return 0.5
    return None


def load(path: Path) -> dict:
    out: dict = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["mean"] in ("", None):
            continue
        out[(r["dataset"], r["attack"], r["metric"])][int(r["row_percent"])] = (
            float(r["mean"]), float(r["std"] or 0.0))
    return out


def figure(data: dict, attack: str, alabel: str, metric: str, out: Path,
           band: bool) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    handles: list = []
    rs_all: set = set()
    for ds in DATASETS:
        cells = data.get((ds, attack, metric), {})
        if not cells:
            continue
        rs = sorted(cells)
        rs_all |= set(rs)
        ys = [cells[r][0] for r in rs]
        sd = [cells[r][1] for r in rs]
        if band:
            ax.fill_between(rs, [y - s_ for y, s_ in zip(ys, sd)],
                            [y + s_ for y, s_ in zip(ys, sd)],
                            color=COLOUR[ds], alpha=0.13, linewidth=0, zorder=2)
        ln, = ax.plot(rs, ys, color=COLOUR[ds], linewidth=2, marker=MARKER[ds],
                      markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.1,
                      label=LABEL[ds], zorder=3, solid_capstyle="round")
        handles.append(ln)
    if not handles:
        plt.close(fig)
        raise SystemExit(f"no data for {attack} / {metric}")

    rs_sorted = sorted(rs_all)
    ch = [chance(metric, r) for r in rs_sorted]
    if all(c is not None for c in ch):
        ax.plot(rs_sorted, ch, color=INK_3, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    ax.set_xticks(rs_sorted)
    ax.set_xlim(rs_sorted[0] - 3, rs_sorted[-1] + 3)
    ax.set_xlabel("R  —  % of context rows corrupted (= the removal budget)",
                  fontsize=9.5, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.set_title(TITLE.get(metric, metric), fontsize=12.5, color=INK, loc="left", pad=22)
    ax.annotate(f"{alabel}  ·  {NOTE.get(metric, '')}", xy=(0, 1.025),
                xycoords="axes fraction", fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.8,
              borderpad=0.2, ncol=len(handles), loc="upper center",
              bbox_to_anchor=(0.5, -0.145), columnspacing=1.4, handletextpad=0.5)

    fig.subplots_adjust(left=0.10, right=0.975, top=0.845, bottom=0.235)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("results/tables/defense_kfold/detection_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/defense_kfold/detection"))
    ap.add_argument("--metrics", nargs="+",
                    default=["precision", "accuracy", "detection_roc_auc",
                             "false_removal_rate", "recall_by_true_class_0",
                             "recall_by_true_class_1"])
    ap.add_argument("--no-band", action="store_true")
    a = ap.parse_args(argv)

    data = load(a.csv)
    for attack, alabel in ATTACKS:
        for m in a.metrics:
            print("wrote " + str(figure(data, attack, alabel, m,
                                        a.out / attack / f"{m}.png", band=not a.no_band)))


if __name__ == "__main__":
    main()

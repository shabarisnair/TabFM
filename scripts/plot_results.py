#!/usr/bin/env python
"""One line chart per metric: metric vs R%, one line per dataset.

Reads ``results/tables/main/main_results.csv`` (written by make_main_tables.py) and
draws the poisoned mean at each budget R. R=0 is the clean baseline -- flipping or
perturbing 0% of the rows *is* the clean context -- so every line starts from its own
dataset's baseline and the degradation is read directly off the curve.

Colour and marker shape are keyed to the dataset (see COLOUR/MARKER), so a dataset
looks the same in every figure regardless of how many series share it, and identity is
recoverable without colour -- shape plus a direct end-label.

    python scripts/plot_results.py                          # label-flip-random
    python scripts/plot_results.py --attack label-flip --std
    python scripts/plot_results.py --attack all             # every attack
    python scripts/plot_results.py --csv results/tables/flip_sweep_hub/hub_flip_results.csv \
        --attack hub-flip --max-r 40 --out results/figures/flip_sweep
"""

from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour is keyed to the DATASET, not to position, so a dataset looks the same whether
# a figure carries three series or five.
#
# Validated with OKLab dE x100 under normal vision plus Machado CVD sims at severity
# 1.0. Across the five sweep datasets every pair clears the normal-vision floor
# (min dE 15.6, aqua/green), but three pairs fall under the CVD target of 8:
# blue/green tritan 7.6, aqua/pink deutan 6.1, amber/pink tritan 5.8. No 5-subset of
# the palette clears all-pairs CVD, so identity is never carried by colour alone --
# every series also gets its own MARKER shape and a direct end-label, and the weak
# pairs are the ones given the most dissimilar shapes. Amber, pink and aqua sit below
# 3:1 contrast on this surface, which is the other reason for the end-labels.
COLOUR = {"credit-approval": "#2a78d6",              # blue
          "url_unique": "#1baf7a",                   # aqua
          "coil2000_insurance_policies": "#eda100",  # amber
          "heloc": "#e87ba4",                        # pink
          "lcld_v2": "#008300",                      # green
          "wids": "#eb6834"}                         # orange
MARKER = {"credit-approval": "o", "url_unique": "s",
          "coil2000_insurance_policies": "^", "heloc": "X",
          "lcld_v2": "D", "wids": "v"}
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies",
            "heloc", "lcld_v2", "wids"]
LABEL = {"coil2000_insurance_policies": "coil2000", "lcld_v2": "lcld_v2",
         "url_unique": "url_unique", "wids": "wids",
         "heloc": "heloc", "credit-approval": "credit-approval"}
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
TITLE = {"ce": "Cross-entropy", "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
         "balanced_accuracy": "Balanced accuracy", "mcc": "Matthews correlation (MCC)"}
ATTACK_TITLE = {"label-flip-random": "random label flips (test-agnostic)",
                "hub-flip": "hub label flips (test-agnostic, k-NN density)",
                "label-flip": "label flips, GA (test-optimised)",
                "x-capgd": "X_train CAPGD (test-optimised)"}
# A reference line only where the metric has a meaningful "no skill" level.
NO_SKILL = {"roc_auc": (0.5, "random ranking"), "mcc": (0.0, "no correlation"),
            "balanced_accuracy": (0.5, "chance")}


def load(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    out: dict = collections.defaultdict(dict)
    for r in rows:
        out[(r["dataset"], r["attack"], r["metric"])][int(r["row_percent"])] = r
    return out


def place_labels(ax, items, gap_frac=0.055):
    """Direct end-labels, nudged apart so they never collide."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    items = sorted(items, key=lambda t: t[0])          # by y, bottom up
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - (hi - gap * 0.4)
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, text, colour) in zip(ys, items):
        ax.annotate(text, xy=(1.012, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8.5, color=colour)


def figure(data: dict, attack: str, metric: str, rs: list[int], show_std: bool, out: Path,
           series: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ends = []
    for ds in (series or DATASETS):
        cells = data.get((ds, attack, metric))
        if not cells:
            continue
        colour, marker = COLOUR[ds], MARKER[ds]
        present = [r for r in rs if r in cells]
        clean = float(cells[present[0]]["clean"])       # same baseline at every R
        xs = [0] + present
        ys = [clean] + [float(cells[r]["poisoned_mean"]) for r in present]
        ax.plot(xs, ys, color=colour, linewidth=2, marker=marker,
                markersize=5.5 if len(rs) <= 12 else 4.4,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[ds],
                zorder=3, solid_capstyle="round")
        if show_std:
            sd = [0.0] + [float(cells[r]["poisoned_std"]) for r in present]
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                            [y + s for y, s in zip(ys, sd)],
                            color=colour, alpha=0.13, linewidth=0, zorder=2)
        ends.append((ys[-1], LABEL[ds], colour))

    # The reference level is explained in the subtitle, not next to the line: a flat
    # series often sits exactly on it (coil2000 at MCC 0 / bacc 0.5) and any inline
    # label would land on top of that series.
    ref = ""
    if metric in NO_SKILL:
        level, note = NO_SKILL[metric]
        ax.axhline(level, color=INK_3, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ref = f"  ·  dashed = {note} ({level:g})"

    ax.set_xlabel("R  —  % of context rows attacked", fontsize=9.5, color=INK_2)
    ax.set_title(TITLE[metric], fontsize=12.5, color=INK, loc="left", pad=14)
    ax.annotate(f"TabPFNv2 · {ATTACK_TITLE.get(attack, attack)}  ·  R=0 = clean{ref}",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=8.5, color=INK_2, va="bottom")

    # Linear in R: the budget is a percentage and the crossover near R=32-50 should be
    # read at true spacing. Only the R values are ticked -- a 0 tick collides with 1,
    # and the subtitle already says the leftmost point is the clean baseline.
    # A dense sweep (R = 2,4,...,50) would put 25 labels on the axis. Tick the round
    # values plus the endpoints; every R is still drawn as a marker.
    ticks = rs if len(rs) <= 12 else sorted({rs[0], rs[-1]} | {r for r in rs if r % 10 == 0})
    ax.set_xticks(ticks)
    ax.set_xlim(-2.5, max(rs) + 2.5)
    ax.annotate("clean", xy=(0, 0), xycoords=("data", "axes fraction"), xytext=(0, -13),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=INK_3)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # With five series an in-axes legend has nowhere to sit: "best" lands it on the
    # curves or on the dashed reference line. One row under the plot never collides.
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.8,
                    borderpad=0.2, ncol=max(len(ends), 1), loc="upper center",
                    bbox_to_anchor=(0.5, -0.155), columnspacing=1.4, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    place_labels(ax, ends)

    fig.subplots_adjust(left=0.085, right=0.855, top=0.86, bottom=0.235)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/tables/main/main_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--attack", default="label-flip-random",
                    help="attack to plot, or 'all'")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--std", action="store_true", help="shade +-1 std across runs")
    ap.add_argument("--series", nargs="+", default=None,
                    help="datasets to plot (default: every one in COLOUR). Colour and marker "
                         "follow the dataset, so restricting this does not recolour anything.")
    ap.add_argument("--max-r", type=int, default=None,
                    help="drop R above this, so series that ran a shorter ladder do not "
                         "make the others look truncated")
    a = ap.parse_args(argv)

    data = load(a.csv)
    unknown = [d for d in (a.series or []) if d not in COLOUR]
    if unknown:
        raise SystemExit(f"--series has no colour for {unknown}; add it to COLOUR/MARKER "
                         f"(known: {', '.join(COLOUR)})")
    series = a.series or DATASETS
    attacks = sorted({k[1] for k in data}) if a.attack == "all" else [a.attack]
    for attack in attacks:
        # R=0, when a sweep includes it, IS the clean baseline -- and every line is
        # already anchored at x=0 from the `clean` column. Drop it to avoid a duplicate.
        rs = sorted({r for k, v in data.items() if k[1] == attack for r in v
                     if r > 0 and (a.max_r is None or r <= a.max_r)})
        if not rs:
            print(f"[skip] no rows for attack {attack!r} in {a.csv}")
            continue
        for m in a.metrics:
            p = figure(data, attack, m, rs, a.std, a.out / attack / f"{m}.png", series)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()

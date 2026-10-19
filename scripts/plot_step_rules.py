#!/usr/bin/env python
"""Figure: how much each step-size rule gains over the V-filter, draw by draw (night 2, E19).

For every hub-flip-local, R = 20%, stratified draw with all of them (5 datasets, V = 50-500 at seed 0
and V = 50 / 200 / 500 at seeds 1-2), the test-accuracy difference to the V-filter of the greedy with
1% / 5% / 10% / 20% of the context per step, 1% steps with the class prior, and the noise-adaptive rule
(1% + prior if the 5-fold CV accuracy of TabPFN on V >= 0.8, else 10%). One dot per draw, coloured by
dataset; the black bar is the mean.

    python scripts/plot_step_rules.py   # -> results/plots/iterative_filter/step_rules.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATASETS = (("gas-drift", "#2a78d6"), ("url_unique", "#eb6834"), ("heloc", "#1baf7a"),
            ("students_dropout", "#eda100"), ("credit-approval", "#e87ba4"))
RULES = (("grow:pct1", "1% steps"), ("grow:pct1:prior", "1% + class prior"), ("grow:pct5", "5% steps"),
         ("grow:pct10", "10% steps"), ("grow:pct20", "20% steps"), ("adaptive", "1% + prior if V looks clean,\nelse 10%"))
INK, MUTED, GRID, SURFACE = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"


def main() -> None:
    rows = {}
    for f in sorted(Path("results/iterative_filter").glob("e*/rows_*.csv")):
        for r in csv.DictReader(open(f)):
            if (r["attack"], r["form"], float(r["R"])) != ("hub-flip-local", "stratified", 20.0):
                continue
            k = (r["dataset"], int(r["V"]), int(r["seed"]))
            rows.setdefault(k, {}).setdefault(r["variant"], 100 * float(r["acc"]))
    vn = {r["draw"]: float(r["v_cv_acc"]) for r in csv.DictReader(open("results/iterative_filter/vnoise.csv"))}
    draws = [k for k, r in rows.items() if (k[2] == 0 or k[1] in (50, 200, 500))
             and all(v in r for v in ("oneshot", "grow:pct1", "grow:pct1:prior", "grow:pct5", "grow:pct10", "grow:pct20"))]
    for k in draws:
        acc_v = vn.get(f"{k[0]}_hub-flip-local_stratified_R20_V{k[1]}s{k[2]}")
        if acc_v is not None:
            rows[k]["adaptive"] = rows[k]["grow:pct1:prior" if acc_v >= 0.8 else "grow:pct10"]
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    ax.axhline(0, color="#8a8985", lw=1.2, zorder=1)
    rng = np.random.default_rng(0)
    for i, (key, label) in enumerate(RULES):
        ks = [k for k in draws if key in rows[k]]
        d = np.array([rows[k][key] - rows[k]["oneshot"] for k in ks])
        for ds, color in DATASETS:
            sel = [j for j, k in enumerate(ks) if k[0] == ds]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(sel)), d[sel], s=26, color=color, edgecolor=SURFACE, lw=0.8,
                       zorder=3, label=ds if i == 0 else None)
        ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=INK, lw=2.4, zorder=4)
        ax.annotate(f"mean {d.mean():+.2f}\n{int((d > 1e-9).sum())} up, {int((d < -1e-9).sum())} down", (i, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -2), textcoords="offset points", ha="center", va="top",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(len(RULES)), [lbl for _, lbl in RULES], fontsize=9, color=INK)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))  # room for the mean labels above the dots
    ax.set_ylabel("test accuracy minus the V-filter (points)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK, ncol=5)
    ax.set_title(f"Greedy growth vs the one-shot V-filter, by rows moved per step ({len(draws)} draws: hub-flip-local, R = 20%, "
                 "V = 50-500, seeds 0-2)", loc="left", fontsize=10.5, color=INK)
    out = Path("results/plots/iterative_filter/step_rules.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()

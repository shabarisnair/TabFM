#!/usr/bin/env python
"""Plots of the V + P benchmark (scripts/bench_vp.py; docs/bench_vp.md): one figure per V size (c = 1, 3, 5 rows per class).

Each figure has a row per poisoning (rand45, hub30) and a column per test metric (ROC-AUC, balanced accuracy, log loss).
The x axis lists the methods, a line per dataset gives its mean over the seeds, and a black line is the mean of the
6 dataset means. A hollow marker means fewer than the 5 seeds have finished (for the mean: a dataset mean behind it
is incomplete or missing). The same numbers go to a markdown file, one table per V size, poisoning and metric.

    python scripts/plot_bench_vp.py [--root results/bench_vp] [--plots results/plots/bench_vp] [--tables results/tables/bench_vp]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

METHODS = [("v_only", "1 V only"), ("vfilter_R", "2 V-filter, R%"), ("vfilter_thr", "3 V-filter, p < 0.5/K"),
           ("greedy_R", "4 greedy to (1-R)%"), ("greedy_R10", "5 greedy, +10 steps"), ("sigmoid", "6 sigmoid soft, it 10"),
           ("m1_last", "7 M1, last it"), ("m1_sel", "7 M1, held-out choice"), ("knn_shapley", "8 KNN-Shapley"),
           ("ieg", "8 IEG, top R%"), ("greedy_thr", "11 greedy, while p > 0.5"), ("vrepair", "12 V-repair, p > 0.9"),
           ("vrepair_05", "12 V-repair, p > 0.5"), ("grepair", "13 greedy repair, 5% of P"), ("v_plus_p", "9 V + P (no cleaning)"),
           ("oracle", "10 oracle (V + clean P)")]
DATASETS = [("gas-drift", "gas-drift"), ("students_dropout", "students"), ("heloc", "heloc"),
            ("ecommerce_shipping", "ecommerce"), ("is_this_a_good_customer", "good_customer"),
            ("coil2000_insurance_policies", "coil2000")]
ATTACKS = [("rand45", "rand45: random flips, R = 45%"), ("hub30", "hub30: dense rows flipped, R = 30%")]
SEEDS = 5
# the dataviz reference palette's categorical slots 1-6 in their fixed order (as in plot_bench_tinyv.py); the mean in
# the primary text ink
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
MEAN_COLOR = "#0b0b0b"
BG, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
METRICS = [("auc", "ROC-AUC", 1, "{:.3f}", True), ("bacc", "balanced accuracy (%)", 100, "{:.1f}", True),
           ("ce", "log loss (lower is better)", 1, "{:.3f}", False)]


def parse_tag(tag: str) -> tuple[str, str, int, int]:
    head, seed = tag.rsplit("_s", 1)
    head, c = head.rsplit("_c", 1)
    ds, attack = head.rsplit("_", 1)
    return ds, attack, int(c), int(seed)


def load(root: Path) -> dict:
    """(dataset, attack, c, method) -> {metric: values over the finished seeds}."""
    vals: dict = {}
    for f in sorted((root / "runs").glob("*/*.json")):
        ds, attack, c, _ = parse_tag(f.parent.name)
        for r in json.loads(f.read_text())["rows"]:
            cell = vals.setdefault((ds, attack, c, r["name"]), {m: [] for m, *_ in METRICS})
            for m, *_ in METRICS:
                cell[m].append(r["test"][m])
    return vals


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_VP_ROOT", "results/bench_vp")))
    ap.add_argument("--plots", type=Path, default=Path("results/plots/bench_vp"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables/bench_vp"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vals = load(a.root)
    a.plots.mkdir(parents=True, exist_ok=True)
    a.tables.mkdir(parents=True, exist_ok=True)
    md = ["# V + P benchmark: every method on every dataset, per V size", "",
          f"Each cell: the mean over the finished seeds (of {SEEDS}); `(n)` after a value: only n seeds finished. "
          "mean = the mean of the 6 dataset means (`*`: one of them incomplete or missing). Bold: the best entry of the "
          "column, the oracle excluded.", ""]
    x = np.arange(len(METHODS))
    for c in (1, 3, 5):
        fig, axes = plt.subplots(len(ATTACKS), len(METRICS), figsize=(24, 13), facecolor=BG)
        md += [f"## V = {c} row{'s' if c > 1 else ''} per class", ""]
        for (attack, att_title), ax_row in zip(ATTACKS, axes):
            for ax, (m, title, scale, fmt, higher) in zip(ax_row, METRICS):
                cells = [[vals.get((ds, attack, c, e), {}).get(m, []) for e, _ in METHODS] for ds, _ in DATASETS]
                table = np.array([[scale * np.mean(v) if v else np.nan for v in row] for row in cells])
                count = np.array([[len(v) for v in row] for row in cells])
                with np.errstate(all="ignore"):
                    mean = np.nanmean(table, axis=0)
                full_mean = (count >= SEEDS).all(axis=0)
                for i, (_, name) in enumerate(DATASETS):
                    ax.plot(x, table[i], color=COLORS[i], linewidth=1.5, alpha=0.85, zorder=2)
                    for done, face in ((count[i] >= SEEDS, COLORS[i]), (count[i] < SEEDS, BG)):
                        ax.plot(x[done], table[i][done], linestyle="none", marker=MARKERS[i], markersize=6.5,
                                color=COLORS[i], markerfacecolor=face, markeredgewidth=1.4, zorder=3)
                ax.plot(x, mean, color=MEAN_COLOR, linewidth=2.6, zorder=5)
                for done, face in ((full_mean, MEAN_COLOR), (~full_mean, BG)):
                    ax.plot(x[done], mean[done], linestyle="none", marker="*", markersize=13, color=MEAN_COLOR,
                            markerfacecolor=face, markeredgewidth=1.4, zorder=6)
                ax.set_facecolor(BG)
                ax.set_title(f"{title} | {att_title}", fontsize=11.5, color=MEAN_COLOR, loc="left")
                ax.set_xticks(x)
                ax.set_xticklabels([name for _, name in METHODS], rotation=40, ha="right", fontsize=9, color=INK2)
                ax.tick_params(axis="y", labelsize=9, colors=INK2)
                if m == "ce":
                    ax.set_yscale("log")  # the oracle on gas-drift sits near 0.02, the worst near 4
                    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6]))
                    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.grid(axis="y", which="both", color=GRID, linewidth=0.7)
                ax.grid(axis="x", color="#efeeea", linewidth=0.5)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color("#bdbcb6")
                # the table of this panel
                cols = [n for _, n in DATASETS] + ["mean"]
                md += [f"### {att_title}: {title}", "", "| method | " + " | ".join(cols) + " |",
                       "|---|" + "---|" * len(cols)]
                full = np.vstack([table, mean])
                best = []
                for row in full:
                    ok = [v for v, (e, _) in zip(row, METHODS) if np.isfinite(v) and e != "oracle"]
                    best.append((max(ok) if higher else min(ok)) if ok else None)
                for j, (e, name) in enumerate(METHODS):
                    out = []
                    for i in range(len(full)):
                        v = full[i, j]
                        s = "–" if not np.isfinite(v) else fmt.format(v)
                        if np.isfinite(v) and e != "oracle" and v == best[i]:
                            s = f"**{s}**"
                        if i < len(DATASETS) and 0 < count[i, j] < SEEDS:
                            s += f" ({count[i, j]})"
                        if i == len(DATASETS) and not full_mean[j]:
                            s += " *"
                        out.append(s)
                    md.append(f"| {name} | " + " | ".join(out) + " |")
                md.append("")
        handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], linewidth=1.5, label=name)
                   for i, (_, name) in enumerate(DATASETS)]
        handles += [Line2D([], [], color=MEAN_COLOR, marker="*", markersize=12, linewidth=2.6, label="mean of the 6 datasets"),
                    Line2D([], [], color=INK2, marker="o", markerfacecolor=BG, linestyle="none",
                           label=f"hollow: fewer than {SEEDS} seeds finished")]
        fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(f"V + P benchmark: V = {c} clean row{'s' if c > 1 else ''} per class (each point: the mean over "
                     f"{SEEDS} seeds)", fontsize=14, color=MEAN_COLOR, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        for ext in ("png", "pdf"):
            fig.savefig(a.plots / f"V{c}.{ext}", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    out = a.tables / "by_v.md"
    out.write_text("\n".join(md) + "\n")
    print(f"-> {a.plots}/V{{1,3,5}}.png (and .pdf), {out}")


if __name__ == "__main__":
    main()

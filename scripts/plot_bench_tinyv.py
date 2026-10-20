#!/usr/bin/env python
"""Per-setting plots of the tiny-V benchmark (no aggregation over settings): one figure per V size at one R.

Each figure has a panel per metric (accuracy, precision of the removed rows, ROC-AUC, log loss); the x axis lists
the methods, a line per dataset, and a 7th line (black) is the mean over the 6 datasets. Methods that relabel
rows are left out: the relabel modes and the V-selected EMLC, IEG and label spreading (they chose relabelling in
almost every instance). The same numbers go to a markdown file, one table per V size and metric.

    python scripts/plot_bench_tinyv.py [--R 30] [--plots results/plots/bench_tinyv] [--tables results/tables/bench_tinyv]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_bench_tinyv as RB  # noqa: E402

PLOTTED = [("v_only", "V only"), ("poisoned_plus_v", "poisoned + V"), ("oracle_plus_v", "oracle + V"),
           ("vfilter", "V-filter"), ("greedy_pct10", "greedy, 10%/step"), ("greedy_pct1", "greedy, 1%/step"),
           ("greedy_m1", "greedy, 1 row/step"), ("attn_topk", "attention top-k (M1)"),
           ("attn_soft", "attention soft (A3)"), ("attn_soft_softw", "A3, soft weights + V"),
           ("nan_topk", "NaN gate, top-k vote"), ("label_spread", "label spreading"), ("knn_shapley", "KNN-Shapley"),
           ("knn_shapley_vsel", "KNN-Shapley (V-sel.)"), ("emlc", "EMLC"), ("ieg", "IEG")]
DATASETS = [("gas-drift", "gas-drift"), ("students_dropout", "students"), ("heloc", "heloc"),
            ("ecommerce_shipping", "ecommerce"), ("is_this_a_good_customer", "good_customer"),
            ("coil2000_insurance_policies", "coil2000")]
# the dataviz reference palette's categorical slots 1-6 in their fixed order; the mean in the primary text ink
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
MEAN_COLOR = "#0b0b0b"
METRICS = [("acc", "accuracy (%)", 100, "{:.1f}", True), ("prec", "precision of the removed rows", 1, "{:.2f}", True),
           ("auc", "ROC-AUC", 1, "{:.3f}", True), ("ce", "log loss (lower is better)", 1, "{:.3f}", False)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_TINYV_ROOT", "results/bench_tinyv")))
    ap.add_argument("--R", type=float, default=30.0)
    ap.add_argument("--plots", type=Path, default=Path("results/plots/bench_tinyv"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables/bench_tinyv"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val = {(r["dataset"], r["c"], r["entry"]): r for r in RB.load(a.root) if r["R"] == a.R}
    cs = sorted({c for _, c, _ in val})
    a.plots.mkdir(parents=True, exist_ok=True)
    md = [f"# Tiny-V benchmark at R = {a.R:g}%: every method on every dataset, per V size", "",
          "One draw per cell, no aggregation over settings; mean = the mean over the 6 datasets. Methods that relabel "
          "rows are left out (the relabel modes and the V-selected EMLC, IEG and label spreading). Precision: the "
          "share of the removed rows that were poisoned ('–': nothing removed). Bold: the best entry of the column, "
          "oracle + V excluded. Accuracy is degenerate on good_customer and coil2000 (the majority class).", ""]
    for c in cs:
        get = lambda ds, e, m: (lambda v: np.nan if v is None or not np.isfinite(v) else float(v))(  # noqa: E731
            val[(ds, c, e)][m] if (ds, c, e) in val else None)
        x = np.arange(len(PLOTTED))
        fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor="#fcfcfb")
        md += [f"## V = {c} row{'s' if c > 1 else ''} per class, R = {a.R:g}%", ""]
        for ax, (m, title, scale, fmt, higher) in zip(axes.flat, METRICS):
            table = np.array([[scale * get(ds, e, m) for e, _ in PLOTTED] for ds, _ in DATASETS])
            with np.errstate(all="ignore"):
                mean = np.array([np.nanmean(col) if np.isfinite(col).any() else np.nan for col in table.T])
            for i, (_, name) in enumerate(DATASETS):
                ax.plot(x, table[i], color=COLORS[i], marker=MARKERS[i], markersize=6, linewidth=1.6, alpha=0.9,
                        label=name)
            ax.plot(x, mean, color=MEAN_COLOR, marker="*", markersize=11, linewidth=2.6, label="mean of the 6 datasets",
                    zorder=5)
            ax.set_facecolor("#fcfcfb")
            ax.set_title(title, fontsize=12, color=MEAN_COLOR, loc="left")
            ax.set_xticks(x)
            ax.set_xticklabels([name for _, name in PLOTTED], rotation=40, ha="right", fontsize=9, color="#52514e")
            ax.tick_params(axis="y", labelsize=9, colors="#52514e")
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.grid(axis="x", color="#efeeea", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#bdbcb6")
            # the table of this panel
            cols = [n for _, n in DATASETS] + ["mean"]
            md += [f"### {title}", "", "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            full = np.vstack([table, mean])
            best = []
            for row in full:
                ok = [v for v, (e, _) in zip(row, PLOTTED) if np.isfinite(v) and not e.startswith("oracle")]
                best.append((max(ok) if higher else min(ok)) if ok else None)
            for j, (e, name) in enumerate(PLOTTED):
                cells = []
                for i in range(len(full)):
                    v = full[i, j]
                    s = "–" if not np.isfinite(v) else fmt.format(v)
                    if np.isfinite(v) and not e.startswith("oracle") and v == best[i]:
                        s = f"**{s}**"
                    cells.append(s)
                md.append(f"| {name} | " + " | ".join(cells) + " |")
            md.append("")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(f"Tiny-V benchmark: V = {c} clean row{'s' if c > 1 else ''} per class, R = {a.R:g}% poisoned "
                     "(one draw per point)", fontsize=14, color=MEAN_COLOR, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(a.plots / f"R{a.R:g}_V{c}.{ext}", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    out = a.tables / f"R{a.R:g}_by_v.md"
    out.write_text("\n".join(md) + "\n")
    print(f"-> {a.plots}/R{a.R:g}_V{{{','.join(str(c) for c in cs)}}}.png (and .pdf), {out}")


if __name__ == "__main__":
    main()

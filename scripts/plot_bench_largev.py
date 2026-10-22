#!/usr/bin/env python
"""Per-setting plots of the larger-V benchmark (no aggregation over settings): one figure per V size at one R.

Each figure has a panel per metric (accuracy, precision of the removed rows, ROC-AUC, log loss); the x axis lists
the methods (all with V in the context), a line per dataset, and a 7th line (black) is the mean over the 6 datasets.
The same numbers go to a markdown file, one table per V size and metric. Styling as scripts/plot_bench_tinyv.py.

    python scripts/plot_bench_largev.py [--R 30] [--plots results/plots/bench_largev] [--tables results/tables/bench_largev]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_bench_tinyv as PT  # noqa: E402
import report_bench_largev as RL  # noqa: E402

PLOTTED = [("v_only", "V only"), ("poisoned_plus_v", "poisoned + V"), ("oracle_plus_v", "oracle + V"),
           ("vfilter", "V-filter + V"), ("attn_topk", "attention top-k (M1)"),
           ("attn_wsoft", "best soft form (w, PGD)"),
           ("nan_soft", "NaN gate soft, hard"), ("nan_soft_softg", "NaN gate soft, soft gates"),
           ("nan_topk", "NaN gate, top-k vote")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_LARGEV_ROOT", "results/bench_largev")))
    ap.add_argument("--R", type=float, default=30.0)
    ap.add_argument("--plots", type=Path, default=Path("results/plots/bench_largev"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables/bench_largev"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = RL.load(a.root)
    df = df[(df.R == a.R) & df.seed.isin(RL.BL.SEEDS)]
    val = {(r.dataset, r.V, r.entry): r for r in df.itertuples()}
    vs = sorted(df.V.unique())
    a.plots.mkdir(parents=True, exist_ok=True)
    md = [f"# Larger-V benchmark at R = {a.R:g}%: every method on every dataset, per V size", "",
          "One draw per cell (seed 0), no aggregation over settings; mean = the mean over the 6 datasets. Every "
          "method has all of V in the context. hard: the k = R% lowest-ranked rows removed; soft gates: nothing "
          "removed; best soft form: w / V-filter / PGD lr 0.8, held-out early stopping, deployed soft (its w = 0 rows "
          "deleted exactly). Precision: the share of the removed rows that were poisoned (for the soft form, the "
          "rows it set to w = 0; '–': nothing removed). Bold: the best entry of the column, oracle + V excluded. "
          "Accuracy is degenerate on good_customer and coil2000 (the majority class).", ""]
    for V in vs:
        def get(ds, e, m):
            r = val.get((ds, V, e))
            v = None if r is None else getattr(r, m)
            return np.nan if v is None or not np.isfinite(v) else float(v)

        x = np.arange(len(PLOTTED))
        fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor="#fcfcfb")
        md += [f"## V = {V} clean rows, R = {a.R:g}%", ""]
        for ax, (m, title, scale, fmt, higher) in zip(axes.flat, PT.METRICS):
            table = np.array([[scale * get(ds, e, m) for e, _ in PLOTTED] for ds, _ in PT.DATASETS])
            with np.errstate(all="ignore"):
                mean = np.array([np.nanmean(col) if np.isfinite(col).any() else np.nan for col in table.T])
            for i, (_, name) in enumerate(PT.DATASETS):
                ax.plot(x, table[i], color=PT.COLORS[i], marker=PT.MARKERS[i], markersize=6, linewidth=1.6, alpha=0.9,
                        label=name)
            ax.plot(x, mean, color=PT.MEAN_COLOR, marker="*", markersize=11, linewidth=2.6,
                    label="mean of the 6 datasets", zorder=5)
            ax.set_facecolor("#fcfcfb")
            ax.set_title(title, fontsize=12, color=PT.MEAN_COLOR, loc="left")
            ax.set_xticks(x)
            ax.set_xticklabels([name for _, name in PLOTTED], rotation=35, ha="right", fontsize=9, color="#52514e")
            ax.tick_params(axis="y", labelsize=9, colors="#52514e")
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.grid(axis="x", color="#efeeea", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#bdbcb6")
            cols = [n for _, n in PT.DATASETS] + ["mean"]
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
        fig.suptitle(f"Larger-V benchmark: V = {V} clean rows, R = {a.R:g}% poisoned (one draw per point)",
                     fontsize=14, color=PT.MEAN_COLOR, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(a.plots / f"R{a.R:g}_V{V}.{ext}", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    out = a.tables / f"R{a.R:g}_by_v.md"
    out.write_text("\n".join(md) + "\n")
    print(f"-> {a.plots}/R{a.R:g}_V{{{','.join(str(v) for v in vs)}}}.png (and .pdf), {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Figure: the clean rows used alone, through soft gates, or as a label filter, vs V.

Reads results/tables/gate_soft/combos_R20.csv (scripts/eval_context_combos.py; hub-flip-local,
R = 20%, validation seeds 0-2) and draws, per dataset, test accuracy against V (mean over the
seeds; the band spans the seeds) for: the V clean rows alone, the soft gates, the V-filter's
kept rows plus the V clean rows, and the oracle plus the V rows as a dashed reference.

    python scripts/plot_vfilter_vs_v.py   # -> results/plots/gate_soft/vfilter_vs_v_R20.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = (("v_only", "V clean rows alone", "#2a78d6"),
          ("soft", "soft gates", "#eb6834"),
          ("vfilter_plus_v", "V-filter + V rows", "#1baf7a"))
DATASETS = (("gas-drift", "gas-drift"), ("url_unique", "url_unique"), ("students_dropout", "students_dropout"))


def main() -> None:
    d = pd.read_csv("results/tables/gate_soft/combos_R20.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (ds, title) in zip(axes, DATASETS):
        g = d[d.dataset == ds].groupby("V")
        Vs = sorted(d[d.dataset == ds].V.unique())
        orc = 100 * g["oracle_plus_v"].mean()
        ax.plot(Vs, orc.loc[Vs], color="#8a8985", lw=1.5, ls="--", zorder=1)
        ax.annotate("oracle + V", (Vs[0], orc.loc[Vs[0]]), xytext=(0, 5), textcoords="offset points",
                    va="bottom", fontsize=9, color="#52514e")
        for key, label, color in SERIES:
            m, lo, hi = 100 * g[key].mean(), 100 * g[key].min(), 100 * g[key].max()
            ax.fill_between(Vs, lo.loc[Vs], hi.loc[Vs], color=color, alpha=0.12, lw=0, zorder=2)
            ax.plot(Vs, m.loc[Vs], color=color, lw=2, marker="o", ms=5, zorder=3, label=label)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("V (clean rows; 20% of them held out for the gates)")
        ax.set_xticks(Vs)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(Vs[0] - 20, Vs[-1] + 20)
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("hub-flip-local, R = 20%: the clean rows as a label filter vs alone vs as a gating target\n"
                 "(mean of 3 validation draws; band = their range)", fontsize=10.5, x=0.01, ha="left")
    out = Path("results/plots/gate_soft/vfilter_vs_v_R20.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()

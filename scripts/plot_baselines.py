#!/usr/bin/env python
"""Figures of the baseline table: one panel per V, one line per dataset, the methods along x.

Reads results/tables/gate_soft/baselines_R20_s0.csv (scripts/make_baseline_table.py;
hub-flip-local, R = 20%, validation seed 0) and draws
  accuracy    test accuracy of the ten ways of using the clean rows (the oracle pair, which
              knows the poisoned rows, on a grey band)
  precision   precision@R of the removed rows for the V-filter, the soft gates (their k
              lowest) and the top-k vote, against removing k rows at random (= R%)

    python scripts/plot_baselines.py   # -> results/plots/gate_soft/baselines_{acc,prec}_R20_s0.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = (("gas-drift", "#2a78d6", "o"), ("url_unique", "#eb6834", "s"), ("heloc", "#1baf7a", "^"),
            ("students_dropout", "#eda100", "D"), ("credit-approval", "#e87ba4", "v"))
ACC = (("v_only", "V only"), ("poisoned", "poisoned"), ("poisoned_plus_v", "V + poisoned"),
       ("vfilter_plus_v", "V + filtered"), ("soft", "soft gate"), ("soft_plus_v", "soft gate + V"), ("topk", "top-k"),
       ("topk_plus_v", "top-k + V"), ("oracle_train", "oracle"), ("oracle_plus_v", "oracle + V"))
PREC = (("prec_vfilter", "V-filter"), ("prec_soft", "soft gate"), ("prec_topk", "top-k"))
INK, MUTED, GRID, SURFACE, BAND = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb", "#efeeea"


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def panels(d: pd.DataFrame, cols, *, ylabel: str, ylim, out: Path, title: str, scale: float, ref=None) -> None:
    Vs = sorted(d.V.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6) if len(cols) > 3 else (12, 7.4), sharex=True, sharey=True,
                             constrained_layout=True)
    x = range(len(cols))
    for ax, V in zip(axes.flat, Vs):
        if cols is ACC:
            ax.axvspan(len(cols) - 2.5, len(cols) - 0.5, color=BAND, lw=0, zorder=0)
        if ref is not None:  # labelled below the line at the left, where no method falls
            ax.axhline(ref[0], color="#8a8985", lw=1.2, ls="--", zorder=1)
            ax.annotate(ref[1], (-0.35, ref[0]), xytext=(0, -4), textcoords="offset points", ha="left",
                        va="top", fontsize=8.5, color=MUTED)
        for ds, color, marker in DATASETS:
            r = d[(d.dataset == ds) & (d.V == V)]
            if r.empty:
                continue
            ax.plot(x, [scale * float(r[k].iloc[0]) for k, _ in cols], color=color, lw=1.8, marker=marker, ms=7,
                    mec=SURFACE, mew=1.2, zorder=3, label=ds)
        ax.set_title(f"V = {V}", loc="left", fontsize=10.5, color=INK)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK)
    for ax in axes[-1]:
        if len(cols) > 3:
            ax.set_xticks(list(x), [lbl for _, lbl in cols], rotation=40, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticks(list(x), [lbl for _, lbl in cols])
    axes[0, 0].set_ylim(*ylim)
    axes[0, 0].set_xlim(-0.4, len(cols) - 0.6)
    seen = {}
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lbl, h)
    labels = [ds for ds, _, _ in DATASETS if ds in seen]
    handles = [seen[lbl] for lbl in labels]
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color=INK)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(DATASETS), fontsize=9.5, frameon=False,
               labelcolor=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


def main() -> None:
    d = pd.read_csv("results/tables/gate_soft/baselines_R20_s0.csv")
    panels(d, ACC, ylabel="test accuracy (%)", ylim=(60, 101), scale=100,
           out=Path("results/plots/gate_soft/baselines_acc_R20_s0.png"),
           title="Test accuracy of each way of using the V clean rows (hub-flip-local, R = 20%, validation seed 0)\n"
                 "grey band: the oracle references, which know the poisoned rows; credit-approval has V ≤ 200 only")
    panels(d, PREC, ylabel="precision of the removed rows", ylim=(0, 1.02), scale=1, ref=(0.2, "random removal (0.20)"),
           out=Path("results/plots/gate_soft/baselines_prec_R20_s0.png"),
           title="Precision@R of the removed rows (hub-flip-local, R = 20%, validation seed 0; the oracle = 1)\n"
                 "soft gate + V and top-k + V share the rows of soft gate and top-k; credit-approval has V ≤ 200 only")


if __name__ == "__main__":
    main()

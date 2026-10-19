#!/usr/bin/env python
"""Figures of the iterative filter (scripts/iterative_filter.py) against the V-filter.

  itf_vs_v_R20.png     per dataset, test accuracy against V (hub-flip-local, R = 20%, stratified,
                       seed 0; results/iterative_filter/e1, e2): the V clean rows alone, the V-filter
                       + V, the greedy growth (1% of n per step) without and with the class prior,
                       oracle + V dashed
  itf_noise_gain.png   every draw run (all experiments): the accuracy gain over the V-filter of the
                       greedy without (left) and with (right) the class prior, against the label
                       noise estimated from V alone (5-fold CV accuracy of TabPFN on the V rows,
                       results/iterative_filter/vnoise.csv)

    python scripts/plot_iterative_filter.py   # -> results/plots/iterative_filter/
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
DS_COLOR = dict(zip(DATASETS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
SERIES = (("v_only", "V clean rows alone", "#eda100"), ("oneshot", "V-filter + V", "#2a78d6"),
          ("grow:pct1", "greedy, 1% of n per step", "#eb6834"), ("grow:pct1:prior", "greedy + class prior", "#1baf7a"))
INK, MUTED, GRID, SURFACE = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"
OUT = Path("results/plots/iterative_filter")


def load(exps) -> dict:
    rows = {}
    for exp in exps:
        for f in sorted(Path("results/iterative_filter", exp).glob("rows_*.csv")):
            for r in csv.DictReader(open(f)):
                key = (r["dataset"], r["attack"], r["form"], float(r["R"]), int(r["V"]), int(r["seed"]))
                rows.setdefault(key, {}).setdefault(r["variant"], 100 * float(r["acc"]))
    return rows


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def vs_v(rows: dict) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4.0), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        draws = sorted((k for k in rows if k[0] == ds and k[1:4] == ("hub-flip-local", "stratified", 20.0) and k[5] == 0
                        and "oneshot" in rows[k]), key=lambda k: k[4])
        Vs = [k[4] for k in draws]
        ax.plot(Vs, [rows[k].get("oracle_plus_v", float("nan")) for k in draws], color="#8a8985", lw=1.4, ls="--", zorder=1)
        ax.annotate("oracle + V", (Vs[-1], rows[draws[-1]]["oracle_plus_v"]), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=MUTED)
        for key, label, color in SERIES:
            ax.plot(Vs, [rows[k].get(key, float("nan")) for k in draws], color=color, lw=2, marker="o", ms=5,
                    mec=SURFACE, mew=1, zorder=3, label=label)
        ax.set_title(ds, loc="left", fontsize=10.5, color=INK)
        ax.set_xticks(Vs)
        ax.set_xlabel("V (clean rows)", color=INK)
        style(ax)
    axes[0].set_ylabel("test accuracy (%)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(SERIES), frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Growing the trusted set from V vs the one-shot V-filter (hub-flip-local, R = 20%, validation seed 0)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "itf_vs_v_R20.png", dpi=150, facecolor=SURFACE)
    print(OUT / "itf_vs_v_R20.png")


def noise_gain(rows: dict) -> None:
    vn = {r["draw"]: float(r["v_cv_acc"]) for r in csv.DictReader(open("results/iterative_filter/vnoise.csv"))}
    markers = {"stratified V": "o", "local V": "s", "boundary attack": "^", "R = 10% or 30%": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True, constrained_layout=True)
    for ax, variant, title in zip(axes, ("grow:pct1", "grow:pct1:prior"), ("greedy (1% steps)", "greedy + class prior")):
        pts = defaultdict(list)
        for k, r in rows.items():
            draw = f"{k[0]}_{k[1]}_{k[2]}_R{k[3]:g}_V{k[4]}s{k[5]}"
            if draw in vn and "oneshot" in r and variant in r and k[2] != "imbalanced":
                setting = ("local V" if k[2] == "local" else "boundary attack" if k[1] == "boundary-flip-knn"
                           else "R = 10% or 30%" if k[3] != 20.0 else "stratified V")
                pts[(k[0], setting)].append((vn[draw], r[variant] - r["oneshot"]))
        ax.axhline(0, color="#8a8985", lw=1.2, zorder=1)
        for (ds, setting), xy in sorted(pts.items(), key=lambda t: (DATASETS.index(t[0][0]), t[0][1])):
            ax.scatter(*zip(*xy), color=DS_COLOR[ds], marker=markers[setting], s=46, edgecolor=SURFACE, lw=1, zorder=3,
                       label=f"{ds}, {setting}")
        n = sum(len(v) for v in pts.values())
        ax.set_title(f"{title}: {n} draws", loc="left", fontsize=10.5, color=INK)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_ylabel("minus the V-filter (test accuracy, points)", color=INK)
    fig.supxlabel("label noise estimated from V alone: 5-fold CV accuracy of TabPFN on the V rows", color=INK, fontsize=10)
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="outside right upper", fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Where growing the trusted set pays", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "itf_noise_gain.png", dpi=150, facecolor=SURFACE)
    print(OUT / "itf_noise_gain.png")


def main() -> None:
    vs_v(load(["e1", "e2"]))
    exps = sorted(p.name for p in Path("results/iterative_filter").glob("e*") if p.is_dir())
    noise_gain(load(exps))


if __name__ == "__main__":
    main()

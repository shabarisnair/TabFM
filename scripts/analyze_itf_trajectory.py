#!/usr/bin/env python
"""When does poison enter the trusted set? Trajectories of the iterative filter's growth.

For every saved greedy run of results/iterative_filter/<exp>/runs (scripts/iterative_filter.py
saves the step at which each context row joined T), the share of the poisoned rows already in T
when T has accepted a given share of the n - k rows it will accept (the rows join in step
order; within a step, in score order). A filter that never admits poison stays at 0 to the end;
the V-filter's final value is 1 - its recall (= 1 - its precision, since it removes k rows).

Writes results/tables/iterative_filter/<exp>_trajectory.md (the share at 25/50/75/90/100% of the
acceptances) and, with --plot, results/plots/iterative_filter/<exp>_trajectory.png (one panel
per dataset, the V = 50 and V = 500 draws of the given variants).

    python scripts/analyze_itf_trajectory.py --exp e1 --plot --variants grow:pct1 grow:m1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
MARKS = (0.25, 0.5, 0.75, 0.9, 1.0)


def curve(z) -> tuple[np.ndarray, np.ndarray]:
    """(accepted share, poisoned share in T) after each accepted row, in acceptance order."""
    acc = np.flatnonzero(z["step_of"] >= 0)
    order = acc[np.lexsort((z["score_at"][acc], z["step_of"][acc]))]
    pois = z["pois"]
    return np.arange(1, len(order) + 1) / len(order), np.cumsum(pois[order]) / pois.sum()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--variants", nargs="+", default=None, help="variants to plot (default: all greedy ones)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    root = Path("results/iterative_filter") / a.exp / "runs"
    L = [f"# When poison enters T ({a.exp})", "",
         "Share of the poisoned rows already in T after T has accepted 25 / 50 / 75 / 90 / 100% of its rows "
         "(oneshot: only the final value).", "",
         "| draw | variant | " + " | ".join(f"{int(100 * m)}%" for m in MARKS) + " |", "|---|---|" + "---|" * len(MARKS)]
    curves = {}
    for d in sorted(root.iterdir()):
        for f in sorted(d.glob("*.npz")):
            if f.name.startswith("base_"):
                continue
            z = np.load(f)
            variant = f.stem.replace("grow_", "grow:").replace("_", ":")
            x, y = curve(z)
            vals = [y[min(len(y) - 1, int(np.ceil(m * len(y))) - 1)] for m in MARKS]
            if variant.startswith("oneshot"):
                vals = ["" for _ in MARKS[:-1]] + [vals[-1]]
            L.append(f"| {d.name} | {variant} | " + " | ".join(v if v == "" else f"{v:.2f}" for v in vals) + " |")
            curves[(d.name, variant)] = (x, y)
    out = Path("results/tables/iterative_filter") / f"{a.exp}_trajectory.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not a.plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = a.variants or sorted({v for _, v in curves if v.startswith("grow")})
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(17, 3.8), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        for i, v in enumerate(variants):
            for V, ls in ((50, "-"), (500, "--")):
                hit = [c for (dn, vn), c in curves.items() if vn == v and dn.startswith(ds + "_") and f"_V{V}s" in dn]
                if hit:
                    ax.plot(*hit[0], color=colors[i % len(colors)], ls=ls, lw=1.8, label=f"{v}, V = {V}")
        ax.plot([0, 1], [0, 1], color="#8a8985", lw=1, ls=":")
        ax.set_title(ds, loc="left", fontsize=10)
        ax.set_xlabel("share of the accepted rows so far")
        ax.grid(color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of the poisoned rows in T")
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="outside lower center", ncol=min(6, len(labels)), frameon=False, fontsize=9)
    fig.suptitle(f"When poison enters the trusted set ({a.exp}; dotted: a random order)", x=0.01, ha="left", fontsize=11)
    png = Path("results/plots/iterative_filter") / f"{a.exp}_trajectory.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, facecolor="#fcfcfb")
    print(png)


if __name__ == "__main__":
    main()

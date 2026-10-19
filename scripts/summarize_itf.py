#!/usr/bin/env python
"""One summary of every iterative-filter run (results/iterative_filter/e*/rows_*.csv).

Groups the draws by setting (attack, form of the clean set, R; validation seed 0 or other seeds)
and dataset, and reports for each variant, over the draws where both it and the V-filter
(oneshot) ran: the mean test-accuracy difference to the V-filter, draws above-below, the worst
draw, and the mean precision of the removed rows (the V-filter's in brackets). Also the V-filter's
own mean accuracy, V only and oracle + V on the same draws.

    python scripts/summarize_itf.py [--variants grow:pct1 grow:pct1:prior ...]
    # -> results/tables/iterative_filter/summary.md
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
DEFAULT = ("grow:pct1", "grow:pct1:prior", "grow:frac0.1", "grow:pct1:conf0.9", "grow:m1")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT))
    a = ap.parse_args(argv)
    rows = {}
    for f in sorted(Path("results/iterative_filter").glob("e*/rows_*.csv")):
        for r in csv.DictReader(open(f)):
            key = (r["dataset"], r["attack"], r["form"], float(r["R"]), int(r["V"]), int(r["seed"]))
            rows.setdefault(key, {}).setdefault(r["variant"], (100 * float(r["acc"]), float(r["prec"]) if r["prec"] else None))
    groups = defaultdict(list)
    for key in rows:
        ds, attack, form, R, V, s = key
        groups[(attack, form, R, "seed 0" if s == 0 else "seeds 1-2", ds)].append(key)
    L = ["# Iterative filter: every run against the V-filter", "",
         "Per setting and dataset: the V-filter's mean test accuracy (V only / oracle + V on the same draws), and per "
         "variant the mean accuracy difference to the V-filter over the draws both ran, draws above-below, the worst "
         "draw, and mean precision (V-filter's in brackets). hub-flip-local unless noted; seed = validation draw.", ""]
    L += ["| setting | dataset | draws | V-filter (V only / oracle+V) | " + " | ".join(a.variants) + " |",
          "|---|---|---|---|" + "---|" * len(a.variants)]
    order = sorted(groups, key=lambda g: (g[0] != "hub-flip-local", g[1] != "stratified", g[2] != 20.0, g[3] != "seed 0",
                                          g[0], g[1], -g[2], g[3], DATASETS.index(g[4])))
    for g in order:
        keys = [k for k in groups[g] if "oneshot" in rows[k]]
        if not keys:
            continue
        os_acc = [rows[k]["oneshot"][0] for k in keys]
        ref = lambda name: np.mean([rows[k][name][0] for k in keys if name in rows[k]]) if any(name in rows[k] for k in keys) else float("nan")  # noqa: E731
        cells = []
        for v in a.variants:
            ks = [k for k in keys if v in rows[k]]
            if not ks:
                cells.append("")
                continue
            d = np.array([rows[k][v][0] - rows[k]["oneshot"][0] for k in ks])
            prec = np.mean([rows[k][v][1] for k in ks])
            prec_os = np.mean([rows[k]["oneshot"][1] for k in ks])
            cells.append(f"{d.mean():+.1f} ({int((d > 1e-9).sum())}-{int((d < -1e-9).sum())}, worst {d.min():+.1f}; "
                         f"prec {prec:.2f}/{prec_os:.2f}) [{len(ks)}]")
        setting = f"{g[0]}, {g[1]} V, R {g[2]:g}%, {g[3]}"
        L.append(f"| {setting} | {g[4]} | {len(keys)} | {np.mean(os_acc):.1f} ({ref('v_only'):.1f} / {ref('oracle_plus_v'):.1f}) | "
                 + " | ".join(cells) + " |")
    out = Path("results/tables/iterative_filter/summary.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

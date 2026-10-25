#!/usr/bin/env python
"""The control runs of scripts/why_tinyv.py (results/why_tinyv/roots/<attack>/runs) against the benchmark's own
hub-flip-local runs (results/bench_tinyv/runs; and its CPU rerun, roots/hub-flip-local-cpu, to check CPU vs GPU): per
attack and method, means over the instances that every compared attack has, of test ROC-AUC / accuracy / log loss and
the precision of the removed rows; also the damage the poison does (no cleaning vs oracle).

    python scripts/report_why_controls.py [--out results/tables/why_tinyv/controls.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_bench_tinyv as RB  # noqa: E402

ROOTS = {"hub-flip-local": Path("results/bench_tinyv"), "hub-flip-local (CPU rerun)": Path("results/why_tinyv/roots/hub-flip-local-cpu"),
         "hub-flip-random": Path("results/why_tinyv/roots/hub-flip-random"), "rand-flip": Path("results/why_tinyv/roots/rand-flip")}
ENTRIES = ["poisoned_plus_v", "oracle_plus_v", "v_only", "vfilter", "greedy_pct10", "greedy_pct1", "metasift", "metasift_small",
           "attn_topk_it300", "tabsift_m1"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/tables/why_tinyv/controls.md"))
    a = ap.parse_args(argv)
    val = {}
    for att, root in ROOTS.items():
        if (root / "runs").exists():
            for r in RB.load(root):
                if r["R"] == 30.0:
                    val[(att, r["dataset"], r["c"], r["entry"])] = r
    lines = ["# Control runs: the same V and context rows, other poisonings (R = 30%)", ""]
    for group, atts, datasets in (
            ("all 6 datasets: hub-flip-local vs rand-flip", ["hub-flip-local", "hub-flip-local (CPU rerun)", "rand-flip"], None),
            ("the multiclass datasets (gas-drift, students): + hub-flip-random", ["hub-flip-local", "hub-flip-random", "rand-flip"],
             ["gas-drift", "students_dropout"])):
        lines += [f"## {group}", "", "Means over the instances that every listed attack has for that method "
                  "(ROC-AUC / accuracy % / log loss / precision of the removed rows; n = instances).", "",
                  "| method | " + " | ".join(atts) + " |", "|---|" + "---|" * len(atts)]
        for e in ENTRIES:
            keys = defaultdict(set)
            for (att, ds, c, en) in val:
                if en == e and (datasets is None or ds in datasets):
                    keys[att].add((ds, c))
            common = set.intersection(*(keys[x] for x in atts if x in keys)) if all(x in keys for x in atts if "CPU" not in x) else set()
            common = {k for k in common if all((x, *k, e) in val for x in atts if x in keys)}
            if not common:
                continue
            cells = []
            for att in atts:
                rs = [val[(att, *k, e)] for k in common if (att, *k, e) in val]
                if not rs:
                    cells.append(".")
                    continue
                pr = [r["prec"] for r in rs if r.get("prec") is not None]
                cells.append(f"{np.mean([r['auc'] for r in rs]):.3f} / {100 * np.mean([r['acc'] for r in rs]):.1f} / "
                             f"{np.mean([r['ce'] for r in rs]):.3f} / {np.mean(pr):.2f}" if pr else
                             f"{np.mean([r['auc'] for r in rs]):.3f} / {100 * np.mean([r['acc'] for r in rs]):.1f} / {np.mean([r['ce'] for r in rs]):.3f}")
            lines.append(f"| {e} (n {len(common)}) | " + " | ".join(cells) + " |")
        lines.append("")
    # per dataset: the poison's damage and each method's precision
    lines += ["## Per dataset (mean over V sizes): ROC-AUC of no cleaning -> oracle, and the removal precision of V-filter / "
              "greedy pct10 / greedy pct1 / Meta-Sift", ""]
    for ds in ["gas-drift", "heloc", "ecommerce_shipping", "students_dropout", "is_this_a_good_customer", "coil2000_insurance_policies"]:
        for att in ("hub-flip-local", "hub-flip-random", "rand-flip"):
            g = lambda e, m: np.mean([val[(att, ds, c, e)][m] for c in (1, 3, 5) if (att, ds, c, e) in val and val[(att, ds, c, e)][m] is not None]) \
                if any((att, ds, c, e) in val for c in (1, 3, 5)) else float("nan")  # noqa: E731
            if np.isnan(g("poisoned_plus_v", "auc")):
                continue
            lines.append(f"- {ds} {att}: AUC {g('poisoned_plus_v', 'auc'):.3f} -> {g('oracle_plus_v', 'auc'):.3f}; precision "
                         + " / ".join(f"{g(e, 'prec'):.2f}" for e in ("vfilter", "greedy_pct10", "greedy_pct1", "metasift")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])

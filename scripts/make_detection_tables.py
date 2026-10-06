#!/usr/bin/env python
"""Detection quality of the outlier-filtering step, per condition.

How well did the filter pick out the rows the attacker actually flipped?

CAUTION -- precision, recall and accuracy are ONE number here, not three. The removal
budget is exactly k = R% of rows and the attacker flipped exactly k, so tp+fp = tp+fn = k
and therefore

    precision == recall            and      accuracy == 1 - 2*(k/n)*(1 - precision)

(verified exactly on all 750 cells). They are all reported because they were asked for,
but they carry no independent evidence: accuracy merely rescales precision by the budget,
which is why it looks flattering at small R -- at R=5 a filter that catches NOTHING still
scores 0.90 accuracy, because 90% of rows are correctly left alone.

``ranking AUC`` is the one number that is independent of the budget: it scores the
ordering of all n rows by p(carried label), so it says whether the signal is there at all,
separately from whether the R% cut exploits it.

Per-class views (both are reported because "per class" is ambiguous once labels are the
thing being corrupted):
    recall by TRUE class c        of the flipped rows whose CLEAN label was c, how many
                                  did we catch -- which class's poison hides best
    precision by GIVEN class c    of the flagged rows CARRYING label c, how many were
                                  really flipped -- which class we wrongly throw away

    python scripts/make_detection_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies",
            "heloc", "lcld_v2"]
ATTACKS = [("label-flip-random", "random label flips"), ("hub-flip", "hub label flips")]
KEYS = [("precision", "precision"), ("recall", "recall"), ("accuracy", "accuracy"),
        ("f1", "F1"), ("detection_roc_auc", "ranking AUC"),
        ("false_removal_rate", "false removal")]


def fmt(d: dict) -> str:
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.4f} +- {d['std']:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/defense_kfold"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/defense_kfold"))
    ap.add_argument("--format", choices=["text", "markdown", "both"], default="both")
    a = ap.parse_args(argv)

    data = {}
    for f in sorted(a.root.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        data[(s["dataset"], s["attack"], int(round(s["row_percent"])))] = s
    if not data:
        raise SystemExit(f"no summary.json under {a.root}")
    a.out.mkdir(parents=True, exist_ok=True)

    rows_csv = []
    for ds in DATASETS:
        for attack, label in ATTACKS:
            rs = sorted(r for (d, at, r) in data if d == ds and at == attack)
            if not rs:
                continue
            s0 = data[(ds, attack, rs[0])]
            lines = [f"{ds}   {label}   context {s0['n_context']} rows   "
                     f"{s0['n_poison_draws']}x{s0['n_fold_draws']} cells, "
                     f"{s0['n_folds']} folds", ""]
            head = f"  {'R%':>4s} {'k':>6s}"
            for _, lab in KEYS:
                head += f"  {lab:>19s}"
            head += f"  {'rec|true0':>10s} {'rec|true1':>10s}"
            lines += [head, "  " + "-" * (len(head) - 2)]
            for r in rs:
                s = data[(ds, attack, r)]
                d = s["aggregate_detection"]
                line = f"  {r:>4d} {s['k_flipped']:>6d}"
                for key, _ in KEYS:
                    line += f"  {fmt(d.get(key)):>19s}"
                for c in ("0", "1"):
                    v = d["recall_by_true_class"][c]
                    line += f"  {('-' if v['mean'] is None else format(v['mean'], '.3f')):>10s}"
                lines.append(line)
                for key, _ in KEYS:
                    v = d.get(key, {})
                    rows_csv.append({
                        "dataset": ds, "attack": attack, "row_percent": r,
                        "k_flipped": s["k_flipped"], "n_context": s["n_context"],
                        "metric": key, "mean": v.get("mean"), "std": v.get("std"),
                        "between_poison_std": v.get("between_poison_std"),
                        "within_poison_fold_std": v.get("within_poison_fold_std"),
                        "n_cells": v.get("n"),
                    })
                for c in ("0", "1"):
                    for key, blk in (("recall_by_true_class", "recall_by_true_class"),
                                     ("precision_by_given_class", "precision_by_given_class")):
                        v = d[blk][c]
                        rows_csv.append({
                            "dataset": ds, "attack": attack, "row_percent": r,
                            "k_flipped": s["k_flipped"], "n_context": s["n_context"],
                            "metric": f"{key}_{c}", "mean": v.get("mean"), "std": v.get("std"),
                            "between_poison_std": v.get("between_poison_std"),
                            "within_poison_fold_std": v.get("within_poison_fold_std"),
                            "n_cells": v.get("n")})
            lines.append("")
            lines.append("  precision == recall, and accuracy == 1 - 2(k/n)(1 - precision): "
                         "one number, three scales.")
            lines.append("  ranking AUC is the budget-free one -- 0.5 = no signal, "
                         "1.0 = every flipped row ranked below every clean one.")
            text = "\n".join(lines)
            if a.format in ("text", "both"):
                print("\n" + text + "\n")
            if a.format in ("markdown", "both"):
                (a.out / f"detection__{ds}__{attack}.md").write_text(
                    "```\n" + text + "\n```\n")

    p = a.out / "detection_results.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"wrote {p} ({len(rows_csv)} rows)")
    return rows_csv


if __name__ == "__main__":
    main()

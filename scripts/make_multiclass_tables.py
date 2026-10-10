#!/usr/bin/env python
"""Tables for the multiclass label-perturbation grid (results/multiclass).

One long CSV -- dataset, attack, R, metric, clean, poisoned mean/std, delta mean/std,
n_runs, rows flipped -- plus a readable table per dataset. Each metric also carries its
NO-SKILL level, the score of a model that ignores the features:

    accuracy           the test set's majority-class rate
    balanced_accuracy  1 / K
    mcc                0
    roc_auc            0.5
    ce                 entropy of the test-set class prior: the cross-entropy of always
                       predicting that prior, the best a constant predictor can do

    python scripts/make_multiclass_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DATASETS = ["students_dropout", "gas-drift"]
ATTACKS = ["label-flip-bernoulli", "hub-flip-local", "hub-flip-random"]
METRICS = ["ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc"]
SHORT = {"ce": "CE", "accuracy": "acc", "balanced_accuracy": "bal.acc", "mcc": "MCC",
         "roc_auc": "ROC-AUC"}


def no_skill(metric: str, class_counts: list[int]) -> float:
    n = sum(class_counts)
    prior = [c / n for c in class_counts]
    return {"accuracy": max(prior),
            "balanced_accuracy": 1 / len(class_counts),
            "mcc": 0.0,
            "roc_auc": 0.5,
            "ce": -sum(p * math.log(p) for p in prior if p > 0)}[metric]


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        for m in METRICS:
            rows.append({
                "dataset": s["dataset"], "attack": s["attack"],
                "row_percent": int(round(s["row_percent"])), "metric": m,
                "n_classes": s["n_classes"], "n_runs": s["n_runs"],
                "clean": s["clean"][m],
                "poisoned_mean": s["aggregate_poisoned"][m]["mean"],
                "poisoned_std": s["aggregate_poisoned"][m]["std"],
                "delta_mean": s["aggregate_delta"][m]["mean"],
                "delta_std": s["aggregate_delta"][m]["std"],
                "no_skill": no_skill(m, s["class_counts_test"]),
                "n_flipped_mean": s["n_flipped"]["mean"],
            })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/multiclass"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/multiclass"))
    a = ap.parse_args(argv)
    rows = load(a.root)
    if not rows:
        raise SystemExit(f"no summary.json under {a.root}")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "multiclass_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    idx = {(r["dataset"], r["attack"], r["row_percent"], r["metric"]): r for r in rows}
    lines = []
    for ds in DATASETS:
        rs = sorted({r["row_percent"] for r in rows if r["dataset"] == ds})
        any_r = next(r for r in rows if r["dataset"] == ds)
        lines.append(f"\n{ds}   K={any_r['n_classes']} classes   mean +- std over runs "
                     f"(label-flip-bernoulli / hub-flip-random: 10 runs; hub-flip-local: 1, deterministic)")
        lines.append("  no-skill level: " + "  ".join(
            f"{SHORT[m]} {idx[(ds, ATTACKS[0], rs[0], m)]['no_skill']:.4f}" for m in METRICS))
        for atk in ATTACKS:
            lines.append(f"\n  {atk}")
            hdr = f"    {'R%':>4s} {'flipped':>8s}" + "".join(f"{SHORT[m]:>18s}" for m in METRICS)
            lines += [hdr, "    " + "-" * (len(hdr) - 4)]
            for r in rs:
                cells = ""
                for m in METRICS:
                    x = idx.get((ds, atk, r, m))
                    cells += (f"{x['poisoned_mean']:>11.4f}+-{x['poisoned_std']:<5.3f}"
                              if x else f"{'-':>18s}")
                nf = idx[(ds, atk, r, METRICS[0])]["n_flipped_mean"] if (ds, atk, r, METRICS[0]) in idx else 0
                lines.append(f"    {r:>4d} {nf:>8.0f}{cells}")
    text = "\n".join(lines)
    print(text)
    (a.out / "multiclass_results.txt").write_text(text + "\n")
    print(f"\nwrote {a.out / 'multiclass_results.csv'} ({len(rows)} rows) and multiclass_results.txt")
    return rows


if __name__ == "__main__":
    main()

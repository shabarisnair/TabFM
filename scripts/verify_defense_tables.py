#!/usr/bin/env python
"""Audit every number in results/tables/defense_kfold against the raw per-cell records.

Nothing here imports the table code. For each condition the per-cell records in
``summary.json["runs"]`` are re-aggregated from scratch, the count-derived metrics are
recomputed with sklearn from the stored confusion matrices, the crossed design is
checked for structural integrity, and only then is the result compared with
``defense_results.csv``. A disagreement anywhere is a failure.

    python scripts/verify_defense_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

VARIANTS = ["poisoned", "defended", "oracle"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]


def vectors_from_counts(tn, fp, fn, tp):
    y = np.array([0] * (tn + fp) + [1] * (fn + tp))
    p = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return y, p


def sklearn_metrics(m: dict) -> dict:
    y, p = vectors_from_counts(m["tn"], m["fp"], m["fn"], m["tp"])
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "mcc": float(matthews_corrcoef(y, p)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "ce": float(m["ce"]), "roc_auc": float(m["roc_auc"])}


def check(cond, msg, fails):
    if not cond:
        fails.append(msg)


def audit(f: Path, tol: float, fails: list) -> dict:
    s = json.loads(f.read_text())
    tag = f"{s['dataset']}/{s['attack']}_r{int(round(s['row_percent'])):03d}"
    runs = s["runs"]
    npd, nfd = s["n_poison_draws"], s["n_fold_draws"]

    # --- 1. the crossed design is complete and correctly indexed ----------------------
    check(len(runs) == npd * nfd, f"{tag}: {len(runs)} cells != {npd}x{nfd}", fails)
    seen = {(r["poison_draw"], r["fold_draw"]) for r in runs}
    check(seen == {(p, q) for p in range(npd) for q in range(nfd)},
          f"{tag}: cell grid is not the full {npd}x{nfd} product", fails)

    # --- 2. poisoned/oracle must NOT depend on the fold draw --------------------------
    # They are scored once per poison draw, so every cell sharing a poison draw must
    # carry byte-identical values. This is the structural claim the decomposition rests
    # on: it is what makes within_poison_fold_std exactly 0 for those two.
    for p in range(npd):
        cells = [r for r in runs if r["poison_draw"] == p]
        for v in ("poisoned", "oracle"):
            if v not in cells[0]:
                continue
            for c in cells[1:]:
                check(c[v] == cells[0][v],
                      f"{tag}: {v} differs across fold draws within poison draw {p}", fails)
        idx = cells[0]["flipped_indices"]
        for c in cells[1:]:
            check(c["flipped_indices"] == idx,
                  f"{tag}: flipped_indices differ across fold draws in poison draw {p}", fails)

    # --- 3. the attack really flipped k rows, and the budget removed k -----------------
    for r in runs:
        check(r["n_flipped"] == s["k_flipped"],
              f"{tag}: n_flipped {r['n_flipped']} != k {s['k_flipped']}", fails)
        check(len(r["flagged_indices"]) == s["k_flipped"],
              f"{tag}: removed {len(r['flagged_indices'])} != budget {s['k_flipped']}", fails)
        check(r["n_context_defended"] == s["n_context"] - s["k_flipped"],
              f"{tag}: defended context size wrong", fails)
        d = r["detection"]
        check(d["tp"] + d["fp"] == s["k_flipped"], f"{tag}: tp+fp != k", fails)
        check(d["tp"] + d["fn"] == s["k_flipped"], f"{tag}: tp+fn != k", fails)
        check(d["tp"] + d["fp"] + d["fn"] + d["tn"] == s["n_context"],
              f"{tag}: detection counts do not sum to n_context", fails)

    # --- 4. stored metrics match sklearn recomputation from the counts -----------------
    for r in runs:
        for v in VARIANTS + ["clean"]:
            if v not in r:
                continue
            ref = sklearn_metrics(r[v])
            for m in ("accuracy", "balanced_accuracy", "mcc"):
                check(abs(r[v][m] - ref[m]) < tol,
                      f"{tag} {v}.{m}: stored {r[v][m]!r} != sklearn {ref[m]!r}", fails)

    # --- 5. re-aggregate from the raw cells -------------------------------------------
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for m in METRICS:
            by_p = [[sklearn_metrics(r[v])[m] for r in sorted(runs, key=lambda x: x["fold_draw"])
                     if r["poison_draw"] == p and v in r] for p in range(npd)]
            by_p = [row for row in by_p if row]
            if not by_p:
                continue
            flat = [x for row in by_p for x in row]
            means = [float(np.mean(row)) for row in by_p]
            within = [float(np.std(row)) for row in by_p if len(row) > 1]
            out[v][m] = {
                "mean": float(np.mean(flat)), "std": float(np.std(flat)),
                "between_poison_std": float(np.std(means)) if len(means) > 1 else 0.0,
                "within_poison_fold_std": (float(np.sqrt(np.mean(np.square(within))))
                                           if within else 0.0)}
    out["clean"] = {m: {"mean": sklearn_metrics(s["clean"])[m], "std": 0.0,
                        "between_poison_std": 0.0, "within_poison_fold_std": 0.0}
                    for m in METRICS}

    # --- 6. cross-check against what defend_context.py itself wrote --------------------
    for v in VARIANTS:
        for m in METRICS:
            got = s["aggregate"].get(v, {}).get(m)
            if got and m in out[v]:
                for key in ("mean", "std", "between_poison_std", "within_poison_fold_std"):
                    check(abs(got[key] - out[v][m][key]) < tol,
                          f"{tag}: summary aggregate.{v}.{m}.{key} "
                          f"{got[key]!r} != recomputed {out[v][m][key]!r}", fails)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/defense_kfold"))
    ap.add_argument("--csv", type=Path,
                    default=Path("results/tables/defense_kfold/defense_results.csv"))
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)

    fails: list[str] = []
    recomputed: dict = {}
    n = 0
    for f in sorted(a.root.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        recomputed[(s["dataset"], s["attack"], int(round(s["row_percent"])))] = audit(f, a.tol, fails)
        n += 1
    print(f"audited {n} conditions from {a.root}")

    cells = 0
    for row in csv.DictReader(open(a.csv)):
        key = (row["dataset"], row["attack"], int(row["row_percent"]))
        res = recomputed.get(key)
        if res is None:
            fails.append(f"{key}: in CSV but no run directory")
            continue
        ref = res.get(row["context"], {}).get(row["metric"])
        if ref is None:
            continue
        for col, val in (("mean", ref["mean"]), ("std", ref["std"]),
                         ("between_poison_std", ref["between_poison_std"]),
                         ("within_poison_fold_std", ref["within_poison_fold_std"])):
            if row[col] in ("", None):
                continue
            check(math.isclose(float(row[col]), val, rel_tol=0, abs_tol=a.tol),
                  f"{key} {row['context']}.{row['metric']}.{col}: "
                  f"CSV {row[col]} != recomputed {val!r}", fails)
        cells += 1
    print(f"cross-checked {cells} CSV cells against {a.csv}")

    if fails:
        print(f"\nFAILED: {len(fails)} check(s)")
        for x in fails[:40]:
            print("  -", x)
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    main()

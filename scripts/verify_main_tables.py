#!/usr/bin/env python
"""Audit every number in results/tables/main against the raw per-trial records.

Nothing here imports the table code. Metrics are recomputed from the confusion counts
in ``trials/run*_sub*.json`` with sklearn, the per-run winners are re-selected from
scratch, and the aggregates are re-derived; only then is the result compared with
``main_results.csv`` and with what ``attack_context.py`` itself wrote into
``summary.json``. A disagreement anywhere is a failure.

    python scripts/verify_main_tables.py
    python scripts/verify_main_tables.py --tol 1e-9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

ATTACKS = ["x-capgd", "label-flip"]
RS = ["001", "004", "016", "064"]


def vectors_from_counts(tn: int, fp: int, fn: int, tp: int):
    """The unique (up to order) label/prediction pair with this confusion matrix."""
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return y_true, y_pred


def metrics_from_counts(m: dict) -> dict:
    """Recompute every reported metric from the counts alone, via sklearn."""
    y, p = vectors_from_counts(m["tn"], m["fp"], m["fn"], m["tp"])
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "mcc": float(matthews_corrcoef(y, p)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "ce": float(m["ce"]), "roc_auc": float(m["roc_auc"])}


def check(cond: bool, msg: str, fails: list) -> None:
    if not cond:
        fails.append(msg)


def audit_run(d: Path, tol: float, fails: list) -> dict | None:
    summary = json.loads((d / "summary.json").read_text())
    trials = [json.loads(p.read_text()) for p in sorted(d.glob("trials/run*_sub*.json"))]
    if not trials:
        fails.append(f"{d}: no trial files")
        return None
    n_test = summary["n_test"]
    tag = f"{d.parent.name}/{d.name}"

    # --- 1. counts are self-consistent and the stored metrics match sklearn -------------
    for t in trials:
        for side in ("clean", "poisoned"):
            m = t[side]
            tot = m["tn"] + m["fp"] + m["fn"] + m["tp"]
            check(tot == n_test, f"{tag} {t['run_id']}/{t['subsample_id']} {side}: "
                                 f"counts sum to {tot}, expected n_test={n_test}", fails)
            ref = metrics_from_counts(m)
            for k in ("accuracy", "f1"):
                check(abs(m[k] - ref[k]) < tol,
                      f"{tag} {side}.{k}: stored {m[k]!r} != sklearn {ref[k]!r}", fails)

    # --- 2. the clean baseline is one number, identical in every trial ------------------
    c0 = trials[0]["clean"]
    for t in trials:
        check(t["clean"] == c0, f"{tag}: clean baseline differs between trials", fails)
    check(summary["clean"] == c0, f"{tag}: summary clean != trial clean", fails)

    # --- 3. re-select the per-run winners from scratch ----------------------------------
    best: dict[int, dict] = {}
    for t in trials:
        r = int(t["run_id"])
        cur = best.get(r)
        key = (t["delta"]["ce"], -int(t["subsample_id"]))
        if cur is None or key > (cur["delta"]["ce"], -int(cur["subsample_id"])):
            best[r] = t
    winners = [best[r] for r in sorted(best)]
    got = [(int(t["run_id"]), int(t["subsample_id"])) for t in winners]
    want = [(int(t["run_id"]), int(t["subsample_id"])) for t in summary["per_run_best"]]
    check(got == want, f"{tag}: per_run_best {want} != recomputed {got}", fails)

    # --- 4. delta really is poisoned - clean -------------------------------------------
    for t in trials:
        for k in ("ce", "accuracy", "f1"):
            check(abs(t["delta"][k] - (t["poisoned"][k] - t["clean"][k])) < tol,
                  f"{tag}: delta.{k} is not poisoned - clean", fails)

    # --- 5. aggregate over the winners, from sklearn-recomputed metrics ------------------
    clean_ref = metrics_from_counts(c0)
    out = {"clean": clean_ref, "n_runs": len(winners), "poisoned": {}, "delta": {}}
    for m in ("ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc", "f1"):
        vals = [metrics_from_counts(t["poisoned"])[m] for t in winners]
        dels = [v - clean_ref[m] for v in vals]
        out["poisoned"][m] = (float(np.mean(vals)), float(np.std(vals)))
        out["delta"][m] = (float(np.mean(dels)), float(np.std(dels)))

    # --- 6. cross-check against what attack_context.py wrote ---------------------------
    stored = summary.get("aggregate_delta", {}).get("overall", {})
    for m in ("ce", "accuracy", "f1", "roc_auc"):
        if m in stored and stored[m].get("mean") is not None:
            check(abs(stored[m]["mean"] - out["delta"][m][0]) < tol,
                  f"{tag}: summary aggregate_delta.{m} {stored[m]['mean']!r} "
                  f"!= recomputed {out['delta'][m][0]!r}", fails)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/main_experiments"))
    ap.add_argument("--csv", type=Path, default=Path("results/tables/main/main_results.csv"))
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)

    fails: list[str] = []
    recomputed: dict[tuple, dict] = {}
    n_runs = 0
    for ds_dir in sorted(p for p in a.root.iterdir() if p.is_dir()):
        for attack in ATTACKS:
            for r in RS:
                d = ds_dir / f"{attack}_random_r{r}"
                if not (d / "summary.json").exists():
                    continue
                res = audit_run(d, a.tol, fails)
                if res:
                    recomputed[(ds_dir.name, attack, int(r))] = res
                    n_runs += 1
    print(f"audited {n_runs} run directories from {a.root}")

    # --- 7. the published CSV must match the independent recomputation ------------------
    n_cells = 0
    if a.csv.exists():
        for row in csv.DictReader(open(a.csv)):
            key = (row["dataset"], row["attack"], int(row["row_percent"]))
            res = recomputed.get(key)
            if res is None:
                fails.append(f"{key}: in CSV but no run directory")
                continue
            m = row["metric"]
            for col, val in (("clean", res["clean"][m]),
                             ("poisoned_mean", res["poisoned"][m][0]),
                             ("poisoned_std", res["poisoned"][m][1]),
                             ("delta_mean", res["delta"][m][0]),
                             ("delta_std", res["delta"][m][1])):
                check(math.isclose(float(row[col]), val, rel_tol=0, abs_tol=a.tol),
                      f"{key} {m}.{col}: CSV {row[col]} != recomputed {val!r}", fails)
            rel = row["relative_pct"]
            if rel:
                want = 100.0 * res["delta"][m][0] / abs(res["clean"][m]) if abs(res["clean"][m]) > 1e-12 else None
                check(want is not None and math.isclose(float(rel), want, rel_tol=1e-9),
                      f"{key} {m}.relative_pct: CSV {rel} != recomputed {want!r}", fails)
            else:
                check(abs(res["clean"][m]) <= 1e-12,
                      f"{key} {m}: relative_pct blank but clean={res['clean'][m]!r} is non-zero", fails)
            n_cells += 1
        print(f"cross-checked {n_cells} CSV cells against {a.csv}")
    else:
        print(f"[warn] {a.csv} not found; run make_main_tables.py first")

    if fails:
        print(f"\nFAILED: {len(fails)} check(s)")
        for f in fails[:40]:
            print("  -", f)
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    main()

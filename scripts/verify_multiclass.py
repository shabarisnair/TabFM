#!/usr/bin/env python
"""Audit every multiclass result against its saved per-trial arrays.

Nothing here reuses the metric or attack code that produced the numbers. For every
condition under --root:

  * each trial's metrics are recomputed from its saved test-set probabilities with
    separate implementations (manual cross-entropy and accuracy, Gorodkin's formula for
    MCC from the confusion matrix, sklearn's balanced accuracy and one-vs-rest macro AUC)
    and compared with summary.json;
  * the poisoning is checked from the saved labels: flipped rows change class, every
    other row is untouched, every class survives, and the per-attack rules hold --
    Bernoulli counts are plausible under Binomial(n, R) and vary between runs, hub rows
    are the same in every run and between hub-flip-local and hub-flip-random, and
    hub-flip-random redraws its targets;
  * the mean +- std aggregates are re-derived and compared.

    python scripts/verify_multiclass.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

METRICS = ("ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc")


def independent_metrics(p: np.ndarray, y: np.ndarray, K: int) -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pred = p.argmax(1)
    C = np.zeros((K, K)); np.add.at(C, (y, pred), 1)
    t, q, c, s = C.sum(1), C.sum(0), np.trace(C), C.sum()
    den = math.sqrt((s * s - q @ q) * (s * s - t @ t))
    # No renormalisation. The float32 rows already sum to 1 within ~1e-7, which passes
    # sklearn's own check, and rescaling each row by a slightly different factor breaks
    # exact float ties -- AUC scores a tie as 1/2, so that alone moved one AUC by 4.5e-6.
    return {"ce": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
            "accuracy": float(c / s),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mcc": 0.0 if den == 0 else float((c * s - t @ q) / den),
            "roc_auc": float(roc_auc_score(y, p, multi_class="ovr", average="macro",
                                           labels=list(range(K))))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/multiclass"))
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(argv)

    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    hub_rows: dict[tuple, dict] = {}
    n_dirs = n_trials = 0
    for summ in sorted(a.root.glob("*/*/summary.json")):
        d = summ.parent
        s = json.loads(summ.read_text())
        tag, K, R, atk = f"{d.parent.name}/{d.name}", s["n_classes"], s["row_percent"], s["attack"]
        n_dirs += 1

        z = np.load(d / "trials" / "clean.npz")
        ref = independent_metrics(z["probs"], z["y_test"], K)
        for m in METRICS:
            check(abs(ref[m] - s["clean"][m]) < a.tol, f"{tag} clean {m}: {s['clean'][m]} vs {ref[m]}")
        y_clean_ctx = z["y_context"].astype(int)

        rows_per_run, targets_per_run = [], []
        per_run = {m: [] for m in METRICS}
        for r in s["runs"]:
            t = np.load(d / "trials" / f"run{r['run_id']}.npz")
            n_trials += 1
            rows = t["flipped_indices"].astype(int)
            yc, yp = t["y_clean"].astype(int), t["y_poisoned"].astype(int)
            n = len(yc)
            check(np.array_equal(yc, y_clean_ctx), f"{tag} run {r['run_id']}: y_clean differs from the context")
            check(len(np.unique(rows)) == len(rows) and (len(rows) == 0 or (rows.min() >= 0 and rows.max() < n)),
                  f"{tag} run {r['run_id']}: bad flipped indices")
            check(bool(np.all(yp[rows] != yc[rows])), f"{tag} run {r['run_id']}: a flipped row kept its class")
            untouched = np.ones(n, bool); untouched[rows] = False
            check(np.array_equal(yp[untouched], yc[untouched]), f"{tag} run {r['run_id']}: an unflipped row changed")
            check(set(np.unique(yp).tolist()) == set(range(K)), f"{tag} run {r['run_id']}: a class vanished")
            check(r["n_flipped"] == len(rows), f"{tag} run {r['run_id']}: n_flipped mismatch")
            T = np.zeros((K, K), int)
            if len(rows):
                np.add.at(T, (yc[rows], yp[rows]), 1)
            check(T.tolist() == r["transitions"], f"{tag} run {r['run_id']}: transition counts mismatch")

            m_ind = independent_metrics(t["probs"], t["y_test"], K)
            for m in METRICS:
                check(abs(m_ind[m] - r["metrics"][m]) < a.tol,
                      f"{tag} run {r['run_id']} {m}: {r['metrics'][m]} vs {m_ind[m]}")
                per_run[m].append(m_ind[m])
            rows_per_run.append(tuple(rows.tolist()))
            targets_per_run.append(tuple(yp[rows].tolist()))

        # ---- per-attack rules ------------------------------------------------------
        if atk == "label-flip-bernoulli" and R > 0:
            mu, sd = n * R / 100, math.sqrt(n * R / 100 * (1 - R / 100))
            for cnt in map(len, rows_per_run):
                check(abs(cnt - mu) <= 6 * sd + 1, f"{tag}: {cnt} flips implausible under Binomial({n}, {R/100})")
            if len(rows_per_run) > 1:
                check(len({len(x) for x in rows_per_run}) > 1, f"{tag}: every run flipped the same count")
                check(len(set(rows_per_run)) == len(rows_per_run), f"{tag}: two runs drew identical rows")
        if atk.startswith("hub-flip"):
            from tabfm_experiments.sampling import k_from_percent
            want = 0 if R <= 0 else k_from_percent(n, R)
            check(all(len(x) == want for x in rows_per_run), f"{tag}: hub count is not k={want}")
            check(len(set(rows_per_run)) == 1, f"{tag}: hub rows differ between runs")
            hub_rows.setdefault((d.parent.name, R), {})[atk] = rows_per_run[0]
            if atk == "hub-flip-random" and R > 0 and len(targets_per_run) > 1:
                check(len(set(targets_per_run)) > 1, f"{tag}: hub-flip-random targets never change")
            if atk == "hub-flip-local":
                check(s["n_runs"] == 1, f"{tag}: hub-flip-local should be a single run")

        # ---- aggregates ------------------------------------------------------------
        for m in METRICS:
            v = np.array(per_run[m])
            for key, val in (("mean", v.mean()), ("std", v.std())):
                check(abs(s["aggregate_poisoned"][m][key] - val) < a.tol,
                      f"{tag} aggregate {m}.{key}: {s['aggregate_poisoned'][m][key]} vs {val}")
            dv = v - s["clean"][m]
            check(abs(s["aggregate_delta"][m]["mean"] - dv.mean()) < a.tol, f"{tag} delta {m}.mean")

    # hub-flip-local and hub-flip-random must have poisoned the same rows
    pairs = 0
    for (ds, R), by in hub_rows.items():
        if len(by) == 2:
            pairs += 1
            check(by["hub-flip-local"] == by["hub-flip-random"],
                  f"{ds} R={R}: hub-flip-local and hub-flip-random flipped different rows")

    print(f"audited {n_dirs} conditions, {n_trials} poisoned trials, {pairs} local/random hub pairs "
          f"under {a.root}")
    if fails:
        print(f"\nFAILED: {len(fails)} check(s)")
        for f in fails[:40]:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

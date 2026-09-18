#!/usr/bin/env python
"""Audit results/tables/transfer against the raw per-fit records.

Imports none of the table code. Every metric is recomputed from the confusion counts of
each individual XGBoost fit with sklearn, the paired deltas are re-derived, the
aggregates are recomputed, and only then compared with ``transfer_results.csv`` and with
what ``xgb_transfer.py`` wrote into ``summary.json``.

    python scripts/verify_transfer_tables.py
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
PROTOCOLS = ["A", "B"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]


def discover_rs(root: Path, subdirs=None) -> list[str]:
    """R values that actually have a finished run under ``root``, numerically sorted."""
    found = set()
    for ds in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if subdirs and ds.name not in subdirs:
            continue
        for p in ds.glob("*_random_r*"):
            if (p / "summary.json").exists():
                found.add(p.name.rsplit("_r", 1)[1])
    return sorted(found, key=int)


def recompute(m: dict) -> dict:
    """Every metric of one fit, from its confusion counts, via sklearn."""
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    y = np.array([0] * (tn + fp) + [1] * (fn + tp))
    p = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "mcc": float(matthews_corrcoef(y, p)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "ce": float(m["ce"]), "roc_auc": float(m["roc_auc"])}


def check(cond: bool, msg: str, fails: list) -> None:
    if not cond:
        fails.append(msg)


def audit(d: Path, tol: float, fails: list) -> dict:
    s = json.loads((d / "summary.json").read_text())
    tag = f"{d.parent.name}/{d.name}"
    n_test = s["n_test"]
    out = {}
    for prot in PROTOCOLS:
        pr = s["protocols"][prot]
        clean, pois = pr["clean"], pr["poisoned"]
        check(len(clean) == len(pois) == len(s["contexts"]),
              f"{tag} {prot}: {len(clean)} clean vs {len(pois)} poisoned fits", fails)

        # counts sum to n_test, and stored metrics match sklearn
        rc, rp = [], []
        for side, fits, store in (("clean", clean, rc), ("poisoned", pois, rp)):
            for i, m in enumerate(fits):
                tot = m["tn"] + m["fp"] + m["fn"] + m["tp"]
                check(tot == n_test, f"{tag} {prot} {side}[{i}]: counts sum to {tot} != {n_test}", fails)
                ref = recompute(m)
                for k in ("accuracy", "balanced_accuracy", "mcc", "f1"):
                    check(abs(m[k] - ref[k]) < tol,
                          f"{tag} {prot} {side}[{i}].{k}: stored {m[k]!r} != sklearn {ref[k]!r}", fails)
                store.append(ref)

        # training sizes follow the protocol
        want = round(s["n_context"] * (1 - s["val_ratio"])) if prot == "A" else s["n_context"]
        for i, m in enumerate(clean + pois):
            check(m["n_train"] == want,
                  f"{tag} {prot}[{i}]: n_train {m['n_train']} != {want}", fails)

        # deltas are paired on seed, not averaged
        for i, dl in enumerate(pr["delta"]):
            check(dl["seed"] == pois[i]["seed"] == clean[i]["seed"],
                  f"{tag} {prot} delta[{i}]: seeds not paired", fails)
            for m in METRICS:
                check(abs(dl[m] - (rp[i][m] - rc[i][m])) < tol,
                      f"{tag} {prot} delta[{i}].{m} is not poisoned - clean", fails)

        agg = {}
        for m in METRICS:
            cv = [r[m] for r in rc]
            pv = [r[m] for r in rp]
            dv = [pv[i] - cv[i] for i in range(len(pv))]
            agg[m] = {"clean": (float(np.mean(cv)), float(np.std(cv))),
                      "poisoned": (float(np.mean(pv)), float(np.std(pv))),
                      "delta": (float(np.mean(dv)), float(np.std(dv)))}
            # cross-check against what xgb_transfer.py stored
            for key, got in (("clean_agg", agg[m]["clean"]), ("poisoned_agg", agg[m]["poisoned"]),
                             ("delta_agg", agg[m]["delta"])):
                st = pr[key][m]
                check(abs(st["mean"] - got[0]) < tol and abs(st["std"] - got[1]) < tol,
                      f"{tag} {prot} {key}.{m}: stored {st['mean']!r}+-{st['std']!r} "
                      f"!= recomputed {got[0]!r}+-{got[1]!r}", fails)
        out[prot] = agg

    # the TabPFNv2 side must match the attack run it claims to come from
    adir = Path(s["attack_dir"])
    if (adir / "summary.json").exists():
        asum = json.loads((adir / "summary.json").read_text())
        want_tags = [f"run{t['run_id']}_sub{t['subsample_id']}" for t in asum["per_run_best"]]
        check(s["contexts"] == want_tags,
              f"{tag}: contexts {s['contexts']} != attack per_run_best {want_tags}", fails)
        for m in METRICS:
            dv = [recompute(t["poisoned"])[m] - recompute(t["clean"])[m] for t in asum["per_run_best"]]
            st = s["tabpfn"]["delta_agg"][m]
            check(abs(st["mean"] - float(np.mean(dv))) < tol,
                  f"{tag}: tabpfn delta_agg.{m} {st['mean']!r} != {float(np.mean(dv))!r}", fails)
    else:
        fails.append(f"{tag}: attack_dir {adir} has no summary.json")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/xgb_transfer"))
    ap.add_argument("--csv", type=Path, default=Path("results/tables/transfer/transfer_results.csv"))
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)

    fails: list[str] = []
    got: dict = {}
    n = 0
    RS = discover_rs(a.root)
    for ds_dir in sorted(p for p in a.root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for attack in ATTACKS:
            for r in RS:
                d = ds_dir / f"{attack}_random_r{r}"
                if (d / "summary.json").exists():
                    got[(ds_dir.name, attack, int(r))] = audit(d, a.tol, fails)
                    n += 1
    print(f"audited {n} transfer run directories from {a.root}")

    cells = 0
    if a.csv.exists():
        for row in csv.DictReader(open(a.csv)):
            key = (row["dataset"], row["attack"], int(row["row_percent"]))
            res = got.get(key)
            if res is None:
                fails.append(f"{key}: in CSV but no run directory")
                continue
            agg = res[row["protocol"]][row["metric"]]
            for col, val in (("clean_mean", agg["clean"][0]), ("clean_std", agg["clean"][1]),
                             ("poisoned_mean", agg["poisoned"][0]), ("poisoned_std", agg["poisoned"][1]),
                             ("delta_mean", agg["delta"][0]), ("delta_std", agg["delta"][1])):
                check(math.isclose(float(row[col]), val, rel_tol=0, abs_tol=a.tol),
                      f"{key} {row['protocol']} {row['metric']}.{col}: "
                      f"CSV {row[col]} != recomputed {val!r}", fails)
            if row["relative_pct"]:
                want = 100.0 * agg["delta"][0] / abs(agg["clean"][0])
                check(math.isclose(float(row["relative_pct"]), want, rel_tol=1e-9),
                      f"{key} {row['metric']}.relative_pct: CSV != {want!r}", fails)
            cells += 1
        print(f"cross-checked {cells} CSV cells against {a.csv}")
    else:
        print(f"[warn] {a.csv} not found; run make_transfer_tables.py first")

    if fails:
        print(f"\nFAILED: {len(fails)} check(s)")
        for f in fails[:40]:
            print("  -", f)
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    main()

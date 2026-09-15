#!/usr/bin/env python
"""Build uniform result tables from the poisoning sweep.

Every table has the same columns. Two families are emitted:

  DEPLOYED  -- ordinary fit() + predict_proba(), i.e. what a user actually runs.
               This is the number that matters for "did the attack work".
  OPTIMISER -- fit_with_differentiable_input() + forward(), the path CAPGD
               differentiates through. The attack maximises THIS loss; it is not
               numerically identical to the deployed path, so the two disagree.

Relative change is (new - old) / |old| * 100. For MCC on a weak model |old| is
tiny, so the percentage explodes and is not meaningful -- those cells are marked
with '~' and should be read from the absolute columns instead.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

RES = Path("results")
OUT = Path("results/tables")
METRICS = [("loss", "loss"), ("accuracy", "acc"), ("balanced_accuracy", "bacc"),
           ("mcc", "mcc"), ("roc_auc", "auc")]
MCC_UNSTABLE = 0.15          # |clean mcc| below this -> % change is noise


def load():
    runs = []
    for p in sorted(RES.glob("*/attack/*/attack.json")):
        d = json.loads(p.read_text())
        d["_ds"], d["_name"] = p.parts[1], p.parts[3]
        runs.append(d)
    return runs


def sides(d, path):
    """Return (clean, poisoned) metric dicts including loss, for one path."""
    if path == "deployed":
        dep = d.get("deployed")
        if not dep:
            return None, None
        return dep["clean"], dep["poisoned"]
    clean = dict(d["clean_metrics"]); clean["loss"] = d["clean_loss"]
    pois = dict(d["poisoned_metrics"]); pois["loss"] = d["poisoned_loss"]
    return clean, pois


def row(d, path):
    clean, pois = sides(d, path)
    if clean is None:
        return None
    r = {
        "model": f"TabPFNv2{'/simple' if d['simple'] else ''}/ne{d['n_estimators']}",
        "dataset": d["_ds"],
        "ctx": d["n_context"],
        "test": d["n_test"],
        "rows": d["n_rows"],
        "pct_ctx": 100.0 * d["n_rows"] / d["n_context"],
        "eps": d["eps"],
        "cons": d["constraints"],
        "iters": d["n_iter"],
        "cells_moved": d["n_features_changed"],
        "cells_total": d["n_features"] * d["n_rows"],
    }
    for key, short in METRICS:
        o, n = clean.get(key), pois.get(key)
        if o is None or n is None:
            r[f"{short}_old"] = r[f"{short}_new"] = r[f"{short}_pct"] = None
            continue
        r[f"{short}_old"], r[f"{short}_new"] = o, n
        r[f"{short}_pct"] = (n - o) / abs(o) * 100 if abs(o) > 1e-12 else None
    r["_mcc_unstable"] = abs(clean.get("mcc", 0.0)) < MCC_UNSTABLE
    return r


COLS = ["model", "dataset", "ctx", "test", "rows", "eps", "cons", "iters",
        "loss_old", "loss_new", "loss_pct", "acc_old", "acc_new", "acc_pct",
        "bacc_old", "bacc_new", "bacc_pct", "mcc_old", "mcc_new", "mcc_pct",
        "auc_old", "auc_new", "auc_pct", "cells_moved", "cells_total"]
HDR = {"loss_pct": "loss%", "acc_pct": "acc%", "bacc_pct": "bacc%",
       "mcc_pct": "mcc%", "auc_pct": "auc%", "cells_moved": "moved",
       "cells_total": "ofcells"}


def fmt(r, c):
    v = r.get(c)
    if v is None:
        return "-"
    if c.endswith("_pct"):
        if c == "mcc_pct" and r["_mcc_unstable"]:
            return f"~{v:+.1f}"
        return f"{v:+.1f}"
    if c in ("loss_old", "loss_new"):
        return f"{v:.4f}"
    if c.endswith(("_old", "_new")):
        return f"{v:.4f}"
    if c == "eps":
        return f"{v:g}"
    return str(v)


def table(title, rows):
    print(f"\n{title}")
    w = {c: max(len(HDR.get(c, c)), max((len(fmt(r, c)) for r in rows), default=0))
         for c in COLS}
    print("  " + "  ".join(f"{HDR.get(c, c):>{w[c]}}" for c in COLS))
    print("  " + "  ".join("-" * w[c] for c in COLS))
    for r in rows:
        print("  " + "  ".join(f"{fmt(r, c):>{w[c]}}" for c in COLS))


def main():
    runs = load()
    OUT.mkdir(parents=True, exist_ok=True)
    abl = re.compile(r"abl_rows(\d+)_(full|none)_it100")

    for path, label in (("deployed", "DEPLOYED PATH (ordinary fit + predict_proba -- what a user sees)"),
                        ("optimiser", "OPTIMISER PATH (differentiable path CAPGD maximises)")):
        print("\n" + "=" * 118)
        print(label)
        print("=" * 118)
        allr = [row(d, path) for d in runs]
        allr = [r for r in allr if r]

        # Table 1: single-row budget sweep
        p1 = [r for d, r in zip(runs, [row(d, path) for d in runs])
              if r and (d["_name"].startswith("p1_") or d["_name"].startswith("p2_"))]
        p1.sort(key=lambda r: (r["dataset"], r["cons"], r["eps"], r["iters"]))
        table("TABLE 1 -- one poisoned row: budget and constraint sweep", p1)

        # Tables 2..4: row-count ablation, one per dataset
        for i, ds in enumerate(sorted({r["dataset"] for r in allr}), start=2):
            rs = [r for d, r in zip(runs, [row(d, path) for d in runs])
                  if r and d["_ds"] == ds and abl.fullmatch(d["_name"])]
            rs.sort(key=lambda r: (r["cons"], r["rows"]))
            table(f"TABLE {i} -- {ds}: poisoned-row ablation (100 iterations, eps 0.5)", rs)

    # tidy CSV with every run and both paths
    csv_path = OUT / "all_runs.csv"
    with csv_path.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["path"] + COLS + ["pct_ctx", "run"])
        wtr.writeheader()
        for d in runs:
            for path in ("deployed", "optimiser"):
                r = row(d, path)
                if not r:
                    continue
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                r["path"] = path
                r["run"] = f"{d['_ds']}/{d['_name']}"
                wtr.writerow(r)
    print(f"\n\ntidy CSV (all {len(runs)} runs x 2 paths): {csv_path}")
    print(f"'~' on mcc% marks a clean |MCC| < {MCC_UNSTABLE} where the percentage is noise;")
    print("read those from the absolute mcc_old / mcc_new columns instead.")




def markdown(runs, path="deployed"):
    """Compact markdown: each metric as old -> new (relative %)."""
    import re as _re
    abl = _re.compile(r"abl_rows(\d+)_(full|none)_it100")
    cell = lambda r, s: ("-" if r.get(f"{s}_old") is None else
                         f"{r[f'{s}_old']:.4f} → {r[f'{s}_new']:.4f} "
                         f"({'~' if s=='mcc' and r['_mcc_unstable'] else ''}"
                         f"{r[f'{s}_pct']:+.1f}%)")
    head = ("| model | dataset | ctx | test | rows | eps | cons | iters | "
            "loss old→new (Δ%) | accuracy old→new (Δ%) | bal.acc old→new (Δ%) | "
            "MCC old→new (Δ%) | ROC-AUC old→new (Δ%) | cells moved |")
    sep = "|" + "---|" * 14
    def emit(title, rs):
        print(f"\n**{title}**\n"); print(head); print(sep)
        for r in rs:
            print(f"| {r['model']} | {r['dataset']} | {r['ctx']} | {r['test']} | {r['rows']} | "
                  f"{r['eps']:g} | {r['cons']} | {r['iters']} | "
                  + " | ".join(cell(r, s) for s in ("loss","acc","bacc","mcc","auc"))
                  + f" | {r['cells_moved']}/{r['cells_total']} |")
    p1 = [row(d, path) for d in runs if d["_name"].startswith(("p1_", "p2_"))]
    p1 = [r for r in p1 if r]; p1.sort(key=lambda r: (r["dataset"], r["cons"], r["eps"], r["iters"]))
    emit("Table 1 — one poisoned row: budget and constraint sweep", p1)
    for i, ds in enumerate(sorted({d["_ds"] for d in runs}), start=2):
        rs = [row(d, path) for d in runs if d["_ds"] == ds and abl.fullmatch(d["_name"])]
        rs = [r for r in rs if r]; rs.sort(key=lambda r: (r["cons"], r["rows"]))
        emit(f"Table {i} — {ds}: poisoned-row ablation (100 iterations, eps 0.5)", rs)


if __name__ == "__main__":
    import sys
    if "--markdown" in sys.argv:
        path = "optimiser" if "--optimiser" in sys.argv else "deployed"
        markdown(load(), path)
    else:
        main()

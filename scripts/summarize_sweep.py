#!/usr/bin/env python
"""Summarise the CAPGD poisoning sweep written by runs/run_poison_sweep.sh.

Two views, both reported on the DEPLOYED inference path (ordinary fit/predict_proba)
as well as the optimiser's own differentiable path, because those are not the same
number and only the deployed one reflects what a user would observe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RES = Path("results")


def load():
    out = []
    for p in sorted(RES.glob("*/attack/*/attack.json")):
        d = json.loads(p.read_text())
        d["_ds"] = p.parts[1]
        d["_name"] = p.parts[3]
        out.append(d)
    return out


def cell(d, path, key):
    if path == "opt":
        return {"loss": d["loss_increase"], "mcc": d["metric_delta"]["mcc"],
                "acc": d["metric_delta"]["accuracy"],
                "bacc": d["metric_delta"]["balanced_accuracy"]}[key]
    dep = d.get("deployed")
    if not dep:
        return float("nan")
    return {"loss": dep["delta"]["loss"], "mcc": dep["delta"]["mcc"],
            "acc": dep["delta"]["accuracy"],
            "bacc": dep["delta"]["balanced_accuracy"]}[key]


def phase1(runs):
    print("=" * 100)
    print("PHASE 1 -- one poisoned row, 10 iterations (plus the 100-iteration high-eps run)")
    print("=" * 100)
    order = ["p1_full", "p1_none", "p1_none_eps5", "p1_none_eps50", "p2_none_eps50_it100"]
    label = {"p1_full": "full  eps0.5 it10", "p1_none": "none  eps0.5 it10",
             "p1_none_eps5": "none  eps5   it10", "p1_none_eps50": "none  eps50  it10",
             "p2_none_eps50_it100": "none  eps50  it100"}
    for ds in sorted({r["_ds"] for r in runs}):
        rows = {r["_name"]: r for r in runs if r["_ds"] == ds}
        if not any(n in rows for n in order):
            continue
        base = next(iter(rows.values()))
        print(f"\n{ds}   (clean deployed mcc "
              f"{base.get('deployed', {}).get('clean', {}).get('mcc', float('nan')):.4f})")
        print(f"  {'config':<20s} {'dloss_opt':>10s} {'dmcc_opt':>9s} | "
              f"{'dloss_dep':>10s} {'dacc_dep':>9s} {'dbacc_dep':>10s} {'dmcc_dep':>9s}  {'moved':>7s}")
        for n in order:
            if n not in rows:
                continue
            d = rows[n]
            print(f"  {label[n]:<20s} {cell(d,'opt','loss'):>+10.5f} {cell(d,'opt','mcc'):>+9.4f} | "
                  f"{cell(d,'dep','loss'):>+10.5f} {cell(d,'dep','acc'):>+9.4f} "
                  f"{cell(d,'dep','bacc'):>+10.4f} {cell(d,'dep','mcc'):>+9.4f}  "
                  f"{d['n_features_changed']:>3d}/{d['n_features']*d['n_rows']:<4d}")


def ablation(runs):
    print("\n" + "=" * 100)
    print("PHASE 2+3 -- poisoned rows vs degradation, 100 iterations")
    print("  constrained (full) = schema-valid row;  unconstrained (none) = eps-ball only")
    print("  deployed path; negative dmcc = the attack HURT the model (what we want)")
    print("=" * 100)
    pat = re.compile(r"abl_rows(\d+)_(full|none)_it100")
    for ds in sorted({r["_ds"] for r in runs}):
        got = {}
        for r in runs:
            if r["_ds"] != ds:
                continue
            m = pat.fullmatch(r["_name"])
            if m:
                got[(int(m.group(1)), m.group(2))] = r
        if not got:
            continue
        ns = sorted({n for n, _ in got})
        clean = next(iter(got.values())).get("deployed", {}).get("clean", {})
        print(f"\n{ds}   clean deployed: mcc {clean.get('mcc', float('nan')):.4f}  "
              f"acc {clean.get('accuracy', float('nan')):.4f}  loss {clean.get('loss', float('nan')):.4f}")
        print(f"  {'rows':>5s} {'%ctx':>6s} | {'FULL dloss':>11s} {'FULL dmcc':>10s} | "
              f"{'NONE dloss':>11s} {'NONE dmcc':>10s} | {'none-full dmcc':>15s}")
        for n in ns:
            f, o = got.get((n, "full")), got.get((n, "none"))
            fl = cell(f, "dep", "loss") if f else float("nan")
            fm = cell(f, "dep", "mcc") if f else float("nan")
            nl = cell(o, "dep", "loss") if o else float("nan")
            nm = cell(o, "dep", "mcc") if o else float("nan")
            pct = 100.0 * n / (f or o)["n_context"]
            print(f"  {n:>5d} {pct:>5.1f}% | {fl:>+11.5f} {fm:>+10.4f} | "
                  f"{nl:>+11.5f} {nm:>+10.4f} | {nm-fm:>+15.4f}")


def main():
    runs = load()
    if not runs:
        raise SystemExit("no results/*/attack/*/attack.json yet")
    print(f"{len(runs)} runs found\n")
    phase1(runs)
    ablation(runs)
    print("\nopt = differentiable path the attack optimises through")
    print("dep = deployed fit/predict_proba path (what a user would actually see)")


if __name__ == "__main__":
    main()

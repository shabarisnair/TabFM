#!/usr/bin/env python
"""Tabulate attack.json files produced by attack_context.py.

Reports both the optimiser's own view (differentiable path) and, where present,
the deployed view (ordinary fit/predict_proba) -- these are not the same number,
and the deployed one is what a real user would observe.
"""
import json
from pathlib import Path

rows = []
for p in sorted(Path("results").glob("*/attack_*/attack.json")):
    d = json.loads(p.read_text())
    c, q = d["clean_metrics"], d["poisoned_metrics"]
    dep = d.get("deployed")
    rows.append({
        "dataset": Path(d["train"]).parts[1], "ctx": d["n_context"], "test": d["n_test"],
        "cons": d.get("constraints", "?"),
        "dloss_opt": d["loss_increase"], "dmcc_opt": q["mcc"] - c["mcc"],
        "dloss_dep": dep["delta"]["loss"] if dep else float("nan"),
        "dmcc_dep": dep["delta"]["mcc"] if dep else float("nan"),
        "sec": d["seconds"],
    })

if not rows:
    raise SystemExit("no results/*/attack_*/attack.json found")
hdr = list(rows[0])
w = {h: max(len(h), max(len(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]))
                        for r in rows)) for h in hdr}
print("  ".join(f"{h:>{w[h]}}" for h in hdr))
print("  ".join("-" * w[h] for h in hdr))
for r in rows:
    print("  ".join(f"{(f'{r[h]:+.4f}' if isinstance(r[h], float) and h.startswith('d') else f'{r[h]:.4f}' if isinstance(r[h], float) else str(r[h])):>{w[h]}}"
                    for h in hdr))
print("\n_opt = differentiable path the attack optimises through")
print("_dep = deployed fit/predict_proba path (what a user would actually see)")

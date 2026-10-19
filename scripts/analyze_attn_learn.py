#!/usr/bin/env python
"""Tables of the attention-weight learning runs (scripts/attn_learn.py).

Per draw (dataset, R, V, seed): every configuration's test accuracy at the iteration chosen on
the held-out validation rows -- soft weights (and + V) at the soft choice, the k lowest-ranked rows
deleted (and + V) at the hard choice -- with the precision and AUROC of the ranking there, the
chosen iterations and the time; next to the references of the same draw (poisoned, V only,
V-filter, oracle; each also + V) and the feature-input soft gate and top-k gate of the earlier
runs on the same draw (results/gate_valsplit, results/gate_topk_gen), when they exist.

    python scripts/analyze_attn_learn.py results/attn_learn/screen1 [--sort hard_v]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

COLS = (("soft", "test", "soft"), ("soft_v", "test_plus_v", "soft"), ("hard", "test", "hard"), ("hard_v", "test_plus_v", "hard"))


def earlier_gates(ds: str, R: float, V: int, seed: int) -> dict:
    out = {}
    for name, pat in (("soft gate (input)", f"gate_valsplit/{ds}/*_r{int(R):03d}/gate-valsplit_{ds}_*_R{int(R)}_V{V}s{seed}_H20_*"),
                      ("top-k gate (input)", f"gate_topk_gen/{ds}/*_r{int(R):03d}/gate-topk_{ds}_*_R{int(R)}_V{V}s{seed}_H20_*init0_probe0.05_it300")):
        hits = sorted(Path("results").glob(pat))
        if hits and (hits[0] / "test_metrics.json").exists():
            t = json.loads((hits[0] / "test_metrics.json").read_text())
            m = t["metrics"]
            out[name] = {"soft": m["poisoned_best_gate_soft"]["accuracy"], "hard": m["poisoned_best_gate_hard"]["accuracy"],
                         "prec": t["best_gate_stats"].get("precision_at_R", float("nan"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--sort", default="hard_v", choices=[c[0] for c in COLS] + ["prec"])
    ap.add_argument("--metric", default="acc", choices=["acc", "bacc", "auc", "ce"])
    a = ap.parse_args()
    for root in a.dirs:
        for draw in sorted(p for p in root.iterdir() if p.is_dir()):
            runs = [json.loads(f.read_text()) for f in sorted(draw.glob("*.json")) if f.name != "references.json"]
            if not runs:
                continue
            d = runs[0]["draw"]
            print(f"\n## {draw.name}  (n {d['n']}, k {d['k']}; test {a.metric}, %)")
            ref = json.loads((draw / "references.json").read_text()) if (draw / "references.json").exists() else {}
            if ref:
                print("references: " + "  ".join(f"{k} {100 * v[a.metric]:.1f}" for k, v in ref.items() if isinstance(v, dict))
                      + f"  vfilter precision {ref['vfilter_prec']:.3f}")
            eg = earlier_gates(d["ds"], d["R"], d["V"], d["seed"])
            for name, v in eg.items():
                print(f"{name}: soft {100 * v['soft']:.1f}  hard {100 * v['hard']:.1f}  precision {v['prec']:.3f}")
            rows = []
            for r in runs:
                vals = {c: 100 * r[f"sel_{sel}"][key][a.metric] for c, key, sel in COLS}
                vals["prec"] = r["sel_hard"]["prec"]
                rows.append((r, vals))
            rows.sort(key=lambda t: -t[1][a.sort])
            print(f"| config | soft | soft+V | hard | hard+V | prec@k | AUROC | it soft/hard | min |")
            print("|---|---|---|---|---|---|---|---|---|")
            for r, v in rows:
                from attn_learn import DEFAULTS, cfg_name
                label = cfg_name(DEFAULTS | r["cfg"]).replace("_floor-1000", "").replace(f"_it{r['cfg']['steps']}", "")
                print(f"| {label} | {v['soft']:.1f} | {v['soft_v']:.1f} | {v['hard']:.1f} | {v['hard_v']:.1f} | "
                      f"{v['prec']:.3f} | {r['sel_hard']['auroc']:.3f} | {r['sel_soft']['it']}/{r['sel_hard']['it']} | "
                      f"{r['seconds'] / 60:.1f} |")


if __name__ == "__main__":
    main()

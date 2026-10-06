#!/usr/bin/env python
"""Four-way comparison of what each sifted subset does to TabPFNv2 on the held-out test set.

Detection precision says how clean a subset is; this says whether that translates into a
usable context. Every method's subset is extracted the same way (balanced quota, m=250
and the full budget) and scored the same way, alongside four reference contexts:

    clean      all rows, true labels      the ceiling
    poisoned   all rows, given labels     doing nothing -- the bar to beat
    oracle     poison perfectly removed   what a flawless detector would give
    pure-m     m rows drawn from the CLEAN rows -- same SIZE, perfect purity, so it
               separates "too few rows" from "not pure enough"

    python scripts/make_downstream_comparison.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
ATTACKS = ["label-flip-random", "hub-flip"]
METHODS = ["k-fold", "sift", "meta-sift", "iclad"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SH = {"coil2000_insurance_policies": "coil2000"}


def load_all(small: int):
    kf = json.loads(Path("results/tables/comparison/kfold_downstream.json").read_text())
    rows = []
    for ds in DATASETS:
        for atk in ATTACKS:
            sf = json.loads(Path(f"results/sift/{ds}/{atk}_r030_lr5_it20/summary.json").read_text())
            full = sf["extraction"]["full_budget"]
            ms = json.loads(Path(f"results/meta_sift/{ds}/{atk}_r030_tuned/summary.json").read_text())
            ic = json.loads(Path(f"results/iclad/{ds}/{atk}_r030_default/summary.json").read_text())
            d_s, d_m, d_i = (sf["extraction"]["downstream"], ms["results"]["downstream"],
                             ic["results"]["downstream"])
            k = kf[f"{ds}|{atk}"]
            for m in (small, full):
                base = {"dataset": ds, "attack": atk, "size": m,
                        "is_full_budget": m == full, "n_context": sf["n_context"],
                        "k_flipped": sf["k_flipped"]}
                ctx = {
                    "k-fold": (k[f"balanced_{m}"], k[f"balanced_{m}"]["purity"]),
                    "sift": (d_s[f"balanced_{m}"],
                             sf["extraction"][f"balanced_{m}"]["purity"]),
                    "meta-sift": (d_m[f"balanced_{m}"],
                                  ms["results"][f"balanced_{m}"]["purity"]),
                    "iclad": (d_i[f"balanced_{m}"],
                              ic["results"]["with_label"][f"balanced_{m}"]["purity"]),
                    "pure-m": (d_s[f"pure_{m}"], 1.0),
                    "clean": (d_s["clean"], 1.0),
                    "poisoned": (d_s["poisoned"], 1 - sf["k_flipped"] / sf["n_context"]),
                    "oracle": (d_s["oracle"], 1.0),
                }
                for name, (vals, pur) in ctx.items():
                    rows.append({**base, "context": name, "purity": pur,
                                 **{mm: vals.get(mm) for mm in METRICS}})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/comparison"))
    ap.add_argument("--small", type=int, default=250)
    ap.add_argument("--metric", default="roc_auc")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = load_all(a.small)

    with open(a.out / "downstream_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines = []
    for full_sel, label in ((False, f"top-{a.small}"), (True, "full budget (n - k)")):
        lines.append(f"\n{a.metric.upper()} on the held-out test set -- {label}")
        lines.append("'poisoned' is the do-nothing bar; a method only helps if it beats that.\n")
        order = ["poisoned"] + METHODS + ["pure-m", "oracle", "clean"]
        lines.append(f"  {'dataset':<16s} {'attack':<18s}" + "".join(f"{c:>11s}" for c in order))
        lines.append("  " + "-" * (34 + 11 * len(order)))
        for ds in DATASETS:
            for atk in ATTACKS:
                sel = {r["context"]: r for r in rows if r["dataset"] == ds
                       and r["attack"] == atk and r["is_full_budget"] == full_sel}
                if not sel:
                    continue
                cells = ""
                pois = sel["poisoned"][a.metric]
                for c in order:
                    v = sel[c][a.metric]
                    better = (v < pois) if a.metric == "ce" else (v > pois)
                    mark = "+" if (c in METHODS and better) else " "
                    cells += f"{v:>10.4f}{mark}"
                lines.append(f"  {SH.get(ds, ds):<16s} {atk:<18s}{cells}")
        lines.append("")
        lines.append(f"  {'method':<12s} {'mean ' + a.metric:>12s} {'beats do-nothing':>18s} "
                     f"{'beats pure-m':>14s}")
        for meth in METHODS:
            v = [r[a.metric] for r in rows if r["context"] == meth and r["is_full_budget"] == full_sel]
            wins = 0
            beats_pure = 0
            for ds in DATASETS:
                for atk in ATTACKS:
                    sel = {r["context"]: r for r in rows if r["dataset"] == ds
                           and r["attack"] == atk and r["is_full_budget"] == full_sel}
                    if not sel:
                        continue
                    x, p, pu = sel[meth][a.metric], sel["poisoned"][a.metric], sel["pure-m"][a.metric]
                    wins += (x < p) if a.metric == "ce" else (x > p)
                    beats_pure += (x < pu) if a.metric == "ce" else (x > pu)
            lines.append(f"  {meth:<12s} {sum(v)/len(v):>12.4f} {wins:>15d}/{len(v):<2d} "
                         f"{beats_pure:>11d}/{len(v):<2d}")
    text = "\n".join(lines)
    print(text)
    (a.out / f"downstream_comparison_{a.metric}.txt").write_text(text + "\n")
    print(f"\nwrote {a.out / 'downstream_comparison.csv'} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()

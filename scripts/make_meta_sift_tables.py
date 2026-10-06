#!/usr/bin/env python
"""Consolidate the Meta-Sift baseline runs into one CSV plus a readable table.

Two blocks per (dataset, attack):
  * SIFTING QUALITY of the selected subset -- purity, how much poison it removed, and
    NCR (Meta-Sift's own metric: 100 = no better than random, 0 = perfect, >100 = worse
    than random).
  * DOWNSTREAM: TabPFNv2 on the held-out test set using that subset as the context,
    against clean / poisoned / oracle, and against pure-m -- the same size drawn from
    the CLEAN rows, which separates "too few rows" from "not pure enough".

    python scripts/make_meta_sift_tables.py
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
ATTACKS = ["label-flip-random", "hub-flip"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SHORT = {"coil2000_insurance_policies": "coil2000"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/meta_sift"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/meta_sift"))
    ap.add_argument("--strategy", default="balanced")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rows, lines = [], []
    for ds in DATASETS:
        for atk in ATTACKS:
            f = a.root / ds / f"{atk}_r030_tuned" / "summary.json"
            if not f.exists():
                print(f"[skip] {ds}/{atk}"); continue
            s = json.loads(f.read_text()); r = s["results"]; d = r.get("downstream", {})
            sizes = r["sizes"]
            lines.append(f"\n{SHORT.get(ds, ds)}  /  {atk}  /  R=30%   "
                         f"context {s['n_context']}x{s['n_features']}, {s['k_flipped']} flipped, "
                         f"full budget {r['full_budget']}")
            lines.append(f"  weight-net direction AUC (low weight -> poisoned): "
                         f"{r['weight_auc_low_means_poisoned']:.4f}"
                         + ("   <-- INVERTED" if r["weight_auc_low_means_poisoned"] < 0.5 else ""))
            hdr = (f"  {'subset':<22s} {'rows':>5s} {'purity':>8s} {'poison kept':>12s} {'NCR':>8s}"
                   + "".join(f"{m:>11s}" for m in METRICS))
            lines += ["", hdr, "  " + "-" * (len(hdr) - 2)]

            for name, key in (("clean (all, true y)", "clean"),
                              ("poisoned (no defence)", "poisoned"),
                              ("oracle (poison removed)", "oracle")):
                if key in d:
                    lines.append(f"  {name:<22s} {'-':>5s} {'-':>8s} {'-':>12s} {'-':>8s}"
                                 + "".join(f"{d[key][m]:>11.4f}" for m in METRICS))
            lines.append("  " + "-" * (len(hdr) - 2))

            for m_ in sizes:
                sel = r[f"{a.strategy}_{m_}"]
                dn = d.get(f"{a.strategy}_{m_}", {})
                lines.append(f"  {'meta-sift top-' + str(m_):<22s} {m_:>5d} {sel['purity']:>8.4f} "
                             f"{sel['poison_in_kept']:>12d} {sel['NCR']:>7.1f}%"
                             + "".join(f"{dn.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                pu = d.get(f"pure_{m_}", {})
                if pu:
                    lines.append(f"  {'  pure-' + str(m_) + ' (control)':<22s} {m_:>5d} {1.0:>8.4f} "
                                 f"{0:>12d} {0.0:>7.1f}%"
                                 + "".join(f"{pu.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                rows.append({
                    "dataset": ds, "attack": atk, "row_percent": 30, "method": "meta-sift",
                    "strategy": a.strategy, "size": m_, "is_full_budget": m_ == r["full_budget"],
                    "n_context": s["n_context"], "k_flipped": s["k_flipped"],
                    "purity": sel["purity"], "poison_in_kept": sel["poison_in_kept"],
                    "poison_removed": sel["poison_removed"], "poison_recall": sel["poison_recall"],
                    "clean_discarded": sel["clean_discarded"], "NCR": sel["NCR"],
                    "weight_auc": r["weight_auc_low_means_poisoned"],
                    **{f"sift_{mm}": dn.get(mm) for mm in METRICS},
                    **{f"pure_{mm}": pu.get(mm) for mm in METRICS},
                    **{f"clean_{mm}": d.get("clean", {}).get(mm) for mm in METRICS},
                    **{f"poisoned_{mm}": d.get("poisoned", {}).get(mm) for mm in METRICS},
                    **{f"oracle_{mm}": d.get("oracle", {}).get(mm) for mm in METRICS},
                })
            lines.append("")

    text = "\n".join(lines)
    print(text)
    (a.out / "meta_sift_results.txt").write_text(text + "\n")
    with open(a.out / "meta_sift_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out / 'meta_sift_results.csv'} ({len(rows)} rows) "
          f"and {a.out / 'meta_sift_results.txt'}")
    return rows


if __name__ == "__main__":
    main()

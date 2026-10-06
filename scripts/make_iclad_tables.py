#!/usr/bin/env python
"""Consolidate the ICLAD baseline runs into one CSV plus a readable table.

Reports, per (dataset, attack): the detection AUC of the anomaly score, the purity /
NCR of the kept subset at each size, and TabPFNv2's metrics using that subset as the
context. The features-only control is carried alongside -- the attacks change labels
only, so it shows what the appended label column actually contributes.

    python scripts/make_iclad_tables.py
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
ATTACKS = ["label-flip-random", "hub-flip"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SH = {"coil2000_insurance_policies": "coil2000"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/iclad"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/iclad"))
    ap.add_argument("--strategy", default="balanced")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rows, lines = [], []
    for ds in DATASETS:
        for atk in ATTACKS:
            f = a.root / ds / f"{atk}_r030_default" / "summary.json"
            if not f.exists():
                print(f"[skip] {ds}/{atk}"); continue
            s = json.loads(f.read_text()); r = s["results"]; d = r.get("downstream", {})
            wl, fo = r["with_label"], r["features_only"]
            lines.append(f"\n{SH.get(ds, ds)}  /  {atk}  /  R=30%   context {s['n_context']}x"
                         f"{s['n_features']}, {s['k_flipped']} flipped, full budget {s['full_budget']}")
            lines.append(f"  AUC(anomaly -> flipped):  with label {wl['detect_auc_anomaly_means_flipped']:.4f}"
                         f"   features only {fo['detect_auc_anomaly_means_flipped']:.4f}"
                         f"   (0.5 = no signal, <0.5 = anti-correlated)")
            hdr = (f"  {'subset':<24s} {'rows':>5s} {'purity':>8s} {'poison kept':>12s} {'NCR':>8s}"
                   + "".join(f"{m:>11s}" for m in METRICS))
            lines += ["", hdr, "  " + "-" * (len(hdr) - 2)]
            for name, key in (("clean (all, true y)", "clean"), ("poisoned (no defence)", "poisoned"),
                              ("oracle (poison removed)", "oracle")):
                if key in d:
                    lines.append(f"  {name:<24s} {'-':>5s} {'-':>8s} {'-':>12s} {'-':>8s}"
                                 + "".join(f"{d[key][m]:>11.4f}" for m in METRICS))
            lines.append("  " + "-" * (len(hdr) - 2))
            for m_ in s["sizes"]:
                sel, dn = wl[f"{a.strategy}_{m_}"], d.get(f"{a.strategy}_{m_}", {})
                lines.append(f"  {'ICLAD top-' + str(m_):<24s} {m_:>5d} {sel['purity']:>8.4f} "
                             f"{sel['poison_in_kept']:>12d} {sel['NCR']:>7.1f}%"
                             + "".join(f"{dn.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                ctl = fo[f"{a.strategy}_{m_}"]
                lines.append(f"  {'  features-only ctl':<24s} {m_:>5d} {ctl['purity']:>8.4f} "
                             f"{ctl['poison_in_kept']:>12d} {ctl['NCR']:>7.1f}%" + " " * 55)
                pu = d.get(f"pure_{m_}", {})
                if pu:
                    lines.append(f"  {'  pure-m (control)':<24s} {m_:>5d} {1.0:>8.4f} {0:>12d} {0.0:>7.1f}%"
                                 + "".join(f"{pu.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                rows.append({
                    "dataset": ds, "attack": atk, "row_percent": 30, "method": "iclad",
                    "strategy": a.strategy, "size": m_, "is_full_budget": m_ == s["full_budget"],
                    "n_context": s["n_context"], "k_flipped": s["k_flipped"],
                    "detect_auc_with_label": wl["detect_auc_anomaly_means_flipped"],
                    "detect_auc_features_only": fo["detect_auc_anomaly_means_flipped"],
                    "purity": sel["purity"], "poison_in_kept": sel["poison_in_kept"],
                    "poison_removed": sel["poison_removed"], "poison_recall": sel["poison_recall"],
                    "clean_discarded": sel["clean_discarded"], "NCR": sel["NCR"],
                    "purity_features_only": ctl["purity"], "NCR_features_only": ctl["NCR"],
                    **{f"iclad_{mm}": dn.get(mm) for mm in METRICS},
                    **{f"pure_{mm}": pu.get(mm) for mm in METRICS},
                    **{f"clean_{mm}": d.get("clean", {}).get(mm) for mm in METRICS},
                    **{f"poisoned_{mm}": d.get("poisoned", {}).get(mm) for mm in METRICS},
                    **{f"oracle_{mm}": d.get("oracle", {}).get(mm) for mm in METRICS},
                })
            lines.append("")
    text = "\n".join(lines)
    print(text)
    (a.out / "iclad_results.txt").write_text(text + "\n")
    with open(a.out / "iclad_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out / 'iclad_results.csv'} ({len(rows)} rows) and {a.out / 'iclad_results.txt'}")
    return rows


if __name__ == "__main__":
    main()

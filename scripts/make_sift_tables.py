#!/usr/bin/env python
"""Consolidate our sift runs into one CSV plus a readable table.

Same layout as make_meta_sift_tables.py and make_iclad_tables.py, so the three methods
line up column for column:

  * SIFTING QUALITY of the kept subset -- purity, poison kept/removed, and NCR
    (100 = no better than random, 0 = perfect, >100 = worse than random).
  * DOWNSTREAM -- TabPFNv2 on the held-out test set with that subset as the context,
    against clean / poisoned / oracle, and against pure-m (the same size drawn from the
    CLEAN rows), which separates "too few rows" from "not pure enough".

    python scripts/make_sift_tables.py
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
    ap.add_argument("--root", type=Path, default=Path("results/sift"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/sift"))
    ap.add_argument("--lr", default="5")
    ap.add_argument("--n-iter", default="20")
    ap.add_argument("--strategy", default="balanced")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rows, lines = [], []
    for ds in DATASETS:
        for atk in ATTACKS:
            f = a.root / ds / f"{atk}_r030_lr{a.lr}_it{a.n_iter}" / "summary.json"
            if not f.exists():
                print(f"[skip] {ds}/{atk}"); continue
            s = json.loads(f.read_text()); e = s["extraction"]; d = e.get("downstream", {})
            tr = s["trace"]
            peak = max(r[f"precision_{a.strategy}"] for r in tr)
            peak_it = max(range(len(tr)), key=lambda i: tr[i][f"precision_{a.strategy}"])
            lines.append(f"\n{SH.get(ds, ds)}  /  {atk}  /  R=30%   context {s['n_context']}x"
                         f"{s['n_features']}, {s['k_flipped']} flipped, full budget {e['full_budget']}"
                         f"   lr={s['lr']}, {s['n_iter']} iters")
            lines.append(f"  full-budget precision: final {tr[-1][f'precision_{a.strategy}']:.4f}, "
                         f"peak {peak:.4f} @iter {peak_it}   (extraction uses iter "
                         f"{e['extract_iter']})")
            hdr = (f"  {'subset':<24s} {'rows':>5s} {'purity':>8s} {'poison kept':>12s} {'NCR':>8s}"
                   + "".join(f"{m:>11s}" for m in METRICS))
            lines += ["", hdr, "  " + "-" * (len(hdr) - 2)]
            for name, key in (("clean (all, true y)", "clean"), ("poisoned (no defence)", "poisoned"),
                              ("oracle (poison removed)", "oracle")):
                if key in d:
                    lines.append(f"  {name:<24s} {'-':>5s} {'-':>8s} {'-':>12s} {'-':>8s}"
                                 + "".join(f"{d[key][m]:>11.4f}" for m in METRICS))
            lines.append("  " + "-" * (len(hdr) - 2))
            for m_ in e["sizes"]:
                sel, dn = e[f"{a.strategy}_{m_}"], d.get(f"{a.strategy}_{m_}", {})
                lines.append(f"  {'sift top-' + str(m_):<24s} {m_:>5d} {sel['purity']:>8.4f} "
                             f"{sel['poison_in_kept']:>12d} {sel['NCR']:>7.1f}%"
                             + "".join(f"{dn.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                pu = d.get(f"pure_{m_}", {})
                if pu:
                    lines.append(f"  {'  pure-m (control)':<24s} {m_:>5d} {1.0:>8.4f} {0:>12d} {0.0:>7.1f}%"
                                 + "".join(f"{pu.get(mm, float('nan')):>11.4f}" for mm in METRICS))
                rows.append({
                    "dataset": ds, "attack": atk, "row_percent": 30, "method": "sift",
                    "lr": s["lr"], "n_iter": s["n_iter"], "extract_iter": e["extract_iter"],
                    "strategy": a.strategy, "size": m_, "is_full_budget": m_ == e["full_budget"],
                    "n_context": s["n_context"], "k_flipped": s["k_flipped"],
                    "purity": sel["purity"], "poison_in_kept": sel["poison_in_kept"],
                    "poison_removed": sel["poison_removed"], "poison_recall": sel["poison_recall"],
                    "clean_discarded": sel["clean_discarded"], "NCR": sel["NCR"],
                    "removal_precision": sel["removal_precision"],
                    "peak_precision": peak, "peak_iter": peak_it,
                    **{f"sift_{mm}": dn.get(mm) for mm in METRICS},
                    **{f"pure_{mm}": pu.get(mm) for mm in METRICS},
                    **{f"clean_{mm}": d.get("clean", {}).get(mm) for mm in METRICS},
                    **{f"poisoned_{mm}": d.get("poisoned", {}).get(mm) for mm in METRICS},
                    **{f"oracle_{mm}": d.get("oracle", {}).get(mm) for mm in METRICS},
                })
            lines.append("")
    if not rows:
        raise SystemExit(f"no finished runs under {a.root}")
    text = "\n".join(lines)
    print(text)
    (a.out / "sift_results.txt").write_text(text + "\n")
    with open(a.out / "sift_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out / 'sift_results.csv'} ({len(rows)} rows) and {a.out / 'sift_results.txt'}")
    return rows


if __name__ == "__main__":
    main()

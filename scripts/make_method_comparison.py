#!/usr/bin/env python
"""Four-way comparison of clean-subset identification at R=30%.

Methods
    k-fold      scripts/defend_context.py  -- rank rows by p(their own label) with the
                row held out of the context; deterministic given the folds, so the five
                fold draws of poison draw 0 are averaged.
    sift        scripts/sift_context.py    -- learn a soft train/query split by
                maximising the held-out loss through TabPFNv2's row attention (lr=5).
    meta-sift   Zeng et al. USENIX'23, tuned backbone/epochs for tabular data.
    ICLAD       Wei & Armanfard, pretrained unsupervised checkpoint, label appended as
                a feature.

All four are scored on the SAME poisoned row set (poison draw 0, row_seed 1), verified
identical, so differences are entirely due to the method.

METRIC. "Detection precision" is the precision of the set each method calls CLEAN, i.e.
the purity of the kept subset: of the m rows kept, how many were not flipped. It is
defined at any m, which removal precision is not -- at m=250 a method discards ~4750
rows of which only 1500 can be poison, so removal precision is bounded by the budget
rather than by skill. At the full budget the two coincide up to a constant.

Chance is 1-R = 0.70 at every m: that is what keeping m rows at random yields.

    python scripts/make_method_comparison.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from tabfm_experiments.sift import select_clean_subset, subset_report  # noqa: E402

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
ATTACKS = ["label-flip-random", "hub-flip"]
METHODS = ["k-fold", "sift", "meta-sift", "iclad"]
SH = {"coil2000_insurance_policies": "coil2000"}


def kfold_scores(ds: str, atk: str) -> list[np.ndarray]:
    """k-fold's per-row score for poison draw 0, one array per fold draw.

    The score is p(row carries its own label): HIGHER means more likely clean, which is
    the same orientation as every other method here.
    """
    out = []
    for f in sorted(glob.glob(f"results/defense_kfold/{ds}/{atk}_r030/p0_f*_scores.npz")):
        out.append(np.load(f))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/comparison"))
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--small", type=int, default=250)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in DATASETS:
        for atk in ATTACKS:
            # geometry from any method's record
            sf = json.loads(Path(f"results/sift/{ds}/{atk}_r030_lr5_it20/summary.json").read_text())
            n, k, full = sf["n_context"], sf["k_flipped"], sf["extraction"]["full_budget"]
            sizes = [a.small, full]

            scored: dict[str, dict[int, tuple[float, float]]] = {}

            # --- k-fold: recompute from the saved per-row scores, mean over fold draws
            zs = kfold_scores(ds, atk)
            if zs:
                per = {m: [] for m in sizes}
                for z in zs:
                    s, yg, flip = z["scores"], z["y_given"].astype(int), z["flipped_indices"]
                    for m in sizes:
                        kept = select_clean_subset(s, yg, m, a.strategy)
                        per[m].append(subset_report(kept, flip, n)["purity"])
                scored["k-fold"] = {m: (float(np.mean(per[m])), float(np.std(per[m]))) for m in sizes}

            # --- the other three: read the purity already recorded
            for meth, path, block in (
                ("sift", f"results/sift/{ds}/{atk}_r030_lr5_it20/summary.json", "extraction"),
                ("meta-sift", f"results/meta_sift/{ds}/{atk}_r030_tuned/summary.json", "results"),
                ("iclad", f"results/iclad/{ds}/{atk}_r030_default/summary.json", "results"),
            ):
                p = Path(path)
                if not p.exists():
                    continue
                s = json.loads(p.read_text())
                blk = s[block]
                if meth == "iclad":
                    blk = blk["with_label"]
                scored[meth] = {m: (blk[f"{a.strategy}_{m}"]["purity"], 0.0) for m in sizes}

            for meth in METHODS:
                if meth not in scored:
                    continue
                for m in sizes:
                    mu, sd = scored[meth][m]
                    rows.append({
                        "dataset": ds, "attack": atk, "method": meth, "size": m,
                        "is_full_budget": m == full, "n_context": n, "k_flipped": k,
                        "chance": 1 - k / n, "purity": mu, "purity_std": sd,
                        "poison_kept": int(round((1 - mu) * m)),
                        "NCR": (1 - mu) / (k / n) * 100,
                        "lift_over_chance": mu - (1 - k / n),
                    })

    with open(a.out / "detection_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- readable tables ------------------------------------------------------------
    lines = []
    for m_label, full_sel in ((f"top-{a.small}", False), ("full budget (n - k)", True)):
        lines.append(f"\nDETECTION PRECISION (purity of the kept subset) -- {m_label}")
        lines.append(f"chance = 0.7000 at every size; the oracle keeps 1.0000\n")
        lines.append(f"  {'dataset':<16s} {'attack':<18s}" + "".join(f"{x:>12s}" for x in METHODS) + "   best")
        lines.append("  " + "-" * 82)
        for ds in DATASETS:
            for atk in ATTACKS:
                sel = {r["method"]: r for r in rows
                       if r["dataset"] == ds and r["attack"] == atk and r["is_full_budget"] == full_sel}
                if not sel:
                    continue
                best = max(sel, key=lambda k_: sel[k_]["purity"])
                cells = ""
                for meth in METHODS:
                    if meth in sel:
                        v = sel[meth]["purity"]
                        cells += f"{v:>11.4f}" + ("*" if meth == best else " ")
                    else:
                        cells += f"{'-':>12s}"
                lines.append(f"  {SH.get(ds, ds):<16s} {atk:<18s}{cells}   {best}")
        lines.append("")
        # per-method summary
        lines.append(f"  {'method':<12s} {'mean purity':>12s} {'cells > chance':>15s} {'cells > 0.9':>12s}")
        for meth in METHODS:
            v = [r["purity"] for r in rows if r["method"] == meth and r["is_full_budget"] == full_sel]
            if not v:
                continue
            ch = [r for r in rows if r["method"] == meth and r["is_full_budget"] == full_sel
                  and r["purity"] > r["chance"]]
            lines.append(f"  {meth:<12s} {np.mean(v):>12.4f} {len(ch):>12d}/{len(v):<2d} "
                         f"{sum(1 for x in v if x > 0.9):>9d}/{len(v):<2d}")
    text = "\n".join(lines)
    print(text)
    (a.out / "detection_comparison.txt").write_text(text + "\n")
    print(f"\nwrote {a.out / 'detection_comparison.csv'} ({len(rows)} rows) "
          f"and {a.out / 'detection_comparison.txt'}")
    return rows


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Per-head anatomy of one layer under hub label flipping.

Answers three questions for each of the 6 heads of a given layer (default L9, the
paper's readout layer), on the target column:

  SHARP?      max attention weight and entropy, clean vs poisoned
  CAPTURED?   attention mass on the flipped rows, clean vs poisoned, and whether the
              head's own attention-weighted vote goes from right to wrong
  ROBUST?     how much the distribution itself moved: total-variation distance and
              Pearson correlation between the clean and poisoned attention vectors

The per-head vote is the paper's rule restricted to one head:
    p_h(c) = sum_i alpha_i^(h) * 1[y_i = c]
evaluated with the labels each condition actually has (clean y, or flipped y).

    python scripts/analyze_l9_heads.py --row-percent 10
    python scripts/analyze_l9_heads.py --row-percent 10 --layer 11
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from tabfm_experiments.config import DATASETS_DIR  # noqa: E402
from tabfm_experiments.data import load_split  # noqa: E402


def preds(d: Path):
    y, p = [], []
    for r in csv.DictReader(open(d / "predictions.csv")):
        y.append(int(r["y_true"]))
        p.append(int(r["pred"]))
    return np.array(y), np.array(p)


def vote_pred(a: np.ndarray, y_ctx: np.ndarray) -> np.ndarray:
    """Attention-weighted label vote -> predicted class, per query."""
    p1 = a[:, y_ctx == 1].sum(-1)
    return (p1 > a[:, y_ctx == 0].sum(-1)).astype(int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/hdd1/ssn899/TabFM/results/hub_flip_attention"))
    ap.add_argument("--datasets", nargs="+", default=["url_unique", "coil2000_insurance_policies"])
    ap.add_argument("--row-percent", type=int, default=10)
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("results/tables/hub_attention"))
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        d0, d1 = a.root / ds / "r000", a.root / ds / f"r{a.row_percent:03d}"
        if not (d1 / "summary.json").exists():
            print(f"[skip] {ds}")
            continue
        att = np.array(json.loads((d1 / "summary.json").read_text())["flipped_indices"])
        _, y_ctx, _ = load_split(DATASETS_DIR / ds / "splits/selected/natural/context_5000.csv")
        y_ctx = y_ctx.astype(int)
        y_pois = y_ctx.copy()
        y_pois[att] = 1 - y_pois[att]
        y, pc = preds(d0 / "run0")
        _, pp = preds(d1 / "run0")
        kept = np.flatnonzero((pc == y) & (pp == y))
        broken = np.flatnonzero((pc == y) & (pp != y))
        mask = np.zeros(len(y_ctx), bool)
        mask[att] = True

        A0 = np.load(d0 / "run0" / "attention.npz")["attn_between_rows_target_full"][a.layer]
        A1 = np.load(d1 / "run0" / "attention.npz")["attn_between_rows_target_full"][a.layer]
        print(f"\n########## {ds}   L{a.layer}   R={a.row_percent}%  "
              f"({len(att)} of {len(y_ctx)} rows flipped = {100*len(att)/len(y_ctx):.0f}%, "
              f"uniform mass on them = {len(att)/len(y_ctx):.3f})")
        print(f"  kept {len(kept)}   broken {len(broken)}")
        hdr = (f"  {'head':>4s} {'group':>6s} | {'agree cln':>9s} {'agree pois':>10s} "
               f"{'mislead':>7s} | {'retained':>8s} {'corr':>6s} | "
               f"{'massFlip cln':>12s} {'pois':>6s} | {'vote cln':>8s} {'vote pois':>9s}")
        print("\n" + hdr + "\n  " + "-" * (len(hdr) - 2))
        for h in range(A0.shape[0]):
            c = A0[h].astype(np.float32)
            p = A1[h].astype(np.float32)
            vc, vp = vote_pred(c, y_ctx), vote_pred(p, y_pois)
            for name, idx in (("kept", kept), ("broken", broken)):
                if len(idx) == 0:
                    continue
                ci, pi = c[idx], p[idx]
                tv = float((0.5 * np.abs(ci - pi).sum(-1)).mean())
                cor = float(np.mean([np.corrcoef(ci[j], pi[j])[0, 1] for j in range(len(idx))]))
                # "agreeing" context row = its CURRENT label equals the query's true label.
                # A flipped row that used to agree now disagrees, so this is the quantity
                # the vote actually integrates.
                yq = y[idx][:, None]
                ag_c = float((ci * (y_ctx[None, :] == yq)).sum(-1).mean())
                ag_p = float((pi * (y_pois[None, :] == yq)).sum(-1).mean())
                # of the poisoned mass, how much sits on flipped rows that now MISLEAD
                mis = float((pi * (mask[None, :] & (y_pois[None, :] != yq))).sum(-1).mean())
                # mass whose placement survived the attack: sum_i min(clean_i, pois_i)
                kept_mass = float(np.minimum(ci, pi).sum(-1).mean())
                r = {"dataset": ds, "layer": a.layer, "head": h, "group": name, "n": len(idx),
                     "max_clean": float(ci.max(-1).mean()), "max_pois": float(pi.max(-1).mean()),
                     "ent_clean": float(-(ci * np.log(np.clip(ci, 1e-12, None))).sum(-1).mean()),
                     "ent_pois": float(-(pi * np.log(np.clip(pi, 1e-12, None))).sum(-1).mean()),
                     "mass_clean": float(ci[:, mask].sum(-1).mean()),
                     "mass_pois": float(pi[:, mask].sum(-1).mean()),
                     "tv_distance": tv, "corr": cor, "retained_mass": kept_mass,
                     "agree_mass_clean": ag_c, "agree_mass_pois": ag_p,
                     "misleading_flip_mass_pois": mis,
                     "vote_acc_clean": float((vc[idx] == y[idx]).mean()),
                     "vote_acc_pois": float((vp[idx] == y[idx]).mean())}
                rows.append(r)
                print(f"  {h:>4d} {name:>6s} | {ag_c:>9.3f} {ag_p:>10.3f} {mis:>7.3f} | "
                      f"{kept_mass:>8.3f} {cor:>6.3f} | "
                      f"{r['mass_clean']:>12.3f} {r['mass_pois']:>6.3f} | "
                      f"{r['vote_acc_clean']:>8.3f} {r['vote_acc_pois']:>9.3f}")
        del A0, A1

    if rows:
        f = a.out / f"l{a.layer}_heads_r{a.row_percent:03d}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {f}")
    return rows


if __name__ == "__main__":
    main()

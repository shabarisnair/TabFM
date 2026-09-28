#!/usr/bin/env python
"""Why does hub flipping break predictions? Compare clean vs poisoned attention.

For a chosen R, splits the query rows by outcome

    kept    clean model right  ->  poisoned model right
    broken  clean model right  ->  poisoned model WRONG

and, for each group, compares the between-row attention of the TARGET column (the y
slot, the one the readout vote reads) before and after the attack.

Reported per layer:
  max          mean of the per-query maximum attention weight  (the "sharpness" claim)
  entropy      mean attention entropy over the 5000 context rows
  mass_att     attention mass the query places on the ATTACKED (flipped) context rows
  top1_flip    fraction of queries whose single most-attended row is an attacked row
  jaccard@32   overlap of the clean vs poisoned top-32 attended rows

``--heads mean`` reproduces the head-averaged view; ``--heads each`` reports every head
separately, which is how the paper states its sharpness result.

    python scripts/analyze_hub_attention.py --row-percent 10
    python scripts/analyze_hub_attention.py --row-percent 10 --layers 9 11 --heads each
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_preds(d: Path):
    y, pred = [], []
    with open(d / "predictions.csv") as fh:
        for r in csv.DictReader(fh):
            y.append(int(r["y_true"]))
            pred.append(int(r["pred"]))
    return np.array(y), np.array(pred)


def entropy(a: np.ndarray) -> np.ndarray:
    return -(a * np.log(np.clip(a, 1e-12, None))).sum(-1)


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    ia = np.argpartition(-a, k, axis=-1)[:, :k]
    ib = np.argpartition(-b, k, axis=-1)[:, :k]
    out = np.empty(len(ia))
    for i in range(len(ia)):
        sa, sb = set(ia[i].tolist()), set(ib[i].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def stats(ac: np.ndarray, ap: np.ndarray, attacked: np.ndarray, k: int) -> dict:
    """ac / ap: [n_query, n_ctx] clean and poisoned attention for one group."""
    mask = np.zeros(ac.shape[-1], dtype=bool)
    mask[attacked] = True
    return {
        "n": len(ac),
        "max_clean": float(ac.max(-1).mean()), "max_pois": float(ap.max(-1).mean()),
        "ent_clean": float(entropy(ac).mean()), "ent_pois": float(entropy(ap).mean()),
        "mass_clean": float(ac[:, mask].sum(-1).mean()),
        "mass_pois": float(ap[:, mask].sum(-1).mean()),
        "top1_flip_clean": float(mask[ac.argmax(-1)].mean()),
        "top1_flip_pois": float(mask[ap.argmax(-1)].mean()),
        "jaccard": float(jaccard_topk(ac, ap, k).mean()),
    }


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--root", type=Path,
                     default=Path("/hdd1/ssn899/TabFM/results/hub_flip_attention"))
    ap_.add_argument("--datasets", nargs="+",
                     default=["url_unique", "coil2000_insurance_policies"])
    ap_.add_argument("--row-percent", type=int, default=10)
    ap_.add_argument("--layers", nargs="+", type=int, default=list(range(12)))
    ap_.add_argument("--heads", default="mean", choices=["mean", "each"])
    ap_.add_argument("--topk", type=int, default=32, help="k for the jaccard overlap")
    ap_.add_argument("--out", type=Path, default=Path("results/tables/hub_attention"))
    a = ap_.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for ds in a.datasets:
        d0 = a.root / ds / "r000"
        d1 = a.root / ds / f"r{a.row_percent:03d}"
        if not (d0 / "summary.json").exists() or not (d1 / "summary.json").exists():
            print(f"[skip] {ds}: need both r000 and r{a.row_percent:03d}")
            continue
        attacked = np.array(json.loads((d1 / "summary.json").read_text())["flipped_indices"])
        y, pc = load_preds(d0 / "run0")
        _, pp = load_preds(d1 / "run0")
        kept = np.flatnonzero((pc == y) & (pp == y))
        broken = np.flatnonzero((pc == y) & (pp != y))
        print(f"\n########## {ds}   R={a.row_percent}%  ({len(attacked)} rows flipped)")
        print(f"  clean correct {int((pc == y).sum())}/{len(y)}   "
              f"-> kept {len(kept)}   broken {len(broken)}   "
              f"(poisoned correct {int((pp == y).sum())})")

        A0 = np.load(d0 / "run0" / "attention.npz")["attn_between_rows_target_full"]
        A1 = np.load(d1 / "run0" / "attention.npz")["attn_between_rows_target_full"]
        H = A0.shape[1]
        heads = [None] if a.heads == "mean" else list(range(H))

        hdr = (f"  {'layer':>5s} {'head':>4s} {'group':>7s} {'n':>5s} | "
               f"{'max clean':>9s} {'max pois':>9s} | {'ent clean':>9s} {'ent pois':>8s} | "
               f"{'mass_att clean':>14s} {'mass_att pois':>13s} | {'top1_att pois':>13s} {'jac@%d' % a.topk:>7s}")
        print("\n" + hdr)
        print("  " + "-" * (len(hdr) - 2))
        for L in a.layers:
            for h in heads:
                ac_all = (A0[L].mean(0) if h is None else A0[L, h]).astype(np.float32)
                ap_all = (A1[L].mean(0) if h is None else A1[L, h]).astype(np.float32)
                for name, idx in (("kept", kept), ("broken", broken)):
                    if len(idx) == 0:
                        continue
                    s = stats(ac_all[idx], ap_all[idx], attacked, a.topk)
                    print(f"  {'L'+str(L):>5s} {('avg' if h is None else h):>4} {name:>7s} {s['n']:>5d} | "
                          f"{s['max_clean']:>9.4f} {s['max_pois']:>9.4f} | "
                          f"{s['ent_clean']:>9.3f} {s['ent_pois']:>8.3f} | "
                          f"{s['mass_clean']:>14.4f} {s['mass_pois']:>13.4f} | "
                          f"{s['top1_flip_pois']:>13.3f} {s['jaccard']:>7.3f}")
                    allrows.append({"dataset": ds, "row_percent": a.row_percent, "layer": L,
                                    "head": "mean" if h is None else h, "group": name, **s})
        del A0, A1

    if allrows:
        f = a.out / f"hub_attention_r{a.row_percent:03d}_{a.heads}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {f} ({len(allrows)} rows)")
    return allrows


if __name__ == "__main__":
    main()

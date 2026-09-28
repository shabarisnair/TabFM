#!/usr/bin/env python
"""Prepare the UCI Credit Approval dataset (690 rows) with a simple train/test split.

Deliberately NOT the protocol used for the large datasets: at this size there is no
room for a validation split, context subsampling or context selection. The recipe is

  1. drop ill-formed rows   (any missing value)
  2. drop duplicate rows    (exact, and feature-duplicates carrying opposite labels)
  3. one random stratified split: 30% test, 70% train

Produces, in datasets/credit-approval/:
  credit-approval_raw_strings.csv   untouched copy of the file as delivered
  credit-approval.csv               numeric, target `class` encoded 0/1
  credit-approval_metadata.csv      feature,type,mutable,min,max
  encoding_codebook.json            every categorical mapping + what was dropped
  splits/train.csv, splits/test.csv, splits/split_info.json

    python scripts/prepare_credit_approval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "class"
# '+' (credit approved) is the minority class, so it takes 1 -- matching the
# "class 1 is the event of interest" convention used by the other datasets.
TARGET_MAP = {"+": 1, "-": 0}


def col_type(s: pd.Series) -> str:
    """cat for strings, int when every value is integral, real otherwise."""
    if not pd.api.types.is_numeric_dtype(s):
        return "cat"
    v = s.dropna().to_numpy()
    return "int" if np.all(np.equal(np.mod(v, 1), 0)) else "real"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/credit-approval"))
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stratify", action="store_true",
                    help="plain random split instead of one stratified by the target")
    a = ap.parse_args(argv)

    backup = a.dir / "credit-approval_raw_strings.csv"
    if not backup.exists():
        pd.read_csv(a.dir / "credit-approval.csv").to_csv(backup, index=False)
        print(f"  backed up the delivered file -> {backup}")
    raw = pd.read_csv(backup)
    n0 = len(raw)
    feats = [c for c in raw.columns if c != TARGET]

    # 1. ill-formed = any missing value in any column
    ill = raw.isna().any(axis=1)
    df = raw.loc[~ill].reset_index(drop=True)
    n_ill = int(ill.sum())

    # 2. duplicates: exact first, then any feature-row whose label is contradicted
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    contradictory = df.groupby(feats, dropna=False)[TARGET].transform("nunique") > 1
    n_contra = int(contradictory.sum())
    df = df.loc[~contradictory].reset_index(drop=True)

    types = {c: col_type(df[c]) for c in feats}
    codebook: dict = {}
    for c in feats:
        if types[c] == "cat":
            levels = sorted(df[c].astype(str).unique())
            mapping = {lv: i for i, lv in enumerate(levels)}
            codebook[c] = mapping
            df[c] = df[c].astype(str).map(mapping).astype(int)
    bad = set(df[TARGET].unique()) - set(TARGET_MAP)
    if bad:
        raise SystemExit(f"unexpected target values {bad}; expected {sorted(TARGET_MAP)}")
    codebook[TARGET] = TARGET_MAP
    df[TARGET] = df[TARGET].map(TARGET_MAP).astype(int)
    df = df[[*feats, TARGET]]
    df.to_csv(a.dir / "credit-approval.csv", index=False)

    pd.DataFrame({
        "feature": feats,
        "type": [types[c] for c in feats],
        "mutable": [True] * len(feats),      # no immutability information is published
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    }).to_csv(a.dir / "credit-approval_metadata.csv", index=False)

    # 3. one random split
    rng = np.random.default_rng(a.seed)
    n = len(df)
    n_test = int(round(n * a.test_frac))
    if a.no_stratify:
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])
    else:                                     # stratified: same class ratio in both sides
        test_idx = []
        for cls in (0, 1):
            idx = np.flatnonzero(df[TARGET].to_numpy() == cls)
            k = int(round(len(idx) * a.test_frac))
            test_idx.append(rng.permutation(idx)[:k])
        test_idx = np.sort(np.concatenate(test_idx))
    mask = np.zeros(n, bool)
    mask[test_idx] = True
    splits_dir = a.dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    test, train = df.loc[mask], df.loc[~mask]
    train.to_csv(splits_dir / "train.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)

    def bal(d):
        vc = d[TARGET].value_counts().to_dict()
        return {str(k): f"{vc.get(k, 0)} ({100*vc.get(k, 0)/len(d):.1f}%)" for k in (0, 1)}

    (splits_dir / "split_info.json").write_text(json.dumps({
        "source": "credit-approval/credit-approval.csv",
        "target": TARGET, "target_index": len(feats),
        "n_rows": n, "n_cols": len(df.columns), "seed": a.seed,
        "test_frac": a.test_frac,
        "protocol": ("small dataset: drop ill-formed and duplicate rows, then a single "
                     f"{'random' if a.no_stratify else 'stratified random'} "
                     f"{int(100*(1-a.test_frac))}/{int(100*a.test_frac)} train/test split. "
                     "No validation split, no context subsampling, no context selection."),
        "rows_dropped": {"source_rows": n0, "ill_formed_any_missing": n_ill,
                         "exact_duplicates": n_dup, "contradictory_feature_rows": n_contra,
                         "rows_kept": n},
        "splits": {"train.csv": {"n": len(train), "balance": bal(train)},
                   "test.csv": {"n": len(test), "balance": bal(test)}},
    }, indent=2))
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    print(f"  rows {n0} -> {n}   (ill-formed {n_ill}, exact dupes {n_dup}, contradictory {n_contra})")
    print(f"  target {TARGET}: " + ", ".join(f"{k}->{v}" for k, v in TARGET_MAP.items())
          + f"   balance {bal(df)}")
    print(f"  features {len(feats)}: " + ", ".join(f"{t}={sum(1 for x in types.values() if x == t)}"
                                                   for t in ("cat", "int", "real")))
    print(f"  train.csv {len(train)}  {bal(train)}")
    print(f"  test.csv  {len(test)}  {bal(test)}")
    return df


if __name__ == "__main__":
    main()

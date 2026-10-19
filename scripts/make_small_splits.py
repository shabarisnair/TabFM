#!/usr/bin/env python
"""Fixed-size stratified train / test split for a table too small for the context protocol.

The larger binary datasets go through make_splits.py (75/25 split, val_2000, test_attack_1000,
context subsamples) and then select_contexts.py. A table of under 2000 rows has no room for a
2000-row validation set or a 1000-row attack-test set, so, like credit-approval, it gets one
stratified random split and nothing else:

    1. dedup on the full row (features + label), keeping the first; rows with a missing cell
       are refused -- clean the source first
    2. test  = an exact stratified draw of --test rows        -> test.csv
    3. train = an exact stratified draw of --train of the rest -> train.csv
    4. any rows left over are recorded in split_info.json and not used

Draws use prepare_multiclass_splits.stratified_exact (largest-remainder class quotas), and rows
are copied BYTE-FOR-BYTE from the source CSV, as make_splits.py does.

    python scripts/make_small_splits.py --dataset is_this_a_good_customer --train 1400 --test 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prepare_multiclass_splits import balance, stratified_exact  # noqa: E402
from tabfm_experiments.config import DATASETS  # noqa: E402
from tabfm_experiments.data import fingerprint_frame  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--train", type=int, required=True, help="rows in train.csv")
    ap.add_argument("--test", type=int, required=True, help="rows in test.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    info_ds = DATASETS[a.dataset]
    src, target = info_ds.raw_csv, info_ds.target
    out = info_ds.splits_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if target not in df.columns:
        raise SystemExit(f"target {target!r} not in {src}")
    if df.isna().any().any():
        raise SystemExit(f"{src} has missing cells; clean it first")
    if info_ds.metadata_csv.exists():
        feats = list(pd.read_csv(info_ds.metadata_csv)["feature"])
        if feats != [c for c in df.columns if c != target]:
            raise SystemExit("metadata feature list does not match the CSV columns / order")

    fp = fingerprint_frame(df)
    keep = np.flatnonzero(~pd.Series(fp).duplicated(keep="first").to_numpy())
    if a.train + a.test > len(keep):
        raise SystemExit(f"asked for {a.train} + {a.test} rows from {len(keep)} unique rows")
    y = df[target].to_numpy().astype(int)
    rng = np.random.default_rng(a.seed)

    test = stratified_exact(keep, y, a.test, rng)
    train = stratified_exact(np.setdiff1d(keep, test), y, a.train, rng)
    unused = np.setdiff1d(keep, np.concatenate([train, test]))
    overlap = len(set(fp[train]) & set(fp[test]))
    if overlap:
        raise SystemExit(f"train and test share {overlap} rows")

    lines = src.read_text().splitlines(keepends=True)
    header, body = lines[0], lines[1:]
    assert len(body) == len(df), "line count disagrees with the parsed frame"
    info = {"source": f"{a.dataset}/{src.name}", "target": target,
            "target_index": int(list(df.columns).index(target)),
            "n_rows": int(len(df)), "n_rows_after_dedup": int(len(keep)), "n_cols": int(df.shape[1]),
            "seed": a.seed,
            "protocol": (f"small dataset: dedup, then exact stratified draws of {a.test} test and "
                         f"{a.train} train rows (largest-remainder class quotas). No validation "
                         "split, no context subsampling, no context selection."),
            "unused_rows": {"n": int(len(unused)), "balance": balance(y, unused) if len(unused) else {},
                            "row_indices": [int(i) for i in unused]},
            "train_test_overlap": overlap, "splits": {}}
    print(f"=== {a.dataset} ===  {len(df)} rows ({len(keep)} unique), target {target!r}")
    for name, ix in (("train.csv", train), ("test.csv", test)):
        (out / name).write_text(header + "".join(body[i] for i in ix))
        info["splits"][name] = {"n": int(len(ix)), "balance": balance(y, ix)}
        print(f"  {name:<10s} {len(ix):>5d}  {balance(y, ix)}")
    print(f"  unused     {len(unused):>5d}  {info['unused_rows']['balance']}")
    (out / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"  wrote {out}/")
    return info


if __name__ == "__main__":
    main()

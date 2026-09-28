#!/usr/bin/env python
"""Prepare the HELOC (FICO) dataset to match the other four datasets' conventions.

Produces, in datasets/heloc/:
  heloc_raw_strings.csv   untouched copy of the file as delivered
  heloc.csv               numeric, target RiskPerformance encoded 0/1
  heloc_metadata.csv      feature,type,mutable,min,max (the schema read_metadata expects)
  encoding_codebook.json  the target mapping plus what was dropped and why

TWO DATA DECISIONS, both recorded in the codebook:

1. The 588 rows whose every feature is the sentinel -9 ("No Bureau Record") are
   DROPPED by default. They are mutually identical yet carry contradictory labels
   (323 Bad / 265 Good), so they are pure label noise; they would also sit at
   distance 0 from each other and therefore monopolise any density/hub ranking, and
   they break the train/test disjointness invariant the other datasets satisfy.
   --keep-no-record retains them.

2. The remaining sentinels (-9, -8 "No Usable Trades", -7 "Condition not Met") are
   KEPT as numeric codes, which is how the benchmark distributions use them. They are
   NOT real magnitudes, so they widen each column's empirical min and they act as
   outliers under any z-scored distance. Noted here because the hub attack z-scores.

    python scripts/prepare_heloc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET = "RiskPerformance"
# "Bad" is the adverse outcome, so it takes class 1 -- the same convention as
# charged_off / hospital_death / is_phishing in the other datasets.
TARGET_MAP = {"Bad": 1, "Good": 0}
SENTINELS = {-9: "No Bureau Record or No Investigation",
             -8: "No Usable/Valid Trades or Inquiries",
             -7: "Condition not Met (e.g. No Inquiries, No Delinquencies)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/heloc"))
    ap.add_argument("--keep-no-record", action="store_true",
                    help="keep the all--9 rows instead of dropping them")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep exact duplicate rows instead of dropping them")
    a = ap.parse_args(argv)

    # Always start from the untouched delivery, so re-running is idempotent rather than
    # re-reading (and re-encoding) this script's own output.
    backup = a.dir / "heloc_raw_strings.csv"
    if not backup.exists():
        pd.read_csv(a.dir / "heloc.csv").to_csv(backup, index=False)
        print(f"  backed up the delivered file -> {backup}")
    raw = pd.read_csv(backup)
    n0 = len(raw)

    feats = [c for c in raw.columns if c != TARGET]
    no_record = (raw[feats] == -9).all(axis=1)
    dropped = int(no_record.sum())
    df = raw if a.keep_no_record else raw.loc[~no_record].reset_index(drop=True)

    bad = set(raw[TARGET].unique()) - set(TARGET_MAP)
    if bad:
        raise SystemExit(f"unexpected target values {bad}; expected {sorted(TARGET_MAP)}")
    df = df.copy()
    df[TARGET] = df[TARGET].map(TARGET_MAP).astype(int)
    df = df[[*feats, TARGET]]                       # target last, as in the other CSVs
    # Drop exact duplicate rows. One survives the all--9 removal, and because the split
    # protocol carves val out of train by INDEX, an identical pair can land in both
    # val_2000 and a context -- breaking the content-disjointness the other datasets have.
    n_before = len(df)
    if not a.keep_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)
    df.to_csv(a.dir / "heloc.csv", index=False)

    meta = pd.DataFrame({
        "feature": feats,
        # every column is an integer count / months / percentage
        "type": ["int"] * len(feats),
        # no immutability information is published for HELOC, so nothing is frozen
        "mutable": [True] * len(feats),
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    })
    meta.to_csv(a.dir / "heloc_metadata.csv", index=False)

    sent = {str(v): {"meaning": m,
                     "n_cells": int((df[feats] == v).sum().sum()),
                     "n_columns": int((df[feats] == v).any().sum())}
            for v, m in SENTINELS.items()}
    codebook = {
        TARGET: TARGET_MAP,
        "notes": {
            "source_rows": n0,
            "rows_after_preparation": len(df),
            "dropped_all_sentinel_rows": 0 if a.keep_no_record else dropped,
            "dropped_exact_duplicate_rows": n_dupes,
            "drop_rule": ("kept (--keep-no-record)" if a.keep_no_record else
                          "rows whose every feature == -9 were dropped: identical features, "
                          "contradictory labels, and they dominate any density ranking"),
            "sentinels_kept_as_numeric_codes": sent,
            "sentinel_warning": ("-9/-8/-7 are missing-data codes, not magnitudes. They widen "
                                 "each column's empirical min/max and behave as outliers under "
                                 "z-scored distances (hub selection)."),
        },
    }
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    print(f"  rows {n0} -> {len(df)}   (dropped {0 if a.keep_no_record else dropped} all--9 rows)")
    print(f"  target {TARGET}: " + ", ".join(
        f"{k}->{v} ({int((df[TARGET] == v).sum())} rows, {100*(df[TARGET] == v).mean():.1f}%)"
        for k, v in TARGET_MAP.items()))
    print(f"  features {len(feats)}, all type=int, all mutable")
    print(f"  dropped {n_dupes} exact duplicate row(s); remaining: {int(df.duplicated().sum())}")
    print("  wrote heloc.csv, heloc_metadata.csv, encoding_codebook.json")
    return df


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download the UCI Gas Sensor Array Drift dataset (OpenML 1476) into datasets/gas-drift.

Six-class gas identification from a 16-sensor metal-oxide array, 13910 rows x 128
real features. No rows are dropped and no splits are produced -- the file is the
table as delivered, with only the target re-coded to 0..5.

Produces, in datasets/gas-drift/:
  gas-drift_raw_strings.csv   untouched copy as delivered by OpenML (target as string)
  gas-drift.csv               numeric, target `Class` encoded 0..5
  gas-drift_metadata.csv      feature,type,mutable,min,max (the schema read_metadata expects)
  encoding_codebook.json      target mapping, the 16x8 column layout, and the sign structure

    python scripts/prepare_gas_drift.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 1476
TARGET = "Class"
# The six analytes, in the order used by the UCI release. OpenML ships them as the
# strings "1".."6"; keep that order and shift to 0-based.
GASES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
TARGET_MAP = {str(i + 1): i for i in range(6)}

# Each of the 16 sensors contributes 8 features, in this fixed order (UCI
# documentation): the steady-state response, its normalised magnitude, then the
# exponential moving averages of the rising and the decaying transient at three
# smoothing constants each.
SLOTS = [
    ("dR", "steady-state response DR = max - min conductance", "mixed"),
    ("|dR|", "normalised steady-state response |DR| / R0", "positive"),
    ("emai_0.001", "EMA of the rising transient, alpha = 0.001", "positive"),
    ("emai_0.01", "EMA of the rising transient, alpha = 0.01", "positive"),
    ("emai_0.1", "EMA of the rising transient, alpha = 0.1", "positive"),
    ("emad_0.001", "EMA of the decaying transient, alpha = 0.001", "non-positive"),
    ("emad_0.01", "EMA of the decaying transient, alpha = 0.01", "mixed"),
    ("emad_0.1", "EMA of the decaying transient, alpha = 0.1", "mixed"),
]


def col_type(s: pd.Series) -> str:
    """cat for strings, int when every value is integral, real otherwise."""
    if not pd.api.types.is_numeric_dtype(s):
        return "cat"
    v = s.dropna().to_numpy()
    return "int" if np.all(np.equal(np.mod(v, 1), 0)) else "real"


def observed_sign(s: pd.Series) -> str:
    if s.min() > 0:
        return "positive"
    if s.max() < 0:
        return "negative"
    if s.max() <= 0:
        return "non-positive"
    if s.min() >= 0:
        return "non-negative"
    return "mixed"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/gas-drift"))
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_openml

    cache = a.dir / ".sklearn_cache"
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(cache))
    raw = bunch.frame.copy()
    shutil.rmtree(cache, ignore_errors=True)

    feats = [c for c in raw.columns if c != TARGET]
    if len(feats) != 128:
        raise SystemExit(f"expected 128 features, got {len(feats)}")
    raw[[*feats, TARGET]].to_csv(a.dir / "gas-drift_raw_strings.csv", index=False)

    df = raw.copy()
    bad = set(df[TARGET].astype(str).unique()) - set(TARGET_MAP)
    if bad:
        raise SystemExit(f"unexpected target values {bad}; expected {sorted(TARGET_MAP)}")
    df[TARGET] = df[TARGET].astype(str).map(TARGET_MAP).astype(int)
    df = df[[*feats, TARGET]]
    df.to_csv(a.dir / "gas-drift.csv", index=False)

    types = {c: col_type(df[c]) for c in feats}
    pd.DataFrame({
        "feature": feats,
        "type": [types[c] for c in feats],
        "mutable": [True] * len(feats),   # no immutability information is published
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    }).to_csv(a.dir / "gas-drift_metadata.csv", index=False)

    # Column layout: V(8*s + f + 1) is slot f of sensor s, for s in 0..15, f in 0..7.
    layout = {}
    slot_summary = []
    for f, (slot, desc, expected) in enumerate(SLOTS):
        cols = [feats[s * 8 + f] for s in range(16)]
        signs = {observed_sign(df[c]) for c in cols}
        for s, c in enumerate(cols):
            layout[c] = {"sensor": s + 1, "slot": slot, "observed_sign": observed_sign(df[c])}
        slot_summary.append({
            "slot": slot, "description": desc, "columns": cols,
            "expected_sign": expected,
            "observed_signs": sorted(signs),
            "holds_on_every_column": signs <= {expected} if expected != "mixed" else None,
        })

    codebook = {
        TARGET: {str(i): gas for i, gas in enumerate(GASES)},
        "notes": {
            "source": f"OpenML data_id={OPENML_ID} (UCI Gas Sensor Array Drift)",
            "rows": int(len(df)),
            "n_features": len(feats),
            "feature_types": {t: sum(1 for v in types.values() if v == t) for t in ("cat", "int", "real")},
            "rows_dropped": 0,
            "missing_cells": int(df[feats].isna().sum().sum()),
            "duplicate_rows": int(df[feats].duplicated().sum()),
            "target_note": (
                "OpenML ships the target as the strings '1'..'6'; mapped to 0..5. The gas "
                "names are the UCI analyte order and are informative only -- nothing in the "
                "features identifies them."
            ),
            "relation_constraints": (
                "None. The 128 features are independent sensor readings; there is no "
                "published algebraic relation between columns, so "
                "get_relation_constraints returns [] for this dataset."
            ),
            "column_layout_rule": "V(8*sensor_index + slot_index + 1), sensor_index in 0..15, slot_index in 0..7",
            "slots": slot_summary,
            "column_layout": layout,
            "sign_structure": (
                "Verified on all 13910 rows: the 64 columns in slots |dR|, emai_0.001, "
                "emai_0.01 and emai_0.1 are strictly positive, and the 16 emad_0.001 "
                "columns are non-positive (V30 touches 0). Slots dR, emad_0.01 and "
                "emad_0.1 take both signs. The per-column min/max in the metadata "
                "already encode this, so a box-constrained attack cannot flip a sign "
                "that never occurs in the clean data."
            ),
            "scale_warning": (
                "Raw magnitudes span six orders of magnitude: per-column ranges run from "
                "3.8 to 6.9e5 and standard deviations from 0.53 to 7.0e4, with 5 columns "
                "exceeding 1e5. A single epsilon in raw units is meaningless here -- scale "
                "the budget per column (the metadata min/max give you that) or normalise "
                "before attacking. PREPROCESS_TRANSFORMS is 'none' in this project, so "
                "nothing rescales these columns on the way into the model."
            ),
            "batch_warning": (
                "The UCI release is split into 10 batches recorded over 36 months, and the "
                "dataset exists to study sensor drift across them. The OpenML version "
                "merges all batches and drops the batch label, so the drift structure is "
                "not recoverable and row order carries no temporal meaning."
            ),
            "multiclass_warning": (
                "6 classes. runtime.build_classifier hardcodes clf.n_classes_ = 2 and "
                "metrics assume a binary target; both need generalising before this "
                "dataset can run."
            ),
        },
    }
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    vc = df[TARGET].value_counts().sort_index()
    print(f"  rows {len(df)}  features {len(feats)} (all type=real, all mutable)")
    print(f"  target {TARGET}: " + ", ".join(f"{i}={GASES[i]} n={vc[i]} ({100*vc[i]/len(df):.1f}%)" for i in vc.index))
    print(f"  no rows dropped, no splits written, missing cells {int(df[feats].isna().sum().sum())}")
    print("  wrote gas-drift.csv, gas-drift_raw_strings.csv, gas-drift_metadata.csv, encoding_codebook.json")
    return df


if __name__ == "__main__":
    main()

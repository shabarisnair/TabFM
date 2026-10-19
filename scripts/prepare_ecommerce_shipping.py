#!/usr/bin/env python
"""Download the E-Commerce Shipping dataset (OpenML 46924, TabArena) into datasets/.

Binary "did the order arrive late" prediction, 10999 rows x 10 features (6 numeric,
4 categorical, none above 5 levels). No rows are dropped and no splits are produced --
the file is the table as delivered, with the categoricals and the target re-coded.

TabArena calls this ``E-CommereShippingData`` (the typo is theirs). It is the 5th
hardest of TabArena's 38 classification tasks: the best of 73 official entrants
reaches only 0.7443 ROC AUC, and the median entrant 0.7414.

Produces, in datasets/ecommerce_shipping/:
  ecommerce_shipping_raw_strings.csv   untouched copy as delivered by OpenML
  ecommerce_shipping.csv               numeric, target `ArrivedLate` encoded 0/1
  ecommerce_shipping_metadata.csv      feature,type,mutable,min,max
  encoding_codebook.json               categorical mappings, target mapping, constraints

    python scripts/prepare_ecommerce_shipping.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 46924
TARGET = "ArrivedLate"
# Arriving late is the event of interest, matching the "class 1 is the event"
# convention of the other datasets -- here that happens to be the majority class.
TARGET_MAP = {"No": 0, "Yes": 1}


def col_type(s: pd.Series) -> str:
    """cat for strings, int when every value is integral, real otherwise."""
    if not pd.api.types.is_numeric_dtype(s):
        return "cat"
    v = s.dropna().to_numpy()
    return "int" if np.all(np.equal(np.mod(v, 1), 0)) else "real"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/ecommerce_shipping"))
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_openml

    cache = a.dir / ".sklearn_cache"
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(cache))
    raw = bunch.frame.copy()
    shutil.rmtree(cache, ignore_errors=True)

    feats = [c for c in raw.columns if c != TARGET]
    raw[[*feats, TARGET]].astype(str).to_csv(a.dir / "ecommerce_shipping_raw_strings.csv", index=False)

    df = raw.copy()
    types = {c: col_type(df[c]) for c in feats}
    codebook: dict = {}
    for c in feats:
        if types[c] == "cat":
            levels = sorted(df[c].astype(str).unique())
            mapping = {lv: i for i, lv in enumerate(levels)}
            codebook[c] = mapping
            df[c] = df[c].astype(str).map(mapping).astype(int)
    bad = set(df[TARGET].astype(str).unique()) - set(TARGET_MAP)
    if bad:
        raise SystemExit(f"unexpected target values {bad}; expected {sorted(TARGET_MAP)}")
    codebook[TARGET] = dict(TARGET_MAP)
    df[TARGET] = df[TARGET].astype(str).map(TARGET_MAP).astype(int)
    df = df[[*feats, TARGET]]
    df.to_csv(a.dir / "ecommerce_shipping.csv", index=False)

    pd.DataFrame({
        "feature": feats,
        "type": [types[c] for c in feats],
        "mutable": [True] * len(feats),   # no immutability information is published
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    }).to_csv(a.dir / "ecommerce_shipping_metadata.csv", index=False)

    # Plausible algebraic relations; re-checked rather than assumed.
    checks = [
        ("Discount_offered <= Cost_of_the_Product",
         int((~(df.Discount_offered <= df.Cost_of_the_Product)).sum())),
        ("Customer_care_calls >= Prior_purchases",
         int((~(df.Customer_care_calls >= df.Prior_purchases)).sum())),
    ]
    # A relation is only worth enforcing if the box bounds do not already imply it.
    vacuous = float(df.Discount_offered.max()) < float(df.Cost_of_the_Product.min())
    vc = df[TARGET].value_counts()
    codebook["notes"] = {
        "source": f"OpenML data_id={OPENML_ID} (TabArena-v0.1; Kaggle E-Commerce Shipping Data)",
        "tabarena_name": "E-CommereShippingData",
        "rows": int(len(df)),
        "n_features": len(feats),
        "feature_types": {t: sum(1 for v in types.values() if v == t) for t in ("cat", "int", "real")},
        "rows_dropped": 0,
        "missing_cells": int(df[feats].isna().sum().sum()),
        "duplicate_rows": int(df[feats].duplicated().sum()),
        "categorical_cardinality": {c: len(m) for c, m in codebook.items() if c not in (TARGET, "notes")},
        "class_balance": {"0 (No)": f"{vc[0]} ({100*vc[0]/len(df):.1f}%)",
                          "1 (Yes)": f"{vc[1]} ({100*vc[1]/len(df):.1f}%)"},
        "relation_constraints_tested": [{"statement": s, "violations": n} for s, n in checks],
        "relation_constraints": (
            "Only box constraints. Of the two candidates tested, "
            "'Customer_care_calls >= Prior_purchases' is violated by "
            f"{dict(checks)['Customer_care_calls >= Prior_purchases']} rows, and "
            "'Discount_offered <= Cost_of_the_Product' holds on every row but is "
            f"vacuous: max(Discount_offered)={int(df.Discount_offered.max())} is below "
            f"min(Cost_of_the_Product)={int(df.Cost_of_the_Product.min())}"
            + (", so the metadata min/max already imply it" if vacuous else "")
            + " and it constrains nothing an attacker could otherwise do. "
              "get_relation_constraints returns [] for this dataset, so it is a "
              "box-constrained-only table like coil2000 and heloc."
        ),
        "difficulty": (
            "5th hardest of TabArena's 38 classification tasks. Best of 73 official "
            "entrants 0.7443 ROC AUC (TabPFNv2 at its default config), median entrant "
            "0.7414, AutoGluon 1.3 (4h) 0.7376. The 0.0029 AUC gap between best and "
            "median is the tightest in the hard tail, so the ceiling is intrinsic label "
            "noise rather than a tuning gap."
        ),
        "encoding_note": (
            "All 4 categoricals are closed low-cardinality lists (warehouse block, "
            "shipment mode, product importance, gender), so integer codes are lossless. "
            "Codes are assigned by sorting the level strings."
        ),
        "immutability_note": (
            "Every feature is marked mutable because no immutability information is "
            "published. For a stricter threat model, Gender and Weight_in_gms are the "
            "defensible immutable pair -- an attacker editing an order record cannot "
            "change the customer's sex or the physical mass of the parcel."
        ),
    }
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    print(f"  rows {len(df)}  features {len(feats)}: " + ", ".join(
        f"{t}={sum(1 for v in types.values() if v == t)}" for t in ("cat", "int", "real")))
    print(f"  target {TARGET}: 0=No {vc[0]} ({100*vc[0]/len(df):.1f}%), 1=Yes {vc[1]} ({100*vc[1]/len(df):.1f}%)")
    for s, n in checks:
        tag = "HOLDS " if n == 0 else "REJECT"
        note = "  [vacuous: implied by the box bounds]" if n == 0 and vacuous else ""
        print(f"    {tag} {s}  ({n} violations){note}")
    print("  wrote ecommerce_shipping.csv, ecommerce_shipping_raw_strings.csv, "
          "ecommerce_shipping_metadata.csv, encoding_codebook.json")
    return df


if __name__ == "__main__":
    main()

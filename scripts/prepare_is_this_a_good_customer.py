#!/usr/bin/env python
"""Download the "Is this a good customer" dataset (OpenML 46938, TabArena) into datasets/.

Binary bad-client prediction for consumer loans, 1723 rows x 13 features (5 numeric,
8 categorical, none above 22 levels). No rows are dropped and no splits are produced --
the file is the table as delivered, with the categoricals and the target re-coded.

It is the 6th hardest of TabArena's 38 classification tasks and the hardest
heavily-imbalanced one that is still cleanly encodable: the best of 73 official
entrants reaches 0.7520 ROC AUC on an 88.6 / 11.4 split.

Produces, in datasets/is_this_a_good_customer/:
  is_this_a_good_customer_raw_strings.csv   untouched copy as delivered by OpenML
  is_this_a_good_customer.csv               numeric, target `bad_client_target` 0/1
  is_this_a_good_customer_metadata.csv      feature,type,mutable,min,max
  encoding_codebook.json                    categorical mappings, target mapping, constraints

    python scripts/prepare_is_this_a_good_customer.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 46938
TARGET = "bad_client_target"
# A bad client is the event of interest and the minority class, matching the
# "class 1 is the event" convention of the other datasets.
TARGET_MAP = {"No": 0, "Yes": 1}

# Already-numeric categorical codes: single digits, so sorting the level strings
# reproduces the original integers rather than permuting them.
NUMERIC_CODED = ("having_children_flg", "region", "phone_operator", "is_client")


def col_type(s: pd.Series) -> str:
    """cat for strings, int when every value is integral, real otherwise."""
    if not pd.api.types.is_numeric_dtype(s):
        return "cat"
    v = s.dropna().to_numpy()
    return "int" if np.all(np.equal(np.mod(v, 1), 0)) else "real"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/is_this_a_good_customer"))
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_openml

    cache = a.dir / ".sklearn_cache"
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(cache))
    raw = bunch.frame.copy()
    shutil.rmtree(cache, ignore_errors=True)

    feats = [c for c in raw.columns if c != TARGET]
    raw[[*feats, TARGET]].astype(str).to_csv(
        a.dir / "is_this_a_good_customer_raw_strings.csv", index=False)

    df = raw.copy()
    types = {c: col_type(df[c]) for c in feats}
    codebook: dict = {}
    identity = {}
    for c in feats:
        if types[c] == "cat":
            levels = sorted(df[c].astype(str).unique())
            mapping = {lv: i for i, lv in enumerate(levels)}
            codebook[c] = mapping
            df[c] = df[c].astype(str).map(mapping).astype(int)
            if c in NUMERIC_CODED:
                identity[c] = all(lv == str(i) for lv, i in mapping.items())
    bad = set(df[TARGET].astype(str).unique()) - set(TARGET_MAP)
    if bad:
        raise SystemExit(f"unexpected target values {bad}; expected {sorted(TARGET_MAP)}")
    codebook[TARGET] = dict(TARGET_MAP)
    df[TARGET] = df[TARGET].astype(str).map(TARGET_MAP).astype(int)
    df = df[[*feats, TARGET]]
    df.to_csv(a.dir / "is_this_a_good_customer.csv", index=False)

    pd.DataFrame({
        "feature": feats,
        "type": [types[c] for c in feats],
        "mutable": [True] * len(feats),   # no immutability information is published
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    }).to_csv(a.dir / "is_this_a_good_customer_metadata.csv", index=False)

    # Plausible algebraic relations between the money columns; re-checked, not assumed.
    checks = [
        ("credit_amount <= income", int((~(df.credit_amount <= df.income)).sum())),
        ("credit_amount <= income * credit_term",
         int((~(df.credit_amount <= df.income * df.credit_term)).sum())),
        ("age >= 18", int((~(df.age >= 18)).sum())),
    ]
    enforceable = [s for s, n in checks if n == 0]
    vc = df[TARGET].value_counts()
    codebook["notes"] = {
        "source": f"OpenML data_id={OPENML_ID} (TabArena-v0.1; Kaggle 'Is this a good customer?')",
        "tabarena_name": "Is-this-a-good-customer",
        "rows": int(len(df)),
        "n_features": len(feats),
        "feature_types": {t: sum(1 for v in types.values() if v == t) for t in ("cat", "int", "real")},
        "rows_dropped": 0,
        "missing_cells": int(df[feats].isna().sum().sum()),
        "duplicate_rows": int(df[feats].duplicated().sum()),
        "categorical_cardinality": {c: len(m) for c, m in codebook.items() if c not in (TARGET, "notes")},
        "class_balance": {"0 (good)": f"{vc[0]} ({100*vc[0]/len(df):.1f}%)",
                          "1 (bad)": f"{vc[1]} ({100*vc[1]/len(df):.1f}%)"},
        "relation_constraints_tested": [{"statement": s, "violations": n} for s, n in checks],
        "relation_constraints": (
            "Only box constraints. "
            + (f"{enforceable} hold but are already implied by the metadata min/max, "
               if enforceable else "")
            + "and no cross-column relation survives, so get_relation_constraints "
              "returns [] for this dataset."
        ),
        "difficulty": (
            "6th hardest of TabArena's 38 classification tasks. Best of 73 official "
            "entrants 0.7520 ROC AUC (Portfolio-N20 ensemble, 4h), median entrant "
            "0.7406, AutoGluon 1.3 (4h) 0.7447. The dummy baseline is 0.5 AUC, so the "
            "best entrant removes only 50.4% of its error."
        ),
        "imbalance_warning": (
            "196 positives in 1723 rows (11.4%). At this size a stratified 1000-row "
            "attack-test split is impossible, and any split leaves well under 100 "
            "positives, so per-class metrics will be noisy across trial seeds."
        ),
        "encoding_note": (
            "All 8 categoricals are closed low-cardinality lists; the largest is "
            "product_type with 22 levels. Codes are assigned by sorting the level "
            "strings. For the four columns that already ship as numeric strings "
            f"({', '.join(NUMERIC_CODED)}) that sort is the identity: {identity}."
        ),
        "immutability_note": (
            "Every feature is marked mutable because no immutability information is "
            "published. For a stricter threat model, sex, age and having_children_flg "
            "are the defensible immutable set."
        ),
    }
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    print(f"  rows {len(df)}  features {len(feats)}: " + ", ".join(
        f"{t}={sum(1 for v in types.values() if v == t)}" for t in ("cat", "int", "real")))
    print(f"  target {TARGET}: 0=No {vc[0]} ({100*vc[0]/len(df):.1f}%), 1=Yes {vc[1]} ({100*vc[1]/len(df):.1f}%)")
    for s, n in checks:
        print(f"    {'HOLDS ' if n == 0 else 'REJECT'} {s}  ({n} violations)")
    print(f"  numeric-coded identity check: {identity}")
    print("  wrote is_this_a_good_customer.csv, is_this_a_good_customer_raw_strings.csv, "
          "is_this_a_good_customer_metadata.csv, encoding_codebook.json")
    return df


if __name__ == "__main__":
    main()

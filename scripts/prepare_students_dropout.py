#!/usr/bin/env python
"""Download the UCI Students' Dropout dataset (OpenML 46960, TabArena) into datasets/.

Three-class academic-outcome prediction, 4424 rows x 36 features (19 numeric,
17 categorical). No rows are dropped and no splits are produced -- the file is the
table as delivered, with the categoricals and the target re-coded to integers.

Unlike coil2000 and heloc, this dataset *does* carry relation constraints between
the per-semester curricular-unit counts. Every candidate relation is re-checked
against all 4424 rows here, and only the ones that hold exactly are written out as
enforceable; the rest are recorded as documented-only.

Produces, in datasets/students_dropout/:
  students_dropout_raw_strings.csv   untouched copy as delivered by OpenML
  students_dropout.csv               numeric, target `AcademicOutcome` encoded 0..2
  students_dropout_metadata.csv      feature,type,mutable,min,max
  encoding_codebook.json             categorical mappings, target mapping, constraints

    python scripts/prepare_students_dropout.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 46960
TARGET = "AcademicOutcome"
# Ordered by how far the student got, which happens to also be alphabetical.
TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

MACRO = ["Unemployment_rate", "Inflation_rate", "GDP"]


def col_type(s: pd.Series) -> str:
    """cat for strings, int when every value is integral, real otherwise."""
    if not pd.api.types.is_numeric_dtype(s):
        return "cat"
    v = s.dropna().to_numpy()
    return "int" if np.all(np.equal(np.mod(v, 1), 0)) else "real"


def candidate_relations(df: pd.DataFrame) -> list[dict]:
    """Every per-semester relation worth testing, as (name, lhs, rhs, mask)."""
    out = []
    for sem in ("1st", "2nd"):
        p = f"Curricular_units_{sem}_sem_"
        for lhs, rhs in [
            (p + "approved", p + "enrolled"),
            (p + "credited", p + "enrolled"),
            (p + "without_evaluations", p + "enrolled"),
            (p + "approved", p + "evaluations"),
        ]:
            out.append({
                "lhs": lhs, "rhs": rhs, "op": "<=",
                "violations": int((~(df[lhs] <= df[rhs])).sum()),
            })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("datasets/students_dropout"))
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import fetch_openml

    cache = a.dir / ".sklearn_cache"
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(cache))
    raw = bunch.frame.copy()
    shutil.rmtree(cache, ignore_errors=True)

    feats = [c for c in raw.columns if c != TARGET]
    raw[[*feats, TARGET]].astype(str).to_csv(a.dir / "students_dropout_raw_strings.csv", index=False)

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
    df.to_csv(a.dir / "students_dropout.csv", index=False)

    pd.DataFrame({
        "feature": feats,
        "type": [types[c] for c in feats],
        "mutable": [True] * len(feats),   # no immutability information is published
        "min": [float(df[c].min()) for c in feats],
        "max": [float(df[c].max()) for c in feats],
    }).to_csv(a.dir / "students_dropout_metadata.csv", index=False)

    rels = candidate_relations(df)
    enforceable = [r for r in rels if r["violations"] == 0]
    rejected = [r for r in rels if r["violations"] > 0]

    # grade is zero exactly when no unit was approved -- a biconditional, so it is
    # recorded but not handed to the linear relation executor.
    biconditionals = []
    for sem in ("1st", "2nd"):
        p = f"Curricular_units_{sem}_sem_"
        g, ap_ = df[p + "grade"], df[p + "approved"]
        biconditionals.append({
            "statement": f"{p}grade == 0  <=>  {p}approved == 0",
            "violations": int((g.eq(0) != ap_.eq(0)).sum()),
        })

    macro_combos = df[MACRO].drop_duplicates().sort_values("GDP")
    nat, intl = raw["Nationality"].astype(str), raw["International"].astype(str)

    codebook["notes"] = {
        "source": f"OpenML data_id={OPENML_ID} (TabArena-v0.1; UCI Predict Students' Dropout and Academic Success)",
        "rows": int(len(df)),
        "n_features": len(feats),
        "feature_types": {t: sum(1 for v in types.values() if v == t) for t in ("cat", "int", "real")},
        "rows_dropped": 0,
        "missing_cells": int(df[feats].isna().sum().sum()),
        "duplicate_rows": int(df[feats].duplicated().sum()),
        "categorical_cardinality": {c: len(m) for c, m in codebook.items() if c not in (TARGET, "notes")},
        "relation_constraints_enforceable": [
            {"statement": f"{r['lhs']} <= {r['rhs']}", "lhs": r["lhs"], "rhs": r["rhs"], "op": "<="}
            for r in enforceable
        ],
        "relation_constraints_rejected": [
            {"statement": f"{r['lhs']} <= {r['rhs']}", "violations": r["violations"],
             "reason": "does not hold on the clean table, so it cannot be used as a repair target"}
            for r in rejected
        ],
        "relation_constraints_documented_only": biconditionals + [
            {"statement": f"({', '.join(MACRO)}) must be one of {len(macro_combos)} observed triples",
             "detail": "one macroeconomic triple per enrolment-year cohort; a joint categorical "
                       "constraint, not expressible as a linear relation",
             "valid_triples": macro_combos.to_dict("records")},
            {"statement": "Nationality == 'Portuguese'  <=>  International == 'no'",
             "violations": int(((nat == "Portuguese") != (intl == "no")).sum())},
        ],
        "constraints_note": (
            "The enforceable list is wired into "
            "scripts/tabfm_experiments/constraints.get_relation_constraints, which "
            "returns these as name-based Feature(...) <= Feature(...) constraints (the "
            "lcld style). Note the asymmetry: without_evaluations <= enrolled holds for "
            "the 1st semester but 3 rows violate it for the 2nd, so only the 1st-semester "
            "version is enforceable."
        ),
        "immutability_note": (
            "Every feature is marked mutable because no immutability information is "
            "published. If you want a stricter threat model, the defensible immutable set "
            "is Gender, Nationality, International, Age_at_enrollment, the four parental "
            "columns, and the three macroeconomic columns -- an attacker editing a "
            "student record cannot change a birth year or the national GDP."
        ),
        "encoding_note": (
            "All 17 categoricals are closed code lists (occupation and qualification "
            "catalogues, course names, yes/no flags), not free text, so integer codes are "
            "lossless. Codes are assigned by sorting the level strings, so they carry no "
            "ordinal meaning even where the underlying catalogue does."
        ),
        "size_warning": (
            "4424 rows total. This cannot support the 5000-context + 2000-validation + "
            "1000-attack-test protocol used for the large datasets; the split protocol "
            "still has to be chosen."
        ),
        "multiclass_warning": (
            "3 classes (Dropout 32.1%, Enrolled 18.0%, Graduate 49.9%). "
            "runtime.build_classifier hardcodes clf.n_classes_ = 2 and metrics assume a "
            "binary target; both need generalising before this dataset can run."
        ),
    }
    (a.dir / "encoding_codebook.json").write_text(json.dumps(codebook, indent=2))

    vc = df[TARGET].value_counts().sort_index()
    inv = {v: k for k, v in TARGET_MAP.items()}
    print(f"  rows {len(df)}  features {len(feats)}: " + ", ".join(
        f"{t}={sum(1 for v in types.values() if v == t)}" for t in ("cat", "int", "real")))
    print(f"  target {TARGET}: " + ", ".join(f"{i}={inv[i]} n={vc[i]} ({100*vc[i]/len(df):.1f}%)" for i in vc.index))
    print(f"  relation constraints: {len(enforceable)} enforceable, {len(rejected)} rejected")
    for r in enforceable:
        print(f"    HOLDS  {r['lhs']} <= {r['rhs']}")
    for r in rejected:
        print(f"    REJECT {r['lhs']} <= {r['rhs']}  ({r['violations']} violations)")
    print("  wrote students_dropout.csv, students_dropout_raw_strings.csv, "
          "students_dropout_metadata.csv, encoding_codebook.json")
    return df


if __name__ == "__main__":
    main()

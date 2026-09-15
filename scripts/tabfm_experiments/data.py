"""CSV loading, target discovery, metadata and row fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, DATASETS_DIR


def find_target(csv: Path, given: str | None) -> str:
    """``given``, else ``target`` from the nearest ``split_info.json`` at or above ``csv``."""
    if given:
        return given
    for d in list(Path(csv).resolve().parents)[:4]:
        info = d / "split_info.json"
        if info.exists():
            return json.loads(info.read_text())["target"]
    name = infer_dataset_name(csv)
    if name in DATASETS:
        return DATASETS[name].target
    raise SystemExit(f"--target not given and no split_info.json found above {csv}")


def infer_dataset_name(csv: Path) -> str | None:
    """Dataset directory name for a CSV under ``datasets/<name>/...`` (None if unknown)."""
    p = Path(csv).resolve()
    for d in p.parents:
        if d.parent == DATASETS_DIR.resolve() or (d / f"{d.name}_metadata.csv").exists():
            return d.name
    return None


def drop_columns_for(name: str | None) -> tuple[str, ...]:
    return DATASETS[name].drop_columns if name in DATASETS else ()


def load_split(csv: Path, target: str | None = None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load ``(X, y, feature_columns)``. Dataset-specific raw-only columns are dropped."""
    csv = Path(csv)
    target = find_target(csv, target)
    df = pd.read_csv(csv)
    if target not in df.columns:
        raise SystemExit(f"target '{target}' not in {csv}")
    drop = [c for c in drop_columns_for(infer_dataset_name(csv)) if c in df.columns]
    X = df.drop(columns=[target, *drop])
    return X, df[target].to_numpy(), list(X.columns)


def parse_mutable(v) -> bool:
    return str(v).strip() in {"true", "1", "True", "TRUE", "1.0"}


def read_metadata(path: Path, feature_columns: list[str], *, strict_order: bool = True) -> pd.DataFrame:
    """Metadata rows for ``feature_columns``, in that order, with ``mutable`` as bool.

    Raises if a feature is missing from the metadata, or (``strict_order``) if the
    CSV column order differs from the metadata order -- index-based relation
    constraints (url, wids) would silently point at the wrong columns otherwise.
    """
    m = pd.read_csv(path)
    m["feature"] = m["feature"].astype(str)
    missing = [c for c in feature_columns if c not in set(m["feature"])]
    if missing:
        raise SystemExit(f"features missing from {path}: {missing[:5]}")
    if strict_order:
        meta_order = [f for f in m["feature"] if f in set(feature_columns)]
        if meta_order != list(feature_columns):
            raise SystemExit(f"feature column order differs from {path}; Feature(i) constraints would misalign")
    m = m.set_index("feature").loc[feature_columns].reset_index()
    m["mutable"] = m["mutable"].map(parse_mutable).astype(bool)
    m["type"] = m["type"].astype(str).str.strip().str.lower()
    m["min"] = m["min"].astype(float)
    m["max"] = m["max"].astype(float)
    return m[["feature", "type", "mutable", "min", "max"]]


def metadata_path_for(csv: Path, given: Path | None = None) -> Path:
    if given is not None:
        return Path(given)
    name = infer_dataset_name(csv)
    if name is None:
        raise SystemExit(f"cannot infer dataset for {csv}; pass --metadata")
    return DATASETS_DIR / name / f"{name}_metadata.csv"


def _canon_cell(v) -> str:
    if v is None:
        return "nan"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if float(v).is_integer():
            return str(int(v))
        return repr(float(v))
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if pd.isna(v):
        return "nan"
    return str(v)


def fingerprint_frame(df: pd.DataFrame) -> np.ndarray:
    """SHA-256 per row of ``repr(tuple(canonical str of each cell))``, in column order.

    Integers and integral floats encode identically, NaN is normalised, the index is
    ignored. Callers must align column order before comparing two frames.
    """
    cols = [df[c].to_numpy(dtype=object) for c in df.columns]
    out = np.empty(len(df), dtype=object)
    for i in range(len(df)):
        key = repr(tuple(_canon_cell(col[i]) for col in cols))
        out[i] = hashlib.sha256(key.encode()).hexdigest()
    return out

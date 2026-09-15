"""Paths, dataset registry, seeds and the fixed TabPFNv2 inference configuration."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
TABULARBENCH_DIR = ROOT / "tabularbench"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    target: str
    # Columns that exist in the raw CSV but must never reach TabPFN.
    drop_columns: tuple[str, ...] = ()
    natural_size: int = 10000

    @property
    def dir(self) -> Path:
        return DATASETS_DIR / self.name

    @property
    def raw_csv(self) -> Path:
        return self.dir / f"{self.name}.csv"

    @property
    def metadata_csv(self) -> Path:
        return self.dir / f"{self.name}_metadata.csv"

    @property
    def splits_dir(self) -> Path:
        return self.dir / "splits"


DATASETS: dict[str, DatasetInfo] = {
    "url_unique": DatasetInfo("url_unique", "is_phishing", natural_size=8000),
    "lcld_v2": DatasetInfo("lcld_v2", "charged_off", drop_columns=("issue_d",), natural_size=10000),
    "wids": DatasetInfo("wids", "hospital_death", natural_size=10000),
    # No relation constraints (get_relation_constraints returns [] for this name).
    "coil2000_insurance_policies": DatasetInfo("coil2000_insurance_policies", "MobileHomePolicy", natural_size=5000),
}


@dataclass(frozen=True)
class Seeds:
    """Independent seed streams (see docs/context_poisoning.md)."""

    model_seed: int = 0
    row_seed: int = 1
    attack_seed: int = 2


# Selected-context protocol
SELECT_N_CANDIDATES = 10
SELECT_BALANCED_CAP = 10000
SELECT_CHILD_SIZES = (5000, 1000)
SELECT_CHILD_SEED_OFFSET = 10_000


def inference_config() -> dict:
    """The fixed TabPFNv2 inference config. Copied from the plan; do not edit."""
    from tabpfn.preprocessing.configs import PreprocessorConfig

    return {
        "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none", categorical_name="numeric")],
        "FINGERPRINT_FEATURE": False,
        "POLYNOMIAL_FEATURES": "no",
        "FEATURE_SHIFT_METHOD": None,
        "CLASS_SHIFT_METHOD": None,
        "OUTLIER_REMOVAL_STD": None,
        "SUBSAMPLE_SAMPLES": None,
    }


def ensure_tabularbench_importable() -> None:
    """Put the local TabularBench checkout on sys.path and patch NumPy 2 aliases.

    TabularBench's ``utils/typing.py`` references ``np.float_``, removed in NumPy 2.
    Re-adding the alias is enough; nothing else in the modules we use depends on
    NumPy 1 behaviour.
    """
    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    try:
        import tabularbench.constraints.constraints  # noqa: F401
    except ImportError:
        # Run from the repo root, the outer ``tabularbench/`` checkout directory is
        # picked up as an empty namespace package. Drop it and use the real package.
        for mod in [m for m in sys.modules if m == "tabularbench" or m.startswith("tabularbench.")]:
            del sys.modules[mod]
        if str(TABULARBENCH_DIR) not in sys.path:
            sys.path.insert(0, str(TABULARBENCH_DIR))
        import tabularbench.constraints.constraints  # noqa: F401

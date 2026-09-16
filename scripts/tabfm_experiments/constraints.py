"""TabularBench constraints, min-max scaling and end-of-attack repair (X_train CAPGD only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import ensure_tabularbench_importable

ensure_tabularbench_importable()

from tabularbench.constraints.constraints import Constraints, get_constraints_from_metadata  # noqa: E402
from tabularbench.constraints.constraints_backend_executor import ConstraintsExecutor  # noqa: E402
from tabularbench.constraints.pytorch_backend import PytorchBackend  # noqa: E402
from tabularbench.constraints.relation_constraint import (  # noqa: E402
    AndConstraint,
    EqualConstraint,
    Feature,
)

KNOWN_TYPES = {"real", "int", "cat"}


# --------------------------------------------------------------------- relations
def get_relation_constraints(dataset_name: str, metadata_df: pd.DataFrame | None = None) -> list:
    """Relation constraints from the TabularBench sample factories.

    url / url_unique and wids use 0-based ``Feature(i)`` indices into X *after*
    dropping the target (metadata order); lcld uses feature names.
    """
    name = dataset_name.lower()
    if name in ("url", "url_unique"):
        from tabularbench.datasets.samples import url

        return url.get_relation_constraints()
    if name in ("lcld", "lcld_v2"):
        from tabularbench.datasets.samples import lcld

        return lcld.get_relation_constraints()
    if name == "wids":
        from tabularbench.datasets.samples import wids

        return wids.get_relation_constraints(metadata_df)
    return []


def build_constraints(dataset_name: str, metadata_df: pd.DataFrame, feature_columns: list[str]) -> Constraints:
    """TabularBench ``Constraints`` for exactly ``feature_columns`` (in that order).

    ``metadata_df`` must already be filtered/ordered to the features (see
    ``data.read_metadata``), with ``mutable`` as bool.
    """
    meta = metadata_df.set_index("feature").loc[list(feature_columns)].reset_index()
    rel = get_relation_constraints(dataset_name, meta)
    cons = get_constraints_from_metadata(meta, rel or None, col_filter=list(feature_columns))
    cons.mutable_features = meta["mutable"].astype(bool).to_numpy()
    cons.feature_types = meta["type"].astype(str).str.lower().to_numpy()
    cons.lower_bounds = meta["min"].astype(float).to_numpy()
    cons.upper_bounds = meta["max"].astype(float).to_numpy()
    cons.feature_names = list(feature_columns)
    unknown = set(cons.feature_types) - KNOWN_TYPES
    if unknown:
        raise ValueError(f"unhandled feature types {sorted(unknown)} (drop them before building constraints)")
    return cons


def _executor(constraint, constraints: Constraints) -> Callable[[torch.Tensor], torch.Tensor]:
    ex = ConstraintsExecutor(constraint, PytorchBackend(), feature_names=constraints.feature_names)

    def run(x_raw: torch.Tensor) -> torch.Tensor:
        v = ex.execute(x_raw).to(x_raw.dtype)
        if v.dim() == 0 or v.shape[0] != x_raw.shape[0]:
            v = v.reshape(-1).expand(x_raw.shape[0])
        return v

    return run


def relation_penalty_fn(constraints: Constraints) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Per-row violation (0 = satisfied) of all relations, summed as ``AndConstraint`` does.

    Matches the term TabularBench subtracts from the CE in ``attack_single_run``.
    Returns ``None`` if the dataset has no relation constraints.
    """
    rel = constraints.relation_constraints
    if not rel:
        return None
    # AndConstraint requires >= 2 operands.
    return _executor(rel[0] if len(rel) == 1 else AndConstraint(rel), constraints)


def single_relation_penalty(constraint, constraints: Constraints) -> Callable[[torch.Tensor], torch.Tensor]:
    return _executor(constraint, constraints)


def has_fixable_equalities(constraints: Constraints) -> bool:
    return any(isinstance(c, EqualConstraint) and isinstance(c.left_operand, Feature)
               for c in (constraints.relation_constraints or []))


# ----------------------------------------------------------------------- scaling
@dataclass
class FeatureSpec:
    """Per-feature min-max scaler plus type / mutability masks (all on one device)."""

    names: list[str]
    lo: torch.Tensor
    rng: torch.Tensor
    mutable: torch.Tensor
    is_int: torch.Tensor
    is_cat: torch.Tensor

    @classmethod
    def from_metadata(cls, meta: pd.DataFrame, *, device, scaler: str = "metadata",
                      X_train: torch.Tensor | None = None) -> "FeatureSpec":
        """``scaler="metadata"`` uses metadata min/max; ``"train"`` uses ``X_train`` min/max.

        Constant features (``max <= min``) get range 1, so they cannot move in scaled units.
        """
        if scaler == "metadata":
            lo = meta["min"].astype(float).to_numpy()
            hi = meta["max"].astype(float).to_numpy()
        elif scaler == "train":
            if X_train is None:
                raise ValueError("scaler='train' needs X_train")
            arr = X_train.detach().double().cpu().numpy()
            lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        else:
            raise ValueError(f"unknown scaler {scaler}")
        rng = np.where(hi - lo <= 0, 1.0, hi - lo)
        typ = meta["type"].astype(str).str.lower().to_numpy()
        t = lambda v, d: torch.as_tensor(np.array(v), dtype=d, device=device)
        return cls(
            names=list(meta["feature"]),
            lo=t(lo, torch.float32), rng=t(rng, torch.float32),
            mutable=t(meta["mutable"].astype(bool).to_numpy(), torch.bool),
            is_int=t(typ == "int", torch.bool), is_cat=t(typ == "cat", torch.bool),
        )

    def to_scaled(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.lo) / self.rng

    def to_raw(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self.rng + self.lo


# ------------------------------------------------------------------------ repair
def fix_types_raw(x_clean, x_adv, is_int, is_cat):
    """Port of ``tabularbench/attacks/utils.py::fix_types`` (raw space).

    ``int`` features truncate the *perturbation* toward zero (``torch.fix``), ``cat``
    features round the *value*. The reference returns early when there are no int
    features, *before* rounding categoricals; we reproduce that. All three datasets
    here have int features, so their categoricals are rounded.
    """
    out = x_adv.clone()
    if not bool(is_int.any()):
        return out
    out = torch.where(is_int, x_clean + torch.fix(x_adv - x_clean), out)
    return torch.where(is_cat, torch.round(x_adv), out)


def fix_immutable_raw(x_clean, x_adv, mutable):
    """Port of ``tabularbench/attacks/utils.py::fix_immutable`` (raw space, no prints)."""
    return torch.where(mutable, x_adv, x_clean)


def fix_equality_raw(constraints: Constraints, x_raw: torch.Tensor) -> torch.Tensor:
    from tabularbench.attacks.utils import fix_equality_constraints

    if not has_fixable_equalities(constraints):
        return x_raw
    return fix_equality_constraints(constraints, x_raw).to(x_raw.dtype)


def repair_end(x_clean: torch.Tensor, x_adv: torch.Tensor, constraints: Constraints,
               *, fix_equality: bool = True) -> torch.Tensor:
    """End-of-attack repair in raw space: ``fix_types -> fix_immutable -> fix_equality``."""
    dev = x_adv.device
    types = np.asarray(constraints.feature_types).astype(str)
    is_int = torch.as_tensor(types == "int", device=dev)
    is_cat = torch.as_tensor(types == "cat", device=dev)
    mutable = torch.as_tensor(np.asarray(constraints.mutable_features).astype(bool), device=dev)
    out = fix_types_raw(x_clean, x_adv, is_int, is_cat)
    out = fix_immutable_raw(x_clean, out, mutable)
    if fix_equality:
        out = fix_equality_raw(constraints, out)
    return out


def repair_end_budgeted(x_clean, x_adv, spec: FeatureSpec, constraints: Constraints, eps: float,
                        *, norm: str = "L2", fix_equality: bool = True, n_iter: int = 6):
    """End repair that also keeps each row within the scaled eps-ball, prioritising the budget.

    Runs the normal end repair once (types, immutable, optional equality) so the row is
    schema-valid, then alternates: (a) shrink any over-budget row's scaled perturbation
    radially back onto the eps-ball, (b) re-round types and restore immutable features.
    The loop ends on a type/immutable repair, so integrality and immutability hold exactly
    and the budget holds up to per-feature rounding slack. Equality constraints may then be
    slightly violated -- budget and exact equality cannot both hold with integer features.
    """
    x = repair_end(x_clean, x_adv, constraints, fix_equality=fix_equality)
    cs = spec.to_scaled(x_clean)
    tol = 1e-5
    # Shrink to a target strictly BELOW eps: the re-rounding that follows can only push
    # the norm back up (torch.round on categoricals moves away from clean), so shrinking
    # to exactly eps oscillates forever. Tighten the target each pass until it fits.
    target = eps * 0.98
    for _ in range(n_iter):
        d = spec.to_scaled(x) - cs
        if norm == "L2":
            n = d.pow(2).sum(dim=-1, keepdim=True).sqrt()
            if not bool((n > eps + tol).any()):
                return x
            d = torch.where(n > eps + tol, d * (target / (n + 1e-12)), d)
        else:  # Linf
            if not bool((d.abs() > eps + tol).any()):
                return x
            d = d.clamp(-target, target)
        x = spec.to_raw(cs + d)
        x = fix_types_raw(x_clean, x, spec.is_int, spec.is_cat)
        x = fix_immutable_raw(x_clean, x, spec.mutable)
        target *= 0.9
    # Guarantee the invariant: any row still over budget reverts to its clean values.
    d = spec.to_scaled(x) - cs
    n = d.pow(2).sum(dim=-1, keepdim=True).sqrt() if norm == "L2" else d.abs().amax(-1, keepdim=True)
    return torch.where(n > eps + tol, x_clean, x)


def equality_post_step(spec: FeatureSpec, constraints: Constraints):
    """In-loop ``unscale -> fix_equality_constraints -> rescale`` (None if nothing to fix)."""
    if not has_fixable_equalities(constraints):
        return None
    return lambda x_scaled: spec.to_scaled(fix_equality_raw(constraints, spec.to_raw(x_scaled)))

"""Binary classification metrics used for every clean / poisoned report."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


@dataclass
class BinaryMetrics:
    ce: float
    roc_auc: float
    accuracy: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict:
        return asdict(self)


def binary_metrics(probs: np.ndarray, y: np.ndarray) -> BinaryMetrics:
    """Metrics from class probabilities ``[n, 2]`` and integer labels ``[n]``.

    CE matches ``F.nll_loss(log(probs.clamp_min(1e-12)), y)``.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"expected probs of shape [n, 2], got {p.shape}")
    ce = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
    pred = p.argmax(1)
    try:
        auc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else math.nan
    except ValueError:
        auc = math.nan
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return BinaryMetrics(
        ce=ce,
        roc_auc=auc,
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def deltas(before: BinaryMetrics, after: BinaryMetrics) -> dict[str, float]:
    """``after - before`` for every metric."""
    return {f.name: getattr(after, f.name) - getattr(before, f.name) for f in fields(BinaryMetrics)}

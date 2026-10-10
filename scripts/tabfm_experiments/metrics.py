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


def balanced_accuracy_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    """Mean of the per-class recalls. A class with no support contributes nothing."""
    rec = [t / (t + f) for t, f in ((tp, fn), (tn, fp)) if (t + f) > 0]
    return float(np.mean(rec)) if rec else math.nan


def mcc_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    """Matthews correlation. 0 when a row/column of the confusion matrix is empty."""
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def with_derived(m: "BinaryMetrics | dict") -> dict:
    """Metric dict plus ``balanced_accuracy`` and ``mcc``, both from the stored counts.

    Lets us read the two extra metrics out of results written before they existed.
    """
    d = dict(m.as_dict() if isinstance(m, BinaryMetrics) else m)
    counts = tuple(int(d[k]) for k in ("tn", "fp", "fn", "tp"))
    d["balanced_accuracy"] = balanced_accuracy_from_counts(*counts)
    d["mcc"] = mcc_from_counts(*counts)
    return d


# ------------------------------------------------------------------ multiclass

MULTICLASS_METRICS = ("ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc")


def multiclass_metrics(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """Metrics for a K-class problem, from probabilities ``[n, K]`` and labels ``[n]``.

    ce                 cross-entropy, -mean log p(true class); matches
                       F.nll_loss(log(probs.clamp_min(1e-12)), y), as binary_metrics does
    accuracy           share of rows whose argmax is the true class
    balanced_accuracy  mean per-class recall, over the classes present in ``y``
    mcc                multiclass Matthews correlation (Gorodkin's R_K)
    roc_auc            one-vs-rest AUC of each class, macro-averaged (unweighted)

    At K=2 every one of these equals its binary_metrics / with_derived counterpart --
    one-vs-rest AUC of class 0 equals that of class 1 because p0 = 1 - p1 -- so binary
    results are a special case, which the tests check.

    ``confusion`` (rows = true class, columns = predicted) and the per-class AUCs are
    returned too, so every number can be re-derived independently.
    """
    from sklearn.metrics import matthews_corrcoef

    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    K = int(n_classes)
    if p.ndim != 2 or p.shape[1] != K:
        raise ValueError(f"expected probs of shape [n, {K}], got {p.shape}")
    if len(y) != len(p):
        raise ValueError(f"{len(y)} labels for {len(p)} probability rows")
    if y.min() < 0 or y.max() >= K:
        raise ValueError(f"labels span {y.min()}..{y.max()}, outside 0..{K - 1}")

    pred = p.argmax(1)
    conf = np.zeros((K, K), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    support = conf.sum(1)
    recall = np.diag(conf)[support > 0] / support[support > 0]

    auc_per_class = []
    for c in range(K):
        pos = y == c
        if pos.all() or not pos.any():
            auc_per_class.append(math.nan)          # undefined with one side empty
        else:
            auc_per_class.append(float(roc_auc_score(pos.astype(int), p[:, c])))
    roc_auc = (math.nan if any(math.isnan(a) for a in auc_per_class)
               else float(np.mean(auc_per_class)))

    return {
        "ce": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
        "accuracy": float(np.trace(conf) / len(y)),
        "balanced_accuracy": float(recall.mean()),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": roc_auc,
        "roc_auc_per_class": auc_per_class,
        "confusion": conf.tolist(),
        "n": int(len(y)),
    }

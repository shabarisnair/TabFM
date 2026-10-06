"""Test-free defences: clean a poisoned context using only the context itself.

THREAT MODEL (shared by every method that lands here)
  * The context labels have been corrupted at a KNOWN rate R%.
  * The attacker's row-choice strategy is UNKNOWN (random, hub-based, or anything else).
  * The clean labels are unknown, and the test set is off limits until the very last
    step -- nothing in the detector may look at it, or the result is meaningless.

METHOD: k-fold self-consistency filtering (``kfold``)
  Split the context into ``n_folds`` folds. Treat each fold in turn as a query set and
  the other folds as the context, and read off p(y_i = the label row i CARRIES). A row
  whose own neighbourhood disagrees with its label gets a low score. After one pass every
  row has a score; flag the k = R% lowest as poisoned and drop them.

  The score is deliberately self-referential: it is the probability of the GIVEN label,
  not of the predicted one. A clean row in a coherent neighbourhood scores high; a
  flipped row scores low because the rest of its neighbourhood still carries clean
  labels. That signal weakens as R grows, since more of each neighbourhood is corrupted.

WHY precision == recall HERE
  The budget is exactly k and the attacker flipped exactly k, so |flagged| = |flipped|
  and TP/(TP+FP) = TP/(TP+FN). Both are reported anyway -- they come apart as soon as a
  method picks its own budget -- but do not read them as independent evidence. The
  threshold-free number to compare methods on is ``detection_roc_auc``, which scores the
  ranking itself and does not depend on the budget at all.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .metrics import binary_metrics, with_derived

# Reported for every context variant. Order is the one the tables use.
METRICS = ("ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc", "f1")


# ------------------------------------------------------------------ fold assignment

def make_folds(y_given: np.ndarray, n_folds: int, seed: int) -> tuple[np.ndarray, bool]:
    """Fold id per row. Returns ``(folds, stratified)``.

    Stratification is on the GIVEN (possibly corrupted) labels, because that is all a
    defender can see. It also keeps every held-out context two-class, which matters on
    coil2000 where only ~6% of rows are positive: an all-one-class context would make
    the whole fold's scores meaningless.

    Falls back to unstratified KFold when some class has fewer members than there are
    folds. That fallback is recorded rather than silently applied.
    """
    y = np.asarray(y_given).astype(int)
    n = len(y)
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")
    stratified = int(np.bincount(y, minlength=2).min()) >= n_folds
    splitter = (StratifiedKFold if stratified else KFold)(
        n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(n, dtype=np.int64)
    for f, (_, held_out) in enumerate(splitter.split(np.zeros(n), y)):
        folds[held_out] = f
    return folds, stratified


# ------------------------------------------------------------------ the detector

def kfold_self_probs(clf, X_ctx: torch.Tensor, y_given: np.ndarray, folds: np.ndarray,
                     evaluate_context, *, test_batch_size: int | None = None,
                     log=None) -> np.ndarray:
    """p(row i carries the label it was given), with row i held out of the context.

    One refit per fold. Returns ``[n]`` float64 in [0, 1]. The test set is never touched.
    """
    y_given = np.asarray(y_given).astype(int)
    n = len(y_given)
    if len(folds) != n:
        raise ValueError(f"folds has {len(folds)} entries for {n} rows")
    dev = X_ctx.device
    y_all = torch.as_tensor(y_given, dtype=torch.float32, device=dev)
    out = np.full(n, np.nan, dtype=np.float64)

    for f in np.unique(folds):
        q = np.flatnonzero(folds == f)
        c = np.flatnonzero(folds != f)
        qi = torch.as_tensor(q, dtype=torch.long, device=dev)
        ci = torch.as_tensor(c, dtype=torch.long, device=dev)
        # y_query is only used for the CE that evaluate_context returns, which we drop;
        # the probabilities are what matter.
        _, _, probs = evaluate_context(clf, X_ctx[ci], y_all[ci], X_ctx[qi],
                                       y_all[qi].long(), need_grad=False,
                                       test_batch_size=test_batch_size)
        p = probs.detach().float().cpu().numpy().astype(np.float64)
        out[q] = p[np.arange(len(q)), y_given[q]]
        if log is not None:
            log(f"    fold {int(f):>2d}/{len(np.unique(folds))}: context {len(c)}, "
                f"query {len(q)}, mean p(given) {out[q].mean():.4f}")

    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} rows never appeared in a fold")
    return out


def flag_lowest(scores: np.ndarray, k: int) -> np.ndarray:
    """The k lowest-scoring row indices, sorted. Ties break on index, so it reproduces."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(int(k), len(scores))
    return np.sort(np.argsort(np.asarray(scores), kind="stable")[:k])


# ------------------------------------------------------------------ scoring the removal

def detection_report(flagged: np.ndarray, flipped: np.ndarray, y_clean: np.ndarray,
                     y_given: np.ndarray, scores: np.ndarray) -> dict:
    """How well the flagged set matches the rows the attacker really flipped.

    Two class-conditional views, because "per class" is ambiguous once labels are the
    thing being corrupted:
      * ``recall_by_true_class[c]``  -- of the flipped rows whose CLEAN label was c, what
        share did we catch? Answers "which class's poison hides best".
      * ``precision_by_given_class[c]`` -- of the rows we flagged that CARRY label c, what
        share were really flipped? Answers "which class do we wrongly throw away".
    """
    n = len(y_clean)
    y_clean = np.asarray(y_clean).astype(int)
    y_given = np.asarray(y_given).astype(int)
    is_flipped = np.zeros(n, dtype=bool)
    is_flipped[np.asarray(flipped, dtype=np.int64)] = True
    is_flagged = np.zeros(n, dtype=bool)
    is_flagged[np.asarray(flagged, dtype=np.int64)] = True

    tp = int(np.sum(is_flagged & is_flipped))
    fp = int(np.sum(is_flagged & ~is_flipped))
    fn = int(np.sum(~is_flagged & is_flipped))
    tn = int(n - tp - fp - fn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")

    # Threshold-free quality of the ranking: lower score should mean "more likely
    # flipped", so the score is negated. Undefined when nothing was flipped (R=0).
    try:
        auc = float(roc_auc_score(is_flipped.astype(int), -np.asarray(scores)))
    except ValueError:
        auc = float("nan")

    by_true, by_given = {}, {}
    for c in (0, 1):
        m = is_flipped & (y_clean == c)
        by_true[str(c)] = {
            "n_flipped": int(m.sum()),
            "n_detected": int((m & is_flagged).sum()),
            "recall": float((m & is_flagged).sum() / m.sum()) if m.sum() else float("nan"),
        }
        g = is_flagged & (y_given == c)
        by_given[str(c)] = {
            "n_flagged": int(g.sum()),
            "n_correct": int((g & is_flipped).sum()),
            "precision": float((g & is_flipped).sum() / g.sum()) if g.sum() else float("nan"),
            "n_rows_with_this_label": int((y_given == c).sum()),
        }

    return {
        "n_context": n, "n_flipped_true": int(is_flipped.sum()),
        "n_flagged": int(is_flagged.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": float(2 * prec * rec / (prec + rec)) if tp else 0.0,
        "accuracy": float((tp + tn) / n),
        # Share of the removed rows that were clean -- the collateral damage.
        "false_removal_rate": float(fp / (tp + fp)) if tp + fp else float("nan"),
        "detection_roc_auc": auc,
        "budget_equals_truth": int(is_flagged.sum()) == int(is_flipped.sum()),
        "recall_by_true_class": by_true,
        "precision_by_given_class": by_given,
        "mean_score_flipped": float(scores[is_flipped].mean()) if is_flipped.any() else float("nan"),
        "mean_score_clean": float(scores[~is_flipped].mean()) if (~is_flipped).any() else float("nan"),
    }


# ------------------------------------------------------------------ downstream scoring

def score_on_test(clf, X_ctx: torch.Tensor, y_ctx: torch.Tensor, X_test: torch.Tensor,
                  y_test: np.ndarray, evaluate_context, *,
                  test_batch_size: int | None = None) -> dict:
    """Fit on the given context and report every metric on the real test set."""
    yt = torch.as_tensor(np.asarray(y_test).astype(np.int64), dtype=torch.long,
                         device=X_test.device)
    _, _, probs = evaluate_context(clf, X_ctx, y_ctx.float(), X_test, yt,
                                   need_grad=False, test_batch_size=test_batch_size)
    return with_derived(binary_metrics(probs.detach().float().cpu().numpy(), np.asarray(y_test)))


def keep_mask(n: int, drop: np.ndarray) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    m[np.asarray(drop, dtype=np.int64)] = False
    return m


def delta(after: dict, before: dict) -> dict:
    """``after - before`` over METRICS only (the counts are not differences)."""
    return {m: float(after[m] - before[m]) for m in METRICS}

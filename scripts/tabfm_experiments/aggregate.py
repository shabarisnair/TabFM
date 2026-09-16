"""Summaries over the run x subsample trial grid.

Trials are hierarchical (runs share row subsets, subsets share CAPGD seeds), so we
report the flat mean/std over all trials *and* the grouped views.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _stats(values: list[float]) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def _collect(trials: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        for m, v in t[key].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[m].append(float(v))
    return out


def best_per_run(trials: list[dict], key: str = "delta", metric: str = "ce") -> list[dict]:
    """For each run_id, the trial with the largest ``trial[key][metric]``.

    Runs capture CAPGD randomness; row subsamples capture which rows were drawn. Taking
    the best subsample per run models an attacker who tries several row subsets and keeps
    the strongest. Ties resolve to the lowest subsample_id. Note this is a maximum, so the
    mean over the returned list is biased upward relative to a mean over all trials.
    """
    best: dict[int, dict] = {}
    for t in trials:
        r = int(t["run_id"])
        cur = best.get(r)
        if cur is None or (t[key][metric], -int(t["subsample_id"])) > (
                cur[key][metric], -int(cur["subsample_id"])):
            best[r] = t
    return [best[r] for r in sorted(best)]


def summarize_trials(trials: list[dict], key: str = "delta") -> dict:
    """Aggregate ``trial[key]`` (a flat metric dict) over trials.

    Returns ``overall`` (all trials), ``by_run`` / ``by_subsample`` (per-group stats),
    and ``across_runs`` / ``across_subsamples`` (stats of the group means).
    """
    if not trials:
        return {"n_trials": 0}
    out = {"n_trials": len(trials), "overall": {m: _stats(v) for m, v in _collect(trials, key).items()}}
    for group, col in (("run", "run_id"), ("subsample", "subsample_id")):
        groups: dict[int, list[dict]] = defaultdict(list)
        for t in trials:
            groups[int(t[col])].append(t)
        per = {str(g): {m: _stats(v) for m, v in _collect(ts, key).items()} for g, ts in sorted(groups.items())}
        out[f"by_{group}"] = per
        metrics = sorted({m for d in per.values() for m in d})
        out[f"across_{group}s"] = {m: _stats([per[g][m]["mean"] for g in per if m in per[g]]) for m in metrics}
    return out

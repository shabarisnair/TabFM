"""Transfer of TabPFNv2-optimised context poison to a classical model (XGBoost).

The poison was optimised white-box against TabPFNv2 in-context learning. Here the same
poisoned context is used as an ordinary *training set* for a model that is trained on it
from scratch, and scored on the same ``test_attack_1000.csv``. Two protocols differ only
in where the validation set used for HPO / early stopping comes from:

  A  internal split of the given context (so a poisoned context yields a poisoned
     validation set -- the attacker owns the whole pipeline input)
  B  a separate clean validation file (a defender with a trusted holdout)

Per-trial ``.npz`` deltas from an attack run are enough to rebuild every poisoned
context, so the transfer aggregate can use the *same* per-run winners that the
TabPFNv2 numbers were aggregated over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary_metrics, with_derived

METRICS = ("ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc", "f1")


# --------------------------------------------------------------------------- contexts

def apply_trial_delta(train_df: pd.DataFrame, target: str, npz) -> pd.DataFrame:
    """Rebuild one trial's poisoned context from the clean frame and its saved delta.

    Mirrors the ``context_poisoned.csv`` writer in ``scripts/attack_context.py`` exactly.
    """
    df = train_df.copy()
    if "cell_delta_raw" in npz:                     # x-capgd: additive cell deltas
        cols = [str(c) for c in npz["feature_names"]]
        idx = npz["row_indices"]
        df[cols] = df[cols].astype("float64")
        df.loc[idx, cols] = df.loc[idx, cols].to_numpy() + npz["cell_delta_raw"]
    elif "y_poisoned" in npz:                       # label flip: labels only
        df[target] = npz["y_poisoned"]
    else:
        raise ValueError(f"delta archive has neither cell_delta_raw nor y_poisoned: {list(npz)}")
    return df


def per_run_best_tags(summary: dict) -> list[str]:
    """``run<r>_sub<s>`` of the winning subsample of each run, as the summary recorded it."""
    return [f"run{int(t['run_id'])}_sub{int(t['subsample_id'])}" for t in summary["per_run_best"]]


def load_attack_contexts(attack_dir: Path, train_df: pd.DataFrame, target: str
                         ) -> tuple[list[tuple[str, pd.DataFrame]], dict]:
    """The per-run-best poisoned contexts of an attack run, rebuilt from ``trials/*.npz``."""
    attack_dir = Path(attack_dir)
    summary = json.loads((attack_dir / "summary.json").read_text())
    out = []
    for tag in per_run_best_tags(summary):
        with np.load(attack_dir / "trials" / f"{tag}_delta.npz", allow_pickle=False) as z:
            out.append((tag, apply_trial_delta(train_df, target, z)))
    return out, summary


def tabpfn_run_metrics(summary: dict) -> dict:
    """Clean / poisoned / delta per winning run, with balanced accuracy and MCC added."""
    clean, pois, delta = [], [], []
    for t in summary["per_run_best"]:
        c, p = with_derived(t["clean"]), with_derived(t["poisoned"])
        clean.append(c)
        pois.append(p)
        delta.append({m: p[m] - c[m] for m in METRICS})
    return {"clean": clean, "poisoned": pois, "delta": delta,
            "tags": per_run_best_tags(summary)}


# --------------------------------------------------------------------------- xgboost

@dataclass
class XgbConfig:
    hpo_trials: int = 30
    n_estimators: int = 2000            # upper bound; early stopping picks the real count
    early_stopping_rounds: int = 50
    n_jobs: int = 8
    device: str = "cpu"
    use_hpo: bool = True                # off -> XGBoost defaults, no optuna study


def _suggest(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 1e2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def _make(params: dict, cfg: XgbConfig, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=cfg.n_estimators, early_stopping_rounds=cfg.early_stopping_rounds,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        device=cfg.device, n_jobs=cfg.n_jobs, random_state=seed, verbosity=0, **params)


def fit_xgb(Xtr, ytr, Xva, yva, cfg: XgbConfig, seed: int) -> dict:
    """Tune on ``(Xva, yva)`` logloss with optuna, then refit with the winning params.

    HPO is redone for every context: a victim retraining on poisoned data re-tunes too,
    so the hyperparameters are part of what the poison gets to move.
    """
    import optuna

    if not cfg.use_hpo:
        best, study_best = {}, None
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))

        def objective(trial):
            m = _make(_suggest(trial), cfg, seed)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return float(m.best_score)

        study.optimize(objective, n_trials=cfg.hpo_trials, show_progress_bar=False)
        best, study_best = study.best_params, float(study.best_value)

    model = _make(best, cfg, seed)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return {"model": model, "params": best, "val_logloss": study_best,
            "best_iteration": int(model.best_iteration), "refit_val_logloss": float(model.best_score)}


def score(model, Xte, yte) -> dict:
    p = model.predict_proba(Xte)
    return with_derived(binary_metrics(np.asarray(p, dtype=np.float64), np.asarray(yte)))


# --------------------------------------------------------------------------- summaries

def stats(values: list[float]) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def agg(records: list[dict], metrics=METRICS) -> dict:
    return {m: stats([r[m] for r in records if m in r]) for m in metrics}


def relative(delta_mean: float | None, clean_mean: float | None) -> float | None:
    if delta_mean is None or clean_mean is None or abs(clean_mean) < 1e-12:
        return None
    return 100.0 * delta_mean / abs(clean_mean)

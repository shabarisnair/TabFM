#!/usr/bin/env python
"""Does context poison optimised against TabPFNv2 transfer to a classical model?

The poisoned contexts are used as ordinary *training sets* for XGBoost, which is tuned
(optuna, logloss) and trained from scratch on each, then scored on the same
``test_attack_1000.csv`` the attack was optimised against. Two protocols:

  A  the validation set is an internal split of the given context, so a poisoned
     context also has a poisoned validation set (attacker owns the pipeline input)
  B  the validation set is a separate clean file (defender with a trusted holdout)

Clean baselines are run once per poisoned context with the matching seed, so the
reported delta is paired per run and aggregates over exactly the runs the TabPFNv2
numbers aggregate over.

Examples
--------
# Whole attack run: rebuilds the per-run-best poisoned contexts from trials/*.npz and
# compares against the TabPFNv2 deltas in its summary.json.
python scripts/xgb_transfer.py \
    --attack-dir results/main_experiments/lcld_v2/label-flip_random_r016 \
    --out results/xgb_transfer/lcld_v2/label-flip_random_r016

# Explicit files.
python scripts/xgb_transfer.py --model xgboost \
    --train datasets/lcld_v2/splits/selected/natural/context_5000.csv \
    --test  datasets/lcld_v2/splits/test_attack_1000.csv \
    --val   datasets/lcld_v2/splits/val_2000.csv \
    --poisoned-train results/.../context_poisoned.csv \
    --out results/xgb_transfer/one_off
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tabfm_experiments.io import setup_logger, sha256_file, write_json  # noqa: E402
from tabfm_experiments.transfer import (  # noqa: E402
    METRICS,
    XgbConfig,
    agg,
    fit_xgb,
    load_attack_contexts,
    relative,
    score,
    tabpfn_run_metrics,
)

PROTOCOL_DOC = {
    "A": "validation is an internal split of the given context (poisoned when the context is)",
    "B": "validation is the separate clean --val file",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xgboost", choices=["xgboost"],
                    help="surrogate model trained on the (poisoned) context")
    ap.add_argument("--train", type=Path, help="clean context CSV (default: --attack-dir's args.json)")
    ap.add_argument("--test", type=Path, help="test CSV, the set the attack was scored on")
    ap.add_argument("--val", type=Path, help="clean validation CSV, protocol B (default: <splits>/val_2000.csv)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="protocol A validation fraction of the context (default 0.2 -> 80/20)")
    ap.add_argument("--poisoned-train", type=Path, nargs="+", default=None,
                    help="one or more poisoned context CSVs; aggregated across them")
    ap.add_argument("--attack-dir", type=Path,
                    help="attack run dir; rebuilds the per-run-best poisoned contexts from trials/*.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default=None, help="label column (default: from split_info.json)")
    ap.add_argument("--protocols", nargs="+", default=["A", "B"], choices=["A", "B"])

    ap.add_argument("--hpo-trials", type=int, default=30, help="optuna trials per fitted model")
    ap.add_argument("--no-hpo", action="store_true", help="skip optuna, use XGBoost defaults")
    ap.add_argument("--n-estimators", type=int, default=2000, help="boosting-round cap before early stopping")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="xgboost device, e.g. cpu or cuda")
    ap.add_argument("--seed", type=int, default=0, help="base seed; model i uses seed + i")
    ap.add_argument("--n-clean-repeats", type=int, default=None,
                    help="clean baseline fits (default: as many as there are poisoned contexts)")
    ap.add_argument("--clean-cache-dir", type=Path, default=None,
                    help="reuse clean baselines across attack dirs of the same dataset")
    a = ap.parse_args(argv)

    if a.attack_dir is None and not a.poisoned_train:
        ap.error("give --poisoned-train or --attack-dir")
    if a.attack_dir is not None:
        args_json = json.loads((a.attack_dir / "args.json").read_text())
        a.train = a.train or Path(args_json["train"])
        a.test = a.test or Path(args_json["test"])
    if a.train is None or a.test is None:
        ap.error("--train and --test are required when --attack-dir is not given")
    if a.val is None:
        found = [p / "val_2000.csv" for p in list(Path(a.train).resolve().parents)[:4]
                 if (p / "val_2000.csv").exists()]
        a.val = found[0] if found else Path(a.train).parent / "val_2000.csv"
    return a


def frame_to_xy(df: pd.DataFrame, target: str, drop: tuple[str, ...]):
    X = df.drop(columns=[target, *[c for c in drop if c in df.columns]])
    return X.astype("float64").to_numpy(), df[target].to_numpy().astype(int), list(X.columns)


def split_context(X, y, ratio: float, seed: int):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=ratio, random_state=seed, stratify=y)


def fit_and_score(X, y, protocol: str, a, cfg: XgbConfig, seed: int, Xval, yval, Xte, yte, log):
    """One (context, protocol, seed) model: build the validation set, tune, fit, score."""
    if protocol == "A":
        Xtr, Xva, ytr, yva = split_context(X, y, a.val_ratio, seed)
    else:
        Xtr, ytr, Xva, yva = X, y, Xval, yval
    t0 = time.time()
    fit = fit_xgb(Xtr, ytr, Xva, yva, cfg, seed)
    m = score(fit["model"], Xte, yte)
    m.update(n_train=int(len(ytr)), n_val=int(len(yva)), seed=seed,
             best_iteration=fit["best_iteration"], params=fit["params"],
             val_logloss=fit["val_logloss"], seconds=round(time.time() - t0, 1))
    log.info(f"      seed {seed}: ce={m['ce']:.5f} acc={m['accuracy']:.4f} "
             f"bacc={m['balanced_accuracy']:.4f} mcc={m['mcc']:.4f} "
             f"(iters={m['best_iteration']}, {m['seconds']:.0f}s)")
    return m


def cache_key(a, protocol: str, n: int, train_sha: str, test_sha: str, val_sha: str | None) -> str:
    import hashlib

    parts = [a.model, protocol, train_sha, test_sha, val_sha or "-", f"{a.val_ratio}",
             f"{a.hpo_trials}", f"{a.no_hpo}", f"{a.n_estimators}", f"{a.early_stopping_rounds}",
             a.device, f"{a.seed}", f"{n}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def main(argv=None):
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(a.out, "xgb_transfer")
    write_json(a.out / "args.json", vars(a))

    from tabfm_experiments.data import drop_columns_for, find_target, infer_dataset_name

    target = find_target(a.train, a.target)
    drop = drop_columns_for(infer_dataset_name(a.train))
    train_df = pd.read_csv(a.train)
    test_df = pd.read_csv(a.test)
    log.info(f"{a.model}: train={a.train} ({len(train_df)} rows)  test={a.test} ({len(test_df)} rows)  "
             f"target={target}  drop={drop or '()'}")

    # --- poisoned contexts -------------------------------------------------------------
    tabpfn = None
    if a.attack_dir is not None:
        contexts, summary = load_attack_contexts(a.attack_dir, train_df, target)
        tabpfn = tabpfn_run_metrics(summary)
        log.info(f"rebuilt {len(contexts)} per-run-best contexts from {a.attack_dir}: "
                 f"{', '.join(t for t, _ in contexts)}  (attack={summary['attack']}, "
                 f"R={summary['row_percent']}%, k={summary['k']})")
    else:
        contexts = [(p.stem, pd.read_csv(p)) for p in a.poisoned_train]
        summary = None
        log.info(f"loaded {len(contexts)} poisoned contexts from --poisoned-train")

    n_clean = a.n_clean_repeats or len(contexts)
    Xte, yte, cols = frame_to_xy(test_df, target, drop)
    Xc, yc, cols_c = frame_to_xy(train_df, target, drop)
    if cols_c != cols:
        raise SystemExit("train and test feature columns differ")

    need_val = "B" in a.protocols
    if need_val and not Path(a.val).exists():
        raise SystemExit(f"protocol B needs a clean validation file; {a.val} not found")
    Xval = yval = None
    if need_val:
        val_df = pd.read_csv(a.val)
        Xval, yval, cols_v = frame_to_xy(val_df, target, drop)
        if cols_v != cols:
            raise SystemExit("val and test feature columns differ")
        log.info(f"clean val={a.val} ({len(yval)} rows)")

    cfg = XgbConfig(hpo_trials=a.hpo_trials, n_estimators=a.n_estimators,
                    early_stopping_rounds=a.early_stopping_rounds, n_jobs=a.n_jobs,
                    device=a.device, use_hpo=not a.no_hpo)

    shas = {"train": sha256_file(a.train), "test": sha256_file(a.test),
            "val": sha256_file(a.val) if need_val else None}

    # --- run both protocols ------------------------------------------------------------
    results = {}
    for protocol in a.protocols:
        log.info(f"\n== protocol {protocol}: {PROTOCOL_DOC[protocol]}")
        key = cache_key(a, protocol, n_clean, shas["train"], shas["test"], shas["val"])
        cache = (a.clean_cache_dir / f"clean_{key}.json") if a.clean_cache_dir else None

        if cache is not None and cache.exists():
            clean = json.loads(cache.read_text())["clean"]
            log.info(f"    clean baseline: {len(clean)} fits reused from {cache}")
        else:
            log.info(f"    clean baseline ({n_clean} fits)")
            clean = [fit_and_score(Xc, yc, protocol, a, cfg, a.seed + i, Xval, yval, Xte, yte, log)
                     for i in range(n_clean)]
            if cache is not None:
                write_json(cache, {"key": key, "protocol": protocol, "shas": shas, "clean": clean})

        log.info(f"    poisoned ({len(contexts)} contexts)")
        poisoned, delta = [], []
        for i, (tag, df) in enumerate(contexts):
            log.info(f"    -- {tag}")
            Xp, yp, cols_p = frame_to_xy(df, target, drop)
            if cols_p != cols:
                raise SystemExit(f"poisoned context {tag} has different feature columns")
            m = fit_and_score(Xp, yp, protocol, a, cfg, a.seed + i, Xval, yval, Xte, yte, log)
            m["tag"] = tag
            poisoned.append(m)
            # Paired with the clean fit that used the same seed: same split, same TPE stream.
            ref = clean[i % len(clean)]
            delta.append({"tag": tag, "seed": m["seed"], **{k: m[k] - ref[k] for k in METRICS}})

        results[protocol] = {
            "doc": PROTOCOL_DOC[protocol], "clean": clean, "poisoned": poisoned, "delta": delta,
            "clean_agg": agg(clean), "poisoned_agg": agg(poisoned), "delta_agg": agg(delta),
        }
        report(protocol, results[protocol], tabpfn, log)

    out = {
        "model": a.model, "train": str(a.train), "test": str(a.test), "val": str(a.val),
        "target": target, "n_context": len(train_df), "n_test": len(test_df),
        "n_features": len(cols), "val_ratio": a.val_ratio, "sha256": shas,
        "attack_dir": str(a.attack_dir) if a.attack_dir else None,
        "contexts": [t for t, _ in contexts],
        "aggregation": ("paired per run against the clean fit with the same seed, then "
                        f"aggregated across the {len(contexts)} contexts"),
        "xgb": {"hpo_trials": 0 if a.no_hpo else a.hpo_trials, "objective": "binary:logistic",
                "hpo_objective": "validation logloss", "n_estimators_cap": a.n_estimators,
                "early_stopping_rounds": a.early_stopping_rounds, "device": a.device},
        "protocols": results,
    }
    if summary is not None:
        out["attack"] = {"attack": summary["attack"], "row_percent": summary["row_percent"],
                         "k": summary["k"], "n_runs": summary["n_runs"],
                         "n_row_subsamples": summary["n_row_subsamples"],
                         "aggregation": summary["aggregation"]}
        out["tabpfn"] = {"per_run": tabpfn, "clean_agg": agg(tabpfn["clean"]),
                         "poisoned_agg": agg(tabpfn["poisoned"]), "delta_agg": agg(tabpfn["delta"])}
    write_json(a.out / "summary.json", out)
    write_json(a.out / "versions.json", versions())
    log.info(f"\nwrote {a.out / 'summary.json'}")
    return out


def versions() -> dict:
    import sklearn
    import optuna
    import xgboost

    return {"python": sys.version, "xgboost": xgboost.__version__, "optuna": optuna.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def report(protocol: str, r: dict, tabpfn: dict | None, log) -> None:
    """One block per protocol: clean, poisoned, delta, relative %, and the TabPFN delta."""
    tp = agg(tabpfn["delta"]) if tabpfn else None
    tpc = agg(tabpfn["clean"]) if tabpfn else None
    head = f"    {'metric':<19s} {'clean':>17s} {'poisoned':>17s} {'delta':>17s} {'rel%':>9s}"
    if tp:
        head += f" | {'TabPFN d':>17s} {'TabPFN rel%':>12s} {'transfer%':>10s}"
    log.info(f"\n  protocol {protocol} -- {r['doc']}")
    log.info(head)
    log.info("    " + "-" * (len(head) - 4))
    for m in METRICS:
        c, p, d = r["clean_agg"][m], r["poisoned_agg"][m], r["delta_agg"][m]
        line = (f"    {m:<19s} {fmt(c):>17s} {fmt(p):>17s} {fmt(d, sign=True):>17s} "
                f"{pct(relative(d['mean'], c['mean']), weak_baseline(m, c)):>9s}")
        if tp:
            td, tc = tp[m], tpc[m]
            tr = (100.0 * d["mean"] / td["mean"]) if td["mean"] and abs(td["mean"]) > 1e-12 else None
            line += (f" | {fmt(td, sign=True):>17s} "
                     f"{pct(relative(td['mean'], tc['mean']), weak_baseline(m, tc)):>12s} "
                     f"{pct(tr, noisy(td), digits=0):>10s}")
        log.info(line)
    if tp:
        log.info("    transfer% = XGBoost delta / TabPFNv2 delta; 100 means the poison hurts both equally.")
    log.info("    '~' marks a ratio whose denominator is near zero or inside its own spread -- read the "
             "absolute columns there.")


MCC_UNSTABLE = 0.15          # |clean mcc| below this -> the percentage is noise


def weak_baseline(metric: str, clean: dict) -> bool:
    """True when the clean value is too small for a relative change to mean anything."""
    m = clean.get("mean")
    return m is not None and (abs(m) < MCC_UNSTABLE if metric in ("mcc", "f1") else False)


def noisy(delta: dict) -> bool:
    """True when a delta is not distinguishable from zero across runs, so ratios blow up."""
    return (delta.get("mean") is not None and delta.get("std") is not None
            and abs(delta["mean"]) <= delta["std"])


def pct(v: float | None, unstable: bool = False, digits: int = 1) -> str:
    return "-" if v is None else f"{'~' if unstable else ''}{v:+.{digits}f}"


def fmt(s: dict, sign: bool = False) -> str:
    if s.get("mean") is None:
        return "-"
    return f"{s['mean']:{'+' if sign else ''}.4f} ± {s['std']:.4f}"


if __name__ == "__main__":
    main()

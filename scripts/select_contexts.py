#!/usr/bin/env python
"""Validation-selected ICL contexts for every dataset x {natural, balanced}.

For each combination: dedup the raw CSV by row fingerprint, exclude rows that appear
in val_2000.csv or test_attack_1000.csv, draw 10 candidate contexts (seeds 0..9),
score each with the shared TabPFNv2 runtime on val_2000.csv (accuracy), keep the
winner and nest 5000 / 1000 children inside it. Writes only under
datasets/<name>/splits/selected/<mode>/.

python scripts/select_contexts.py --gpu 1
python scripts/select_contexts.py --gpu 1 --dataset url_unique --mode natural
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tabfm_experiments.config import (  # noqa: E402
    DATASETS,
    SELECT_BALANCED_CAP,
    SELECT_CHILD_SEED_OFFSET,
    SELECT_CHILD_SIZES,
    SELECT_N_CANDIDATES,
    DatasetInfo,
)
from tabfm_experiments.data import fingerprint_frame  # noqa: E402
from tabfm_experiments.io import setup_logger, sha256_file, write_json  # noqa: E402

RESERVED_FILES = ("val_2000.csv", "test_attack_1000.csv")
MODES = ("natural", "balanced")


# ------------------------------------------------------------------ pure helpers
def prepare_frame(df: pd.DataFrame, drop_columns, columns: list[str]) -> pd.DataFrame:
    """Drop raw-only columns (e.g. LCLD ``issue_d``) and order columns like the split files."""
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from raw frame: {missing[:5]}")
    return df[columns]


def build_pool(raw: pd.DataFrame, reserved: list[pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Dedup ``raw`` by fingerprint (keep first) and drop rows whose fingerprint is reserved.

    All frames must already share the same column order. Returns the pool (fresh
    RangeIndex), its fingerprints, and bookkeeping counts.
    """
    fp = fingerprint_frame(raw)
    dup = pd.Series(fp).duplicated(keep="first").to_numpy()
    reserved_fp: set[str] = set()
    for r in reserved:
        if list(r.columns) != list(raw.columns):
            raise ValueError("reserved frame columns differ from raw columns")
        reserved_fp.update(fingerprint_frame(r))
    in_reserved = pd.Series(fp).isin(reserved_fp).to_numpy()
    keep = ~dup & ~in_reserved
    stats = {
        "n_raw_rows": int(len(raw)),
        "duplicate_count": int(dup.sum()),
        "reserved_fingerprints_count": int(len(reserved_fp)),
        "raw_rows_matching_reserved": int(in_reserved.sum()),
        "unique_raw_rows_matching_reserved": int((in_reserved & ~dup).sum()),
        "pool_size": int(keep.sum()),
    }
    return raw.loc[keep].reset_index(drop=True), fp[keep], stats


def candidate_size(mode: str, y_pool: np.ndarray, natural_size: int) -> int:
    if mode == "natural":
        return int(min(natural_size, len(y_pool)))
    counts = np.bincount(np.asarray(y_pool).astype(int), minlength=2)
    return int(min(SELECT_BALANCED_CAP, 2 * counts.min()))


def draw_candidate(y: np.ndarray, size: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Sorted, unique row indices. Balanced draws ``size // 2`` rows per class."""
    y = np.asarray(y).astype(int)
    if mode == "natural":
        return np.sort(rng.choice(len(y), size=size, replace=False))
    per = size // 2
    parts = [rng.choice(np.flatnonzero(y == c), size=per, replace=False) for c in (0, 1)]
    return np.sort(np.concatenate(parts))


def nested_children(parent: np.ndarray, y: np.ndarray, sizes, mode: str,
                    rng: np.random.Generator) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Children of ``parent`` without replacement, each nested in the previous (larger) one."""
    children, skipped = {}, {}
    cur = np.asarray(parent)
    y = np.asarray(y).astype(int)
    for size in sorted(sizes, reverse=True):
        if size >= len(parent):
            skipped[size] = (f"winner has {len(parent)} rows (<= {size})"
                             if size > len(parent) else f"winner already has exactly {size} rows")
            continue
        if len(cur) < size:
            skipped[size] = f"parent has {len(cur)} rows < {size}"
            continue
        if mode == "natural":
            child = rng.choice(cur, size=size, replace=False)
        else:
            per = size // 2
            child = np.concatenate([rng.choice(cur[y[cur] == c], size=per, replace=False) for c in (0, 1)])
        cur = np.sort(child)
        children[size] = cur
    return children, skipped


def class_counts(y) -> dict[str, int]:
    c = np.bincount(np.asarray(y).astype(int), minlength=2)
    return {"0": int(c[0]), "1": int(c[1])}


# ---------------------------------------------------------------------- driver
def select_one(info: DatasetInfo, mode: str, *, clf, device: str, pool_cache: dict, n_candidates: int,
               val_batch_size: int | None, log, model_seed: int, deterministic_mode: str) -> dict:
    import torch
    from tabfm_experiments.metrics import binary_metrics
    from tabfm_experiments.runtime import evaluate_context

    splits = info.splits_dir
    val_df = pd.read_csv(splits / "val_2000.csv")
    columns = list(val_df.columns)
    target = info.target

    if info.name not in pool_cache:
        t0 = time.time()
        raw = prepare_frame(pd.read_csv(info.raw_csv, low_memory=False), info.drop_columns, columns)
        reserved = [pd.read_csv(splits / f)[columns] for f in RESERVED_FILES]
        pool, pool_fp, stats = build_pool(raw, reserved)
        del raw
        pool_cache.clear()
        pool_cache[info.name] = (pool, pool_fp, stats, reserved)
        log.info(f"  pool {info.name}: {stats}  [{time.time() - t0:.0f}s]")
    pool, pool_fp, stats, reserved = pool_cache[info.name]

    feat = [c for c in columns if c != target]
    y_pool = pool[target].to_numpy().astype(int)
    X_pool = pool[feat].to_numpy(dtype=np.float32)
    Xv = torch.tensor(val_df[feat].to_numpy(dtype=np.float32), device=device)
    yv_np = val_df[target].to_numpy().astype(int)
    yv = torch.tensor(yv_np, device=device)

    size = candidate_size(mode, y_pool, info.natural_size)
    log.info(f"  [{info.name}/{mode}] candidate size {size} (pool {len(pool)}, classes {class_counts(y_pool)})")
    cands = []
    for seed in range(n_candidates):
        idx = draw_candidate(y_pool, size, mode, np.random.default_rng(seed))
        t0 = time.time()
        _, _, probs = evaluate_context(clf, torch.tensor(X_pool[idx], device=device),
                                       torch.tensor(y_pool[idx].astype(np.float32), device=device),
                                       Xv, yv, need_grad=False, test_batch_size=val_batch_size)
        m = binary_metrics(probs.cpu().numpy(), yv_np)
        cands.append({"seed": seed, "idx": idx, "accuracy": m.accuracy, "val_metrics": m.as_dict(),
                      "class_counts": class_counts(y_pool[idx])})
        log.info(f"    seed {seed}: acc={m.accuracy:.4f} auc={m.roc_auc:.4f} ce={m.ce:.4f}  [{time.time() - t0:.1f}s]")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    winner = max(cands, key=lambda c: (c["accuracy"], -c["seed"]))
    child_seed = SELECT_CHILD_SEED_OFFSET + winner["seed"]
    children, skipped = nested_children(winner["idx"], y_pool, SELECT_CHILD_SIZES, mode,
                                        np.random.default_rng(child_seed))

    out_dir = splits / "selected" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {len(winner["idx"]): winner["idx"], **children}
    reserved_fp = set()
    for r in reserved:
        reserved_fp.update(fingerprint_frame(r))
    file_info, checks = {}, {"unique_rows": {}, "overlap_with_reserved": {}}
    for n, idx in sorted(files.items(), reverse=True):
        name = f"context_{n}.csv"
        pool.iloc[idx][columns].to_csv(out_dir / name, index=False)
        fps = pool_fp[idx]
        checks["unique_rows"][name] = bool(len(set(fps)) == len(fps))
        checks["overlap_with_reserved"][name] = int(sum(f in reserved_fp for f in fps))
        file_info[name] = {"n": int(len(idx)), "class_counts": class_counts(y_pool[idx]),
                           "sha256": sha256_file(out_dir / name)}
    sets = [set(pool_fp[files[n]]) for n in sorted(files)]
    checks["nesting"] = bool(all(a <= b for a, b in zip(sets, sets[1:])))
    checks["nesting_order"] = [f"context_{n}.csv" for n in sorted(files)]
    written_cols = list(pd.read_csv(out_dir / f"context_{len(winner['idx'])}.csv", nrows=1).columns)
    checks["issue_d_absent"] = "issue_d" not in written_cols
    checks["columns_match_val"] = written_cols == columns

    from tabfm_experiments.io import versions_info

    manifest = {
        "dataset": info.name, "mode": mode, "target": target, "dropped_columns": list(info.drop_columns),
        "columns": columns, **stats, "candidate_size": size,
        "natural_size_requested": info.natural_size if mode == "natural" else None,
        "balanced_cap": SELECT_BALANCED_CAP if mode == "balanced" else None,
        "selection_metric": "accuracy on val_2000.csv", "tie_break": "lower seed",
        "candidates": [{k: v for k, v in c.items() if k != "idx"} for c in cands],
        "candidate_seeds": [c["seed"] for c in cands],
        "winning_seed": winner["seed"], "winning_accuracy": winner["accuracy"],
        "child_seed": child_seed, "skipped_children": {str(k): v for k, v in skipped.items()},
        "files": file_info, "checks": checks,
        "source_hashes": {"raw_csv": sha256_file(info.raw_csv),
                          **{f: sha256_file(splits / f) for f in RESERVED_FILES}},
        "model": {"name": "tabpfnv2", "model_seed": model_seed, "deterministic_mode": deterministic_mode,
                  "n_estimators": 1, "device": device, "val_batch_size": val_batch_size},
        "versions": versions_info(clf),
    }
    write_json(out_dir / "manifest.json", manifest)
    ok = checks["nesting"] and all(checks["unique_rows"].values()) and not any(checks["overlap_with_reserved"].values())
    log.info(f"  [{info.name}/{mode}] winner seed {winner['seed']} acc={winner['accuracy']:.4f}  "
             f"files {sorted(file_info)}  checks {'OK' if ok and checks['issue_d_absent'] else 'FAILED'}  -> {out_dir}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--deterministic-mode", default="best-effort", choices=["best-effort", "strict"])
    ap.add_argument("--n-candidates", type=int, default=SELECT_N_CANDIDATES)
    ap.add_argument("--val-batch-size", type=int, default=None,
                    help="chunk val rows (use only if a 10k context OOMs)")
    a = ap.parse_args(argv)

    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu

    log = setup_logger(None, "select_contexts")
    device = device_from_gpu(a.gpu)
    names = list(DATASETS) if a.dataset == "all" else [a.dataset]
    modes = list(MODES) if a.mode == "all" else [a.mode]
    cache: dict = {}
    for name in names:
        # One classifier per dataset: TabPFN caches the feature schema on the first fit.
        clf = build_tabpfn_v2(device, a.model_seed, a.deterministic_mode)
        for mode in modes:
            select_one(DATASETS[name], mode, clf=clf, device=device, pool_cache=cache, n_candidates=a.n_candidates,
                       val_batch_size=a.val_batch_size, log=log, model_seed=a.model_seed,
                       deterministic_mode=a.deterministic_mode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train / validation / attack-test splits for the multiclass datasets.

Follows make_splits.py where it applies -- a stratified 75/25 train/test split with seed
0, the context drawn from the train side, the attack test set drawn from the test side --
and adds a VALIDATION set drawn from the SAME distribution as the attack test set:

    1. stratified 75/25 split of the deduplicated rows   -> train.csv / test.csv
    2. stratified draw of 2N rows from test.csv          (the "bigger test set")
    3. stratified half of those 2N                       -> val_<N>.csv
       the other half                                    -> test_attack_<N>.csv
    4. stratified draw of M rows from train.csv          -> context_<M>.csv

so val and the attack test are two disjoint halves of one sample from the held-out
split, and neither shares a row with the context (they come from opposite sides of
step 1). All draws are stratified on the target with an exact largest-remainder
allocation, so every file has the requested size and the class prior of its parent.

Rows are copied BYTE-FOR-BYTE from the source CSV, as make_splits.py does, so no value is
re-formatted on the way through. A split_info.json alongside records the target, the
seed, every file's size and class balance, and the disjointness checks; the existing
loaders (tabfm_experiments.data.load_split) read the target column from it.

NOT DONE HERE: the candidate-context selection step (select_contexts.py) that the binary
datasets went through. It scores candidates with the shared runtime, which is binary-
only -- runtime.py fixes clf.n_classes_ = 2 and truncates the softmax to two logits -- so
on these datasets it would not fail, it would silently rank candidates on a wrong
2-class problem. It has to wait until the runtime is generalised.

    python scripts/prepare_multiclass_splits.py --dataset gas-drift --context 5000 --eval 1000
    python scripts/prepare_multiclass_splits.py --dataset students_dropout --context 3000 --eval 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tabfm_experiments.config import DATASETS_DIR  # noqa: E402
from tabfm_experiments.data import fingerprint_frame  # noqa: E402

TEST_FRAC, SEED = 0.25, 0          # identical to make_splits.py
KNOWN_TARGETS = {"gas-drift": "Class", "students_dropout": "AcademicOutcome"}


def stratified_split(idx: np.ndarray, y: np.ndarray, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Per-class shuffle, then round(len * frac) of each class goes to the second part.

    The same rule as make_splits.strat_split, so the train/test boundary is drawn the
    way it was for every other dataset.
    """
    a, b = [], []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c].copy()
        rng.shuffle(members)
        k = int(round(len(members) * frac))
        b.append(members[:k])
        a.append(members[k:])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def stratified_exact(idx: np.ndarray, y: np.ndarray, n: int, rng) -> np.ndarray:
    """Exactly ``n`` rows from ``idx``, class counts proportional to y[idx].

    Largest-remainder allocation: floor every class's share, then hand the leftover
    rows to the classes with the largest fractional parts. Unlike rounding each class
    independently and truncating to n, this neither overshoots nor drops rows by their
    position in the file.
    """
    if n > len(idx):
        raise ValueError(f"asked for {n} rows from a pool of {len(idx)}")
    classes, counts = np.unique(y[idx], return_counts=True)
    share = n * counts / counts.sum()
    quota = np.floor(share).astype(int)
    leftover = n - int(quota.sum())
    quota[np.argsort(-(share - quota), kind="stable")[:leftover]] += 1
    out = []
    for c, q in zip(classes, quota):
        members = idx[y[idx] == c]
        if q > len(members):
            raise ValueError(f"class {c}: quota {q} exceeds its {len(members)} rows")
        out.append(rng.choice(members, size=int(q), replace=False))
    picked = np.sort(np.concatenate(out))
    assert len(picked) == n
    return picked


def balance(y: np.ndarray, idx: np.ndarray) -> dict:
    c = pd.Series(y[idx]).value_counts().sort_index()
    return {str(int(k)): f"{int(v)} ({100 * v / len(idx):.1f}%)" for k, v in c.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--context", type=int, required=True, help="context (train) rows, M")
    ap.add_argument("--eval", type=int, required=True,
                    help="rows in EACH of val and the attack test set, N (2N are drawn)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    root = DATASETS_DIR / a.dataset
    src = root / f"{a.dataset}.csv"
    target = a.target or KNOWN_TARGETS.get(a.dataset)
    if target is None:
        raise SystemExit("--target is required for an unknown dataset")
    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if target not in df.columns:
        raise SystemExit(f"target {target!r} not in {src}")
    if df.isna().any().any():
        raise SystemExit(f"{src} has missing cells; clean it first")
    meta = root / f"{a.dataset}_metadata.csv"
    if meta.exists():
        feats = list(pd.read_csv(meta)["feature"])
        if feats != [c for c in df.columns if c != target]:
            raise SystemExit("metadata feature list does not match the CSV columns / order")

    # Dedup on the full row (features + label), keeping the first -- the rule
    # select_contexts.py applies to the binary datasets. A no-op on both datasets here,
    # but it keeps the guarantee below honest if the source ever changes.
    fp = fingerprint_frame(df)
    keep = np.flatnonzero(~pd.Series(fp).duplicated(keep="first").to_numpy())
    y = df[target].to_numpy().astype(int)
    rng = np.random.default_rng(a.seed)

    tr, te = stratified_split(keep, y, TEST_FRAC, rng)
    held = stratified_exact(te, y, 2 * a.eval, rng)       # the "bigger test set"
    val = stratified_exact(held, y, a.eval, rng)
    atk = np.setdiff1d(held, val)
    ctx = stratified_exact(tr, y, a.context, rng)

    # ---- guarantees, checked rather than assumed ----------------------------------
    fps = {name: set(fp[ix]) for name, ix in (("context", ctx), ("val", val), ("test_attack", atk))}
    overlaps = {f"{p}&{q}": len(fps[p] & fps[q])
                for p, q in (("context", "val"), ("context", "test_attack"), ("val", "test_attack"))}
    if any(overlaps.values()):
        raise SystemExit(f"splits share rows: {overlaps}")
    assert len(val) == len(atk) == a.eval and len(ctx) == a.context
    prior = lambda ix: np.bincount(y[ix], minlength=y.max() + 1) / len(ix)  # noqa: E731
    max_prior_gap = float(np.abs(prior(val) - prior(atk)).max())

    files = {"train.csv": tr, "test.csv": te, f"context_{a.context}.csv": ctx,
             f"val_{a.eval}.csv": val, f"test_attack_{a.eval}.csv": atk}

    # Byte-for-byte copy of the selected source lines, as make_splits.py does.
    lines = src.read_text().splitlines(keepends=True)
    header, body = lines[0], lines[1:]
    assert len(body) == len(df), "line count disagrees with the parsed frame"
    print(f"=== {a.dataset} ===  {len(df)} rows, {df.shape[1] - 1} features, "
          f"target {target!r} with {len(np.unique(y))} classes\n")
    info = {"source": src.name, "target": target, "n_classes": int(len(np.unique(y))),
            "n_rows": int(len(df)), "n_rows_after_dedup": int(len(keep)),
            "seed": a.seed, "test_frac": TEST_FRAC,
            "protocol": ("stratified 75/25 train/test; context drawn from train; a 2N "
                         "stratified draw from test split into val and test_attack halves"),
            "context_selection": ("NOT performed -- the shared runtime is binary-only "
                                  "(see prepare_multiclass_splits.py)"),
            "disjoint_check": overlaps, "val_vs_test_attack_max_class_prior_gap": max_prior_gap,
            "splits": {}}
    for name, ix in files.items():
        (out / name).write_text(header + "".join(body[i] for i in ix))
        info["splits"][name] = {"n": int(len(ix)), "balance": balance(y, ix)}
        print(f"  {name:<24s} {len(ix):>6d}  {balance(y, ix)}")
    (out / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\n  context / val / test_attack pairwise disjoint: {overlaps}")
    print(f"  largest class-share difference between val and test_attack: {max_prior_gap:.4f}")
    print(f"  wrote {out}/")
    return info


if __name__ == "__main__":
    main()

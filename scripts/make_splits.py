#!/usr/bin/env python3
"""Build train/test splits + context subsamples for the TabularBench datasets.

Protocol follows Djilani et al., "On the Robustness of Tabular Foundation Models"
(arXiv:2506.02978):
  * random split stratified by target class, 75% train / 25% test
    (their LCLD split is 915K/305K of 1,220,092 = 75/25)
  * TFMs cap the context at 10k rows, so the context is subsampled from train;
    the paper ablates context size over {1000, 5000, 10000}
  * subsamples are produced both unbalanced (natural prior) and rebalanced,
    because the paper selects between the two by validation MCC
  * validation rows are carved out of train FIRST so they are disjoint from
    every context ("a subset of the train data that is never perturbed")
"""
import argparse, csv, json, os, random, sys

ROOT = "/home/ssn899/Desktop/TabFM/datasets"
ALL = {                            # dir : (csv relative to ROOT, target column)
    "url":         ("url/url.csv",                 "is_phishing"),
    "url_unique":  ("url_unique/url_unique.csv",   "is_phishing"),
    "lcld_v2":     ("lcld_v2/lcld_v2.csv",         "charged_off"),
    "wids":        ("wids/wids.csv",               "hospital_death"),
    "coil2000_insurance_policies": ("coil2000_insurance_policies/coil2000_insurance_policies.csv", "MobileHomePolicy"),
}

_ap = argparse.ArgumentParser(description="Build stratified splits for one or more datasets.")
_ap.add_argument("names", nargs="*", default=None,
                 help=f"dataset dirs to build (default: all of {sorted(ALL)})")
_args = _ap.parse_args()
DATASETS = ({n: ALL[n] for n in _args.names} if _args.names else dict(ALL))
TEST_FRAC, N_VAL, N_ATTACK, SEED = 0.25, 2000, 1000, 0
CONTEXT_SIZES = [1000, 5000, 10000]

def scan(path, target):
    """One pass: record (line_index, label) only. Keeps LCLD out of RAM."""
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        ti = hdr.index(target)
        labels = [row[ti] for row in rdr if row]
    return hdr, ti, labels

def strat_split(labels, frac, rng):
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(y, []).append(i)
    a, b = [], []
    for y, idx in sorted(by.items()):
        rng.shuffle(idx)
        k = round(len(idx) * frac)
        b.extend(idx[:k]); a.extend(idx[k:])
    return sorted(a), sorted(b)

def take(pool, labels, n, rng, balanced):
    """Sample n indices from pool; stratified to the natural prior, or 50/50."""
    by = {}
    for i in pool:
        by.setdefault(labels[i], []).append(i)
    for v in by.values():
        rng.shuffle(v)
    out, classes = [], sorted(by)
    if balanced:
        per = n // len(classes)
        for y in classes:
            out.extend(by[y][:per])
    else:
        for y in classes:
            out.extend(by[y][:round(n * len(by[y]) / len(pool))])
    return sorted(out)[:n]

def write(src, hdr, want, dst):
    want = set(want)
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        rdr = csv.reader(fi); next(rdr)
        w = csv.writer(fo); w.writerow(hdr)
        n = 0
        for i, row in enumerate(rdr):
            if i in want:
                w.writerow(row); n += 1
    return n

def balance_of(labels, idx):
    c = {}
    for i in idx:
        c[labels[i]] = c.get(labels[i], 0) + 1
    t = len(idx)
    return {k: f"{v} ({100*v/t:.1f}%)" for k, v in sorted(c.items())}

for name, (rel, target) in DATASETS.items():
    src = os.path.join(ROOT, rel)
    out = os.path.join(ROOT, name, "splits")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(SEED)
    print(f"\n=== {name} ===", flush=True)

    hdr, ti, labels = scan(src, target)
    print(f"  {len(labels):,} rows, {len(hdr)} cols, target '{target}' at index {ti}")

    tr, te = strat_split(labels, TEST_FRAC, rng)
    val = take(tr, labels, N_VAL, rng, balanced=False)
    pool = sorted(set(tr) - set(val))
    atk = take(te, labels, N_ATTACK, rng, balanced=False)

    files = {"train.csv": tr, "test.csv": te,
             f"val_{N_VAL}.csv": val, f"test_attack_{N_ATTACK}.csv": atk}
    for n in CONTEXT_SIZES:
        if n <= len(pool):
            files[f"context_{n}.csv"] = take(pool, labels, n, rng, balanced=False)
    files[f"context_5000_balanced.csv"] = take(pool, labels, 5000, rng, balanced=True)

    info = {"source": rel, "target": target, "target_index": ti,
            "n_rows": len(labels), "n_cols": len(hdr),
            "seed": SEED, "test_frac": TEST_FRAC, "splits": {}}
    for fn, idx in files.items():
        k = write(src, hdr, idx, os.path.join(out, fn))
        info["splits"][fn] = {"n": k, "balance": balance_of(labels, idx)}
        print(f"  {fn:28s} {k:>8,}  {balance_of(labels, idx)}")
    with open(os.path.join(out, "split_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

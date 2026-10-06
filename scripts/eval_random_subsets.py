#!/usr/bin/env python
"""The size-matched do-nothing control: m rows drawn at RANDOM from the poisoned set.

The 'poisoned' baseline in the comparison uses the whole contaminated context (n rows),
so pitting a 250-row sifted subset against it mixes two effects: whether the method
picked good rows, and whether a 250-row context is better or worse than a 5000-row one
in the first place.

random-m removes that confound. It keeps the same size and the same class quota as the
methods, but ranks rows at RANDOM instead of by a learned score, so its purity is 1-R by
construction. A method only demonstrates that its RANKING is worth anything if it beats
this, not merely the full poisoned set.

Averaged over --n-draws random draws.

    python scripts/eval_random_subsets.py --gpu 0
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits  # noqa: E402
from tabfm_experiments.io import write_json  # noqa: E402
from tabfm_experiments.sift import select_clean_subset, subset_report  # noqa: E402

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies", "heloc", "lcld_v2"]
ATTACKS = ["label-flip-random", "hub-flip"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--small", type=int, default=250)
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("results/tables/comparison/random_subset_downstream.json"))
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.defense import METRICS, score_on_test
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    dev = device_from_gpu(a.gpu)
    out = {}
    for ds in DATASETS:
        train, test = default_splits(ds)
        Xdf, y_clean, _ = load_split(train)
        Xte_df, y_te, _ = load_split(test)
        X_all = torch.tensor(Xdf.to_numpy(np.float32), device=dev)
        X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=dev)
        y_clean = np.asarray(y_clean).astype(int)
        y_te = np.asarray(y_te).astype(int)
        n = len(y_clean)
        clf = build_tabpfn_v2(dev, 0)

        def tab(rows, labels):
            rows = np.asarray(rows)
            i = torch.as_tensor(rows, dtype=torch.long, device=dev)
            yc = torch.as_tensor(labels[rows].astype(np.float32), device=dev)
            return score_on_test(clf, X_all[i], yc, X_test, y_te, evaluate_context)

        for atk in ATTACKS:
            f = sorted(glob.glob(f"results/defense_kfold/{ds}/{atk}_r030/p0_f*_scores.npz"))
            if not f:
                continue
            z = np.load(f[0])
            flip, y_given = z["flipped_indices"], z["y_given"].astype(int)
            k = len(flip)
            rec = {"n_context": n, "k_flipped": int(k), "n_draws": a.n_draws}
            for m in (a.small, n - k):
                per, pur = [], []
                for d in range(a.n_draws):
                    # A random score -> the class quota is honoured, the ordering is not
                    # informative. Exactly the methods' setting minus the ranking.
                    score = np.random.default_rng(d).random(n)
                    kept = select_clean_subset(score, y_given, m, a.strategy)
                    per.append(tab(kept, y_given))
                    pur.append(subset_report(kept, flip, n)["purity"])
                rec[f"{a.strategy}_{m}"] = {mm: float(np.mean([p[mm] for p in per])) for mm in METRICS}
                rec[f"{a.strategy}_{m}"]["purity"] = float(np.mean(pur))
                print(f"  {ds:<30s} {atk:<18s} m={m:<5d} purity {np.mean(pur):.4f}  "
                      f"roc_auc {rec[f'{a.strategy}_{m}']['roc_auc']:.4f}")
            out[f"{ds}|{atk}"] = rec
        del clf
        torch.cuda.empty_cache()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, out)
    print(f"\nwrote {a.out}")
    return out


if __name__ == "__main__":
    main()

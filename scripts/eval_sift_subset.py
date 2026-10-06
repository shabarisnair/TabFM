#!/usr/bin/env python
"""Score a sifted subset on the held-out test set, against the right controls.

A sifted subset is both SMALLER and PURER than the poisoned context, and those two
things pull in opposite directions. Comparing "top-m by X" straight to the oracle
(which removes exactly the poison and keeps every clean row) therefore measures both at
once. The controls separate them:

    clean         all 5000 rows, true labels          ceiling
    poisoned      all 5000 rows, given labels         floor
    oracle        3500 rows, poison perfectly removed the full-budget target
    sift-m        the top m by X                      the method
    pure-m        m rows drawn from the CLEAN rows    perfect purity AT THAT SIZE
    random-m      m rows drawn from D at random       that size, no purity gain

sift-m vs pure-m is the cost of the poison the sift left in.
pure-m vs oracle is the cost of the smaller context.

    python scripts/eval_sift_subset.py --gpu 0 --run results/sift/url_unique/hub-flip_r030_lr5_it20 --iter 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits  # noqa: E402
from tabfm_experiments.defense import METRICS, score_on_test  # noqa: E402
from tabfm_experiments.sift import select_clean_subset, subset_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", default="url_unique")
    ap.add_argument("--run", type=Path, required=True, help="a results/sift/<ds>/<tag> directory")
    ap.add_argument("--iter", type=int, default=-1, help="which saved X to use (-1 = final)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000])
    ap.add_argument("--strategy", default="balanced", choices=["balanced", "proportional", "global"])
    ap.add_argument("--n-control-draws", type=int, default=5,
                    help="random/pure controls are averaged over this many draws")
    ap.add_argument("--model-seed", type=int, default=0)
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    z = np.load(a.run / "history.npz")
    x = z["x_history"][a.iter]
    flipped, y_clean, y_given = z["flipped_indices"], z["y_clean"].astype(int), z["y_given"].astype(int)
    n = len(y_clean)
    clean_rows = np.setdiff1d(np.arange(n), flipped)

    train, test = default_splits(a.dataset)
    device = device_from_gpu(a.gpu)
    Xdf, y_tr, _ = load_split(train)
    Xte_df, y_te, _ = load_split(test)
    X_ctx = torch.tensor(Xdf.to_numpy(np.float32), device=device)
    X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=device)
    y_te = np.asarray(y_te).astype(int)
    assert np.array_equal(np.asarray(y_tr).astype(int), y_clean), "context labels moved"

    clf = build_tabpfn_v2(device, a.model_seed)

    def score(rows, labels):
        idx = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=device)
        yc = torch.as_tensor(labels[rows].astype(np.float32), device=device)
        return score_on_test(clf, X_ctx[idx], yc, X_test, y_te, evaluate_context)

    def show(name, m, purity, res):
        print(f"  {name:<14s} {m:>5d} {('-' if purity is None else format(purity, '.4f')):>8s}"
              + "".join(f"{res[k]:>11.4f}" for k in METRICS[:5]))

    print(f"{a.dataset}  {a.run.name}  X from iter {a.iter}  quota={a.strategy}\n")
    print(f"  {'context':<14s} {'rows':>5s} {'purity':>8s}"
          + "".join(f"{k:>11s}" for k in METRICS[:5]))
    print("  " + "-" * 82)

    show("clean", n, 1.0, score(np.arange(n), y_clean))
    show("poisoned", n, 1 - len(flipped) / n, score(np.arange(n), y_given))
    show("oracle", len(clean_rows), 1.0, score(clean_rows, y_given))
    print("  " + "-" * 82)

    rng = np.random.default_rng(0)
    for m in a.sizes:
        kept = select_clean_subset(x, y_given, m, a.strategy)
        rep = subset_report(kept, flipped, n)
        show(f"sift-{m}", m, rep["purity"], score(kept, y_given))
        for label, pool in (("pure", clean_rows), ("random", np.arange(n))):
            accum = []
            for d in range(a.n_control_draws):
                sel = rng.choice(pool, m, replace=False)
                accum.append(score(sel, y_given))
            avg = {k: float(np.mean([r[k] for r in accum])) for k in METRICS}
            pur = 1.0 if label == "pure" else 1 - len(flipped) / n
            show(f"  {label}-{m}", m, pur, avg)
        print("  " + "-" * 82)
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Targeted attack: inject ONE test row into the context with its label flipped.

For a chosen test row (x, y) the context becomes ``context_5000 + (x, 1-y)`` -- 5001
rows, one of which is a verbatim copy of the query carrying the wrong label. The model
is then asked about x. Does one poisoned near-duplicate outvote 5000 clean rows?

Each sample is scored three ways on the same query:
    clean     the untouched 5000-row context
    flipped   + (x, 1-y)      <- the attack
    honest    + (x, y)        <- control: the same duplication, correct label

The honest control separates "the model copies a near-duplicate" from "adding any row
moved the prediction".

    python scripts/single_sample_flip.py --gpu 3
    python scripts/single_sample_flip.py --gpu 3 --n-samples 20 --datasets wids
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tabfm_experiments.config import DATASETS_DIR  # noqa: E402
from tabfm_experiments.io import write_json  # noqa: E402

DEFAULT_DATASETS = ["coil2000_insurance_policies", "lcld_v2", "url_unique", "wids"]


def probs_for_query(clf, X_ctx, y_ctx, x_row, y_true, evaluate_context):
    """p(class) for the single query row under the given context."""
    _, _, p = evaluate_context(clf, X_ctx, y_ctx, x_row[None, :],
                               torch.tensor([y_true], dtype=torch.long, device=x_row.device),
                               need_grad=False)
    return p[0].float().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="which test rows are drawn")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/single_sample_flip"))
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    device = device_from_gpu(a.gpu)
    report = {"n_samples": a.n_samples, "seed": a.seed, "datasets": {}}

    for ds in a.datasets:
        ctx = DATASETS_DIR / ds / "splits/selected/natural/context_5000.csv"
        tst = DATASETS_DIR / ds / "splits/test_attack_1000.csv"
        Xtr_df, ytr, cols = load_split(ctx)
        Xte_df, yte, _ = load_split(tst)
        Xc = torch.tensor(Xtr_df.to_numpy(np.float32), device=device)
        yc = torch.tensor(ytr.astype(np.float32), device=device)
        Xt = torch.tensor(Xte_df.to_numpy(np.float32), device=device)

        rng = np.random.default_rng(a.seed)
        idx = np.sort(rng.choice(len(yte), a.n_samples, replace=False))
        clf = build_tabpfn_v2(device, a.model_seed)

        rows, t0 = [], time.time()
        print(f"\n=== {ds}  context {tuple(Xc.shape)}  querying {a.n_samples} test rows")
        print(f"  {'test_row':>8s} {'y':>2s} | {'clean p(true)':>13s} {'pred':>4s} | "
              f"{'flipped p(true)':>15s} {'pred':>4s} {'attack':>7s} | {'honest p(true)':>14s}")
        for i in idx:
            x, y = Xt[i], int(yte[i])
            clean = probs_for_query(clf, Xc, yc, x, y, evaluate_context)
            out = {"test_index": int(i), "y_true": y, "clean_p_true": float(clean[y]),
                   "clean_pred": int(clean.argmax())}
            for name, lab in (("flipped", 1 - y), ("honest", y)):
                Xp = torch.cat([Xc, x[None, :]])
                yp = torch.cat([yc, torch.tensor([float(lab)], device=device)])
                p = probs_for_query(clf, Xp, yp, x, y, evaluate_context)
                out[f"{name}_p_true"] = float(p[y])
                out[f"{name}_pred"] = int(p.argmax())
            out["attack_succeeded"] = bool(out["clean_pred"] == y and out["flipped_pred"] != y)
            rows.append(out)
            mark = "FLIP" if out["flipped_pred"] != y else "-"
            print(f"  {i:>8d} {y:>2d} | {out['clean_p_true']:>13.4f} {out['clean_pred']:>4d} | "
                  f"{out['flipped_p_true']:>15.4f} {out['flipped_pred']:>4d} {mark:>7s} | "
                  f"{out['honest_p_true']:>14.4f}")

        correct = [r for r in rows if r["clean_pred"] == r["y_true"]]
        flipped = [r for r in rows if r["flipped_pred"] != r["y_true"]]
        succ = [r for r in rows if r["attack_succeeded"]]
        drop = float(np.mean([r["clean_p_true"] - r["flipped_p_true"] for r in rows]))
        gain = float(np.mean([r["honest_p_true"] - r["clean_p_true"] for r in rows]))
        print(f"  -> clean correct {len(correct)}/{len(rows)}   wrong after flip {len(flipped)}/{len(rows)}"
              f"   attack succeeded {len(succ)}/{len(correct) if correct else 0}"
              f"   mean p(true) drop {drop:+.4f}   honest-control gain {gain:+.4f}   [{time.time()-t0:.0f}s]")
        report["datasets"][ds] = {
            "n_context": int(Xc.shape[0]), "n_features": len(cols),
            "clean_correct": len(correct), "wrong_after_flip": len(flipped),
            "attack_succeeded": len(succ), "mean_p_true_drop": drop,
            "mean_honest_gain": gain, "samples": rows,
        }
        del clf
        torch.cuda.empty_cache()

    write_json(a.out / "summary.json", report)
    print(f"\nwrote {a.out / 'summary.json'}")
    return report


if __name__ == "__main__":
    main()

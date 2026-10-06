#!/usr/bin/env python
"""Sift a poisoned context by learning a soft train/query split (see sift.py).

Puts the whole poisoned set D on both sides of TabPFNv2, learns a weight X[i] per row by
ascending the held-out loss, then keeps the (1-R)% highest-weight rows and reports how
much poison ended up in what was kept.

The extraction is done under every ``--strategies`` rule in one go, since they share the
same optimisation and differ only in how the per-class quota is set.

Nothing in the optimisation sees which rows were flipped; the flip set is used only to
score the result afterwards.

    python scripts/sift_context.py --gpu 0 --dataset url_unique --attack hub-flip \
        --row-percent 20 --lr 0.5 --n-iter 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits, poison_rows  # noqa: E402
from tabfm_experiments.config import Seeds  # noqa: E402
from tabfm_experiments.io import setup_logger, write_json  # noqa: E402
from tabfm_experiments.sampling import k_from_percent  # noqa: E402
from tabfm_experiments.sift import (  # noqa: E402
    RowWeights, optimise, row_attention_modules, select_clean_subset, subset_report,
)

STRATEGIES = ["proportional", "balanced", "global"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", required=True, choices=["label-flip-random", "hub-flip"])
    ap.add_argument("--row-percent", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=256, help="query rows per forward/backward")
    ap.add_argument("--layers", default="all", help="'all', 'first', or comma-separated indices")
    ap.add_argument("--init-logit", type=float, default=4.0,
                    help="X starts at sigmoid(this): 4.0 -> 0.982, i.e. everything in the context")
    ap.add_argument("--mass-penalty", type=float, default=0.0)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--sizes", type=int, nargs="+", default=[250],
                    help="extra subset sizes; the (1-R) full budget is always included")
    ap.add_argument("--extract-iter", type=int, default=-1,
                    help="which saved X to extract from (-1 = final). Every iteration is "
                         "kept in history.npz either way")
    ap.add_argument("--no-downstream", action="store_true",
                    help="skip scoring the kept subsets with TabPFNv2")
    ap.add_argument("--n-pure-draws", type=int, default=5)
    ap.add_argument("--poison-draw", type=int, default=0)
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed)
    ap.add_argument("--hub-metric", default="knn")
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--model-seed", type=int, default=Seeds.model_seed)
    ap.add_argument("--out", type=Path, default=Path("results/sift"))
    ap.add_argument("--tag", default=None, help="subdirectory name (default: built from the args)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    layers = a.layers if a.layers in ("all", "first") else [int(v) for v in a.layers.split(",")]
    tag = a.tag or f"{a.attack}_r{int(round(a.row_percent)):03d}_lr{a.lr:g}_it{a.n_iter}"
    out = a.out / a.dataset / tag
    if (out / "summary.json").exists() and not a.force:
        print(f"[skip] {out}/summary.json exists")
        return None
    out.mkdir(parents=True, exist_ok=True)
    log_ = setup_logger(out, "sift")
    log = log_.info

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import (
        build_tabpfn_v2, device_from_gpu, evaluate_context,
    )

    train, test = default_splits(a.dataset)
    device = device_from_gpu(a.gpu)
    Xdf, y_clean, cols = load_split(train)
    X_ctx = torch.tensor(Xdf.to_numpy(np.float32), device=device)
    y_clean = np.asarray(y_clean).astype(int)
    n = len(y_clean)
    k = k_from_percent(n, a.row_percent)
    keep_n = n - k

    flipped = poison_rows(a.attack, X_ctx, y_clean, k, run_id=a.poison_draw,
                          row_seed=a.row_seed, hub_metric=a.hub_metric, hub_k=a.hub_k)
    y_given = y_clean.copy()
    y_given[flipped] = 1 - y_given[flipped]
    y_ctx = torch.as_tensor(y_given.astype(np.float32), device=device)

    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%  context {n}x{len(cols)}  "
        f"k={k} flipped, keeping {keep_n}")
    log(f"  lr={a.lr:g}  iters={a.n_iter}  chunk={a.chunk}  layers={a.layers}  "
        f"init_logit={a.init_logit}  mass_penalty={a.mass_penalty:g}")
    log(f"  given class counts {np.bincount(y_given, minlength=2).tolist()}")

    clf = build_tabpfn_v2(device, a.model_seed, recompute_layers=True)
    with torch.no_grad():
        clf.fit_with_differentiable_input(X_ctx, y_ctx)     # builds clf.models_
    modules = row_attention_modules(clf, layers)
    weights = RowWeights(n, device, init_logit=a.init_logit)

    t0 = time.time()
    trace, x_history, ce_history = optimise(
        clf, X_ctx, y_ctx, weights, modules, y_np=y_given, n_iter=a.n_iter, lr=a.lr,
        chunk=a.chunk, keep_frac=keep_n / n, mass_penalty=a.mass_penalty,
        flipped=flipped, strategies=tuple(a.strategies), log=log)
    secs = time.time() - t0

    x = x_history[a.extract_iter]
    sizes = sorted({m for m in a.sizes if m < keep_n} | {keep_n})
    results = {"extract_iter": int(a.extract_iter), "sizes": [int(m) for m in sizes],
               "full_budget": int(keep_n)}
    log(f"\n  extraction from iteration {a.extract_iter}, sizes {sizes} "
        f"(full budget = n - k = {keep_n}):")
    for strat in a.strategies:
        for m in sizes:
            kept = select_clean_subset(x, y_given, m, strat)
            rep = subset_report(kept, flipped, n)
            # At the full budget the removed set is the same size as the true flip set,
            # so precision == recall there; at smaller m they come apart.
            rep["removal_precision"] = rep["poison_removed"] / max(n - rep["n_kept"], 1)
            rep["NCR"] = (1 - rep["purity"]) / (k / n) * 100
            rep["kept_class_counts"] = np.bincount(y_given[kept], minlength=2).tolist()
            results[f"{strat}_{m}"] = rep
            if strat == "balanced":
                log(f"    {strat:<13s} m={m:<5d} purity {rep['purity']:.4f}  "
                    f"poison kept {rep['poison_in_kept']:<5d} NCR {rep['NCR']:.1f}%  "
                    f"removed {rep['poison_removed']}/{rep['poison_total']} "
                    f"(precision {rep['removal_precision']:.4f})")

    # ---- downstream: TabPFNv2 on the held-out test set with the kept subset ----------
    if not a.no_downstream:
        from tabfm_experiments.defense import METRICS, score_on_test
        Xte_df, y_te, _ = load_split(test)
        X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=device)
        y_te = np.asarray(y_te).astype(int)
        clean_rows = np.setdiff1d(np.arange(n), flipped)

        def tab_score(rows, labels):
            rows = np.asarray(rows)
            i = torch.as_tensor(rows, dtype=torch.long, device=device)
            yc = torch.as_tensor(labels[rows].astype(np.float32), device=device)
            return score_on_test(clf, X_ctx[i], yc, X_test, y_te, evaluate_context)

        down = {"clean": tab_score(np.arange(n), y_clean),
                "poisoned": tab_score(np.arange(n), y_given),
                "oracle": tab_score(clean_rows, y_given)}
        log(f"\n  TabPFNv2 on the held-out test set ({len(y_te)} rows):")
        for kk in ("clean", "poisoned", "oracle"):
            log(f"    {kk:<22s} " + "  ".join(f"{m} {down[kk][m]:.4f}" for m in METRICS[:5]))
        rng = np.random.default_rng(0)
        for m in sizes:
            for strat in ("balanced", "global"):
                kept = select_clean_subset(x, y_given, m, strat)
                down[f"{strat}_{m}"] = tab_score(kept, y_given)
            acc = [tab_score(rng.choice(clean_rows, min(m, len(clean_rows)), replace=False),
                             y_given) for _ in range(a.n_pure_draws)]
            down[f"pure_{m}"] = {kk: float(np.mean([r[kk] for r in acc])) for kk in acc[0]}
            log(f"    m={m:<5d} balanced " + "  ".join(
                f"{mm} {down[f'balanced_{m}'][mm]:.4f}" for mm in METRICS[:5]))
            log(f"    {'':<10s} pure-m   " + "  ".join(
                f"{mm} {down[f'pure_{m}'][mm]:.4f}" for mm in METRICS[:5]))
        results["downstream"] = down

    # The whole trajectory, not just the endpoint: x_history[t] is the state the loss
    # and ce_history[t] were measured at, so a later analysis can tell a run that
    # improved monotonically from one that peaked early and then decayed, and can see
    # exactly which rows moved when.
    np.savez_compressed(
        out / "history.npz",
        x_history=x_history,                  # [n_iter + 1, n]
        ce_history=ce_history,                # [n_iter, n]
        loss=np.array([r["loss"] for r in trace], dtype=np.float32),
        x=x.astype(np.float32),
        logit=weights.logit.detach().float().cpu().numpy(),
        flipped_indices=np.asarray(flipped),
        y_clean=y_clean.astype(np.int8), y_given=y_given.astype(np.int8))
    write_json(out / "summary.json", {
        "dataset": a.dataset, "attack": a.attack, "row_percent": a.row_percent,
        "k_flipped": int(k), "n_context": n, "n_features": len(cols), "n_kept": keep_n,
        "lr": a.lr, "n_iter": a.n_iter, "chunk": a.chunk, "layers": a.layers,
        "init_logit": a.init_logit, "mass_penalty": a.mass_penalty,
        "poison_draw": a.poison_draw, "row_seed": a.row_seed, "model_seed": a.model_seed,
        "class_counts_clean": np.bincount(y_clean, minlength=2).tolist(),
        "class_counts_given": np.bincount(y_given, minlength=2).tolist(),
        "seconds": secs, "trace": trace, "extraction": results,
        "sizes": [int(m) for m in sizes], "full_budget": int(keep_n),
    })
    log(f"\n  [{secs:.0f}s]  wrote {out / 'summary.json'}")
    return results


if __name__ == "__main__":
    main()

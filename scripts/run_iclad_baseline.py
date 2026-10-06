#!/usr/bin/env python
"""ICLAD (Wei & Armanfard) as an unsupervised anomaly-detection baseline for sifting.

ICLAD is an in-context anomaly detector: it conditions on a context table and scores
query rows by p(anomalous | x, context), with no weight updates. Its UNSUPERVISED regime
is the one that matches our problem -- labels absent, context contaminated at an unknown
rate -- so the pretrained ``iclad_mixedprior_unsup`` checkpoint is used as-is.

MAPPING ONTO LABEL POISONING
    ICLAD has no separate target: a row is just a feature vector. The poisoned label is
    therefore appended AS A FEATURE, so each row is [x_1..x_d, y_given]. A label flip
    then shows up as a point that is atypical in the JOINT feature-label space, which is
    what an anomaly detector can see. The contaminated set is passed as both the context
    and the query set, which is exactly ICLAD's unsupervised setting.

    ``predict_score`` returns p(anomalous), so the rows most likely to be CLEAN are the
    LOWEST-scoring ones. This script reports normality = 1 - anomaly so that, as with
    every other method here, higher means "more likely clean".

CONTROL
    The attacks in this project change labels only -- features are untouched. So the
    same run is repeated on the features ALONE, without the label column. That variant
    should carry no signal at all, and it is the check that the label column is what
    ICLAD is reading.

DEFAULTS
    Everything is left at the library's defaults. Worth noting that at our sizes no
    subsampling happens at all: the model takes up to 512 features (we have at most 86,
    including the appended label) and max_train_len is 10000 (our contexts are <= 5000),
    so every ensemble member sees every row and every column.

    python scripts/run_iclad_baseline.py --gpu 0 --dataset url_unique \
        --attack hub-flip --row-percent 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "iclad" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits, poison_rows  # noqa: E402
from tabfm_experiments.config import Seeds  # noqa: E402
from tabfm_experiments.io import setup_logger, write_json  # noqa: E402
from tabfm_experiments.sampling import k_from_percent  # noqa: E402
from tabfm_experiments.sift import select_clean_subset, subset_report  # noqa: E402

DEFAULT_CKPT = REPO / "iclad" / "iclad_mixedprior_unsup.pth"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", required=True, choices=["hub-flip", "label-flip-random"])
    ap.add_argument("--row-percent", type=float, default=30.0)
    ap.add_argument("--poison-draw", type=int, default=0)
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed)
    ap.add_argument("--hub-metric", default="knn")
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[250],
                    help="fixed subset sizes; the (1-R) full budget is always added")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--iclad-seed", type=int, default=0)
    ap.add_argument("--no-downstream", action="store_true")
    ap.add_argument("--n-pure-draws", type=int, default=5)
    ap.add_argument("--out", type=Path, default=REPO / "results/iclad")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args(argv)

    from sklearn.metrics import roc_auc_score

    from iclad import ICLAD
    from tabfm_experiments.data import load_split

    out = a.out / a.dataset / f"{a.attack}_r{int(round(a.row_percent)):03d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, "iclad").info
    dev = torch.device(f"cuda:{a.gpu}")

    train, test = default_splits(a.dataset)
    Xdf, y_clean, cols = load_split(train)
    Xraw = Xdf.to_numpy(np.float32)
    y_clean = np.asarray(y_clean).astype(int)
    n, n_feat = Xraw.shape

    k = k_from_percent(n, a.row_percent)
    flipped = poison_rows(a.attack, torch.tensor(Xraw, device=dev), y_clean, k,
                          run_id=a.poison_draw, row_seed=a.row_seed,
                          hub_metric=a.hub_metric, hub_k=a.hub_k)
    y_given = y_clean.copy()
    y_given[flipped] = 1 - y_given[flipped]
    is_flipped = np.zeros(n, dtype=int)
    is_flipped[flipped] = 1

    full_budget = n - k
    sizes = sorted({m for m in a.sizes if m < full_budget} | {full_budget})
    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%  context {n}x{n_feat}, "
        f"{k} flipped, full budget {full_budget}, sizes {sizes}")

    model = ICLAD(checkpoint_path=str(a.checkpoint), model_name="ICLAD_UNSUP",
                  device=dev, seed=a.iclad_seed, verbose=False)
    log(f"  checkpoint {a.checkpoint.name}; model takes up to "
        f"{model.model.config['max_num_features']} features, max_train_len "
        f"{model.max_train_len}, ensemble {model.ensemble_size} "
        f"-> no feature or row subsampling at this size")

    variants = {
        # The poisoned label appended as an extra column: the setting of interest.
        "with_label": np.concatenate([Xraw, y_given[:, None].astype(np.float32)], axis=1),
        # Control: features only. The attacks never touch features, so this should show
        # no ability to find the flipped rows.
        "features_only": Xraw,
    }

    scores, results = {}, {}
    for name, X in variants.items():
        model.fit(X)                       # y_train=None -> unsupervised regime
        anomaly = np.asarray(model.predict_score(X), dtype=np.float64)
        normality = 1.0 - anomaly          # higher = more likely clean, as elsewhere
        scores[name] = normality
        auc = float(roc_auc_score(is_flipped, anomaly))
        results[name] = {"detect_auc_anomaly_means_flipped": auc,
                         "anomaly_mean_flipped": float(anomaly[is_flipped == 1].mean()),
                         "anomaly_mean_clean": float(anomaly[is_flipped == 0].mean())}
        log(f"\n  [{name}] AUC(anomaly score -> flipped) = {auc:.4f}   "
            f"mean score: flipped {anomaly[is_flipped == 1].mean():.4f} vs "
            f"clean {anomaly[is_flipped == 0].mean():.4f}")
        for strat in ("balanced", "proportional", "global"):
            for m in sizes:
                kept = select_clean_subset(normality, y_given, m, strat)
                rep = subset_report(kept, flipped, n)
                rep["NCR"] = (1 - rep["purity"]) / (k / n) * 100
                rep["kept_class_counts"] = np.bincount(y_given[kept], minlength=2).tolist()
                results[name][f"{strat}_{m}"] = rep
                if strat == "balanced":
                    log(f"    {strat:<13s} m={m:<5d} purity {rep['purity']:.4f}  "
                        f"poison kept {rep['poison_in_kept']:<5d} NCR {rep['NCR']:.1f}%")

    # ---- downstream: TabPFNv2 on the held-out test set with the kept subset ----------
    if not a.no_downstream:
        from tabfm_experiments.defense import METRICS, score_on_test
        from tabfm_experiments.runtime import (
            build_tabpfn_v2, device_from_gpu, evaluate_context,
        )
        tdev = device_from_gpu(a.gpu)
        Xte_df, y_te, _ = load_split(test)
        X_all = torch.tensor(Xraw, device=tdev)
        X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=tdev)
        y_te = np.asarray(y_te).astype(int)
        clf = build_tabpfn_v2(tdev, 0)
        clean_rows = np.setdiff1d(np.arange(n), flipped)

        def tab_score(rows, labels):
            rows = np.asarray(rows)
            i = torch.as_tensor(rows, dtype=torch.long, device=tdev)
            yc = torch.as_tensor(labels[rows].astype(np.float32), device=tdev)
            return score_on_test(clf, X_all[i], yc, X_test, y_te, evaluate_context)

        down = {"clean": tab_score(np.arange(n), y_clean),
                "poisoned": tab_score(np.arange(n), y_given),
                "oracle": tab_score(clean_rows, y_given)}
        log(f"\n  TabPFNv2 on the held-out test set ({len(y_te)} rows):")
        for kk in ("clean", "poisoned", "oracle"):
            log(f"    {kk:<22s} " + "  ".join(f"{m} {down[kk][m]:.4f}" for m in METRICS[:5]))
        rng = np.random.default_rng(0)
        for m in sizes:
            for strat in ("balanced", "global"):
                kept = select_clean_subset(scores["with_label"], y_given, m, strat)
                down[f"{strat}_{m}"] = tab_score(kept, y_given)
            acc = [tab_score(rng.choice(clean_rows, min(m, len(clean_rows)), replace=False),
                             y_given) for _ in range(a.n_pure_draws)]
            down[f"pure_{m}"] = {kk: float(np.mean([r[kk] for r in acc])) for kk in acc[0]}
            log(f"    m={m:<5d} balanced " + "  ".join(
                f"{mm} {down[f'balanced_{m}'][mm]:.4f}" for mm in METRICS[:5]))
            log(f"    {'':<10s} pure-m   " + "  ".join(
                f"{mm} {down[f'pure_{m}'][mm]:.4f}" for mm in METRICS[:5]))
        results["downstream"] = down

    np.savez_compressed(out / "scores.npz",
                        normality_with_label=scores["with_label"].astype(np.float32),
                        normality_features_only=scores["features_only"].astype(np.float32),
                        flipped_indices=np.asarray(flipped),
                        y_clean=y_clean.astype(np.int8), y_given=y_given.astype(np.int8))
    write_json(out / "summary.json", {
        "method": "iclad", "variant_checkpoint": a.checkpoint.name,
        "setting": "unsupervised (labels absent, contaminated context)",
        "dataset": a.dataset, "attack": a.attack, "row_percent": a.row_percent,
        "k_flipped": int(k), "n_context": n, "n_features": n_feat,
        "full_budget": int(full_budget), "sizes": [int(m) for m in sizes],
        "poison_draw": a.poison_draw, "iclad_seed": a.iclad_seed,
        "ensemble_size": int(model.ensemble_size),
        "max_train_len": int(model.max_train_len),
        "max_n_features": int(model.max_n_features),
        "class_counts_given": np.bincount(y_given, minlength=2).tolist(),
        "results": results,
    })
    log(f"\n  wrote {out / 'summary.json'}")
    return results


if __name__ == "__main__":
    main()

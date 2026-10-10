#!/usr/bin/env python
"""Robustness of TabPFNv2 to label perturbation on K-class datasets.

One (dataset, attack, R) condition per call:

    label-flip-bernoulli  each row, with probability R, relabelled to a uniformly random
                          other class (redrawn every run)
    hub-flip-local        the R% hubs, each relabelled to its locally closest other class
                          (deterministic -- one run)
    hub-flip-random       the same hub rows, relabelled to uniformly random other classes
                          (targets redrawn every run)

See tabfm_experiments/multiclass_attacks.py for the definitions and why "closest" is
local. Only the context is ever perturbed; the attack test set is used purely to score.
The validation set is not used.

Every run is scored on the held-out attack test set with cross-entropy, accuracy,
balanced accuracy, multiclass MCC and one-vs-rest macro-averaged ROC-AUC, then averaged
across runs. Each trial's test-set probabilities are saved next to its poisoned labels,
so verify_multiclass.py can re-derive every number independently.

    python scripts/attack_multiclass.py --gpu 1 --dataset gas-drift \
        --attack hub-flip-local --row-percent 20
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

from tabfm_experiments.config import DATASETS_DIR, Seeds  # noqa: E402
from tabfm_experiments.io import setup_logger, write_json  # noqa: E402
from tabfm_experiments.metrics import MULTICLASS_METRICS, multiclass_metrics  # noqa: E402
from tabfm_experiments.multiclass_attacks import (  # noqa: E402
    ATTACKS, STOCHASTIC, poison, transition_counts,
)


def default_splits(dataset: str) -> tuple[Path, Path, int]:
    """(context, attack test, n_classes) from the dataset's split_info.json."""
    d = DATASETS_DIR / dataset / "splits"
    info = json.loads((d / "split_info.json").read_text())
    names = list(info["splits"])
    ctx = [n for n in names if n.startswith("context_")]
    atk = [n for n in names if n.startswith("test_attack_")]
    if len(ctx) != 1 or len(atk) != 1:
        raise SystemExit(f"expected one context_*.csv and one test_attack_*.csv in {d}, "
                         f"found {ctx} and {atk}; pass --train / --test")
    return d / ctx[0], d / atk[0], int(info["n_classes"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", required=True, choices=ATTACKS)
    ap.add_argument("--row-percent", type=float, required=True)
    ap.add_argument("--n-runs", type=int, default=None,
                    help="default 10 for the stochastic attacks, 1 for hub-flip-local")
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--n-classes", type=int, default=None, help="default: from split_info.json")
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed)
    ap.add_argument("--model-seed", type=int, default=Seeds.model_seed)
    ap.add_argument("--hub-metric", default="knn", choices=["knn", "all"])
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--target-k", type=int, default=10,
                    help="nearest members per class used to define the locally closest class")
    ap.add_argument("--test-batch-size", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/multiclass"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.n_runs is None:
        a.n_runs = 10 if a.attack in STOCHASTIC else 1
    if a.attack == "hub-flip-local" and a.n_runs != 1:
        raise SystemExit("hub-flip-local is deterministic: extra runs would be identical; use --n-runs 1")

    out = a.out / a.dataset / f"{a.attack}_r{int(round(a.row_percent)):03d}"
    if (out / "summary.json").exists() and not a.force:
        print(f"[skip] {out}/summary.json exists")
        return None
    (out / "trials").mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, "attack_multiclass").info

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import (
        build_tabpfn_v2, check_all_classes_present, device_from_gpu, evaluate_context,
    )

    ctx_path, test_path, k_info = default_splits(a.dataset)
    ctx_path, test_path = a.train or ctx_path, a.test or test_path
    K = a.n_classes or k_info
    device = device_from_gpu(a.gpu)

    Xc_df, yc, cols = load_split(ctx_path)
    Xt_df, yt, _ = load_split(test_path)
    yc, yt = np.asarray(yc).astype(int), np.asarray(yt).astype(int)
    Xc = torch.tensor(Xc_df.to_numpy(np.float32), device=device)
    Xt = torch.tensor(Xt_df.to_numpy(np.float32), device=device)
    yt_t = torch.as_tensor(yt, dtype=torch.long, device=device)
    check_all_classes_present(yc, K)
    check_all_classes_present(yt, K)
    n = len(yc)

    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%  context {n}x{len(cols)}, "
        f"test {len(yt)}, K={K} classes, {a.n_runs} run(s)")
    log(f"  context {ctx_path}\n  test    {test_path}")

    clf = build_tabpfn_v2(device, a.model_seed, n_classes=K)

    def score(y_ctx: np.ndarray) -> tuple[dict, np.ndarray]:
        check_all_classes_present(y_ctx, K)
        yct = torch.as_tensor(y_ctx.astype(np.float32), device=device)
        ce, _, probs = evaluate_context(clf, Xc, yct, Xt, yt_t, need_grad=False,
                                        test_batch_size=a.test_batch_size)
        p = probs.detach().float().cpu().numpy()
        m = multiclass_metrics(p, yt, K)
        # The runtime's own CE and the metric's must agree: a second, independent path.
        if abs(float(ce) - m["ce"]) > 1e-4:
            raise AssertionError(f"runtime CE {float(ce):.6f} != metric CE {m['ce']:.6f}")
        return m, p

    t0 = time.time()
    clean, p_clean = score(yc)
    log("  clean:    " + "  ".join(f"{k} {clean[k]:.4f}" for k in MULTICLASS_METRICS)
        + f"   [{time.time() - t0:.0f}s]")
    np.savez_compressed(out / "trials" / "clean.npz", probs=p_clean.astype(np.float32),
                        y_test=yt.astype(np.int16), y_context=yc.astype(np.int16))

    runs = []
    for run in range(a.n_runs):
        t_run = time.time()
        rows, y_pois = poison(a.attack, Xc, yc, K, a.row_percent, run_id=run, row_seed=a.row_seed,
                              hub_metric=a.hub_metric, hub_k=a.hub_k, target_k=a.target_k)
        m, p = score(y_pois)
        runs.append({"run_id": run, "n_flipped": int(len(rows)),
                     "transitions": transition_counts(yc, y_pois, rows, K),
                     "metrics": m,
                     "delta": {k: m[k] - clean[k] for k in MULTICLASS_METRICS}})
        np.savez_compressed(out / "trials" / f"run{run}.npz",
                            flipped_indices=rows, y_clean=yc.astype(np.int16),
                            y_poisoned=y_pois.astype(np.int16),
                            probs=p.astype(np.float32), y_test=yt.astype(np.int16))
        log(f"  run {run}: flipped {len(rows):>5d}   "
            + "  ".join(f"{k} {m[k]:.4f}" for k in MULTICLASS_METRICS)
            + f"   [{time.time() - t_run:.0f}s]")

    def agg(key: str) -> dict:
        out_ = {}
        for k in MULTICLASS_METRICS:
            v = np.array([r[key][k] for r in runs], dtype=np.float64)
            out_[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        return out_

    summary = {
        "dataset": a.dataset, "attack": a.attack, "row_percent": a.row_percent,
        "n_classes": K, "n_context": n, "n_features": len(cols), "n_test": len(yt),
        "context": str(ctx_path), "test": str(test_path), "n_runs": a.n_runs,
        "stochastic": a.attack in STOCHASTIC,
        "row_sampling": ("independent per-row draw with probability R" if a.attack == "label-flip-bernoulli"
                         else "the R% hubs (fixed-size, deterministic)"),
        "target_rule": {"label-flip-bernoulli": "uniformly random other class",
                        "hub-flip-local": f"locally closest other class ({a.target_k} nearest members)",
                        "hub-flip-random": "uniformly random other class"}[a.attack],
        "seeds": {"row_seed": a.row_seed, "model_seed": a.model_seed},
        "hub": {"metric": a.hub_metric, "k": a.hub_k},
        "class_counts_context": np.bincount(yc, minlength=K).tolist(),
        "class_counts_test": np.bincount(yt, minlength=K).tolist(),
        "aggregation": f"mean +- std over {a.n_runs} run(s), no selection",
        "metrics": list(MULTICLASS_METRICS),
        "clean": clean,
        "aggregate_poisoned": agg("metrics"),
        "aggregate_delta": agg("delta"),
        "n_flipped": {"mean": float(np.mean([r["n_flipped"] for r in runs])),
                      "per_run": [r["n_flipped"] for r in runs]},
        "runs": runs,
    }
    write_json(out / "summary.json", summary)
    ag = summary["aggregate_poisoned"]
    log("  mean:     " + "  ".join(f"{k} {ag[k]['mean']:.4f}+-{ag[k]['std']:.4f}"
                                   for k in MULTICLASS_METRICS))
    log(f"  wrote {out / 'summary.json'}   [{time.time() - t0:.0f}s total]")
    return summary


if __name__ == "__main__":
    main()

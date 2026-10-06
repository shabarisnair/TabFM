#!/usr/bin/env python
"""Why does the k-fold defence raise cross-entropy while accuracy/AUC can improve?

CE is the only reported metric that reads the probability VALUE; accuracy and ROC-AUC
read only the argmax and the ranking. So the three can move apart, and the question is
which part of the probability changed.

This rebuilds the four contexts for one cell from its saved ``p<p>_f<f>_scores.npz``
(no 50-fold detection is redone -- the flagged set is already stored), scores the real
test set, and decomposes:

    CE = (n_right/n) * mean CE on rows it got RIGHT
       + (n_wrong/n) * mean CE on rows it got WRONG

If the defence raises CE while accuracy improves, the second term must be carrying it:
fewer mistakes, each far more confident. Confidence, Brier and ECE are reported
alongside to say whether the shift is calibration (probabilities too extreme) or
discrimination (the ranking itself got worse).

    python scripts/diagnose_defense_ce.py --dataset heloc --attack hub-flip --row-percent 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defend_context import default_splits  # noqa: E402
from tabfm_experiments.metrics import binary_metrics, with_derived  # noqa: E402


def ece(p_pos: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the predicted-class confidence."""
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = ((p_pos >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def describe(name: str, probs: np.ndarray, y: np.ndarray) -> dict:
    p_true = probs[np.arange(len(y)), y]
    pred = probs.argmax(1)
    right = pred == y
    conf = probs.max(1)
    nll = -np.log(np.clip(p_true, 1e-12, None))
    m = with_derived(binary_metrics(probs, y))
    return {
        "name": name, "ce": m["ce"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
        "n_right": int(right.sum()), "n_wrong": int((~right).sum()),
        "ce_on_right": float(nll[right].mean()) if right.any() else float("nan"),
        "ce_on_wrong": float(nll[~right].mean()) if (~right).any() else float("nan"),
        "share_from_wrong": float(nll[~right].sum() / nll.sum()) if nll.sum() else float("nan"),
        "mean_confidence": float(conf.mean()),
        "conf_when_wrong": float(conf[~right].mean()) if (~right).any() else float("nan"),
        "n_conf_gt_99_wrong": int(((conf > 0.99) & ~right).sum()),
        "n_conf_gt_90_wrong": int(((conf > 0.90) & ~right).sum()),
        "brier": float(np.mean((probs[:, 1] - y) ** 2)),
        "ece": ece(probs[:, 1], y),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--row-percent", type=float, required=True)
    ap.add_argument("--poison-draw", type=int, default=0)
    ap.add_argument("--fold-draw", type=int, default=0)
    ap.add_argument("--root", type=Path, default=Path("results/defense_kfold"))
    ap.add_argument("--model-seed", type=int, default=0)
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    cell = (a.root / a.dataset / f"{a.attack}_r{int(round(a.row_percent)):03d}"
            / f"p{a.poison_draw}_f{a.fold_draw}_scores.npz")
    if not cell.exists():
        raise SystemExit(f"no such cell: {cell}")
    z = np.load(cell)
    y_clean = z["y_clean"].astype(int)
    y_given = z["y_given"].astype(int)
    flipped, flagged = z["flipped_indices"], z["flagged_indices"]

    train, test = default_splits(a.dataset)
    device = device_from_gpu(a.gpu)
    Xtr_df, y_tr, _ = load_split(train)
    Xte_df, y_te, _ = load_split(test)
    assert np.array_equal(np.asarray(y_tr).astype(int), y_clean), "context labels moved"
    X_ctx = torch.tensor(Xtr_df.to_numpy(np.float32), device=device)
    X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=device)
    y_te = np.asarray(y_te).astype(int)
    yt = torch.as_tensor(y_te, dtype=torch.long, device=device)
    n = len(y_clean)

    clf = build_tabpfn_v2(device, a.model_seed)

    def probs_for(rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
        idx = torch.as_tensor(rows, dtype=torch.long, device=device)
        yc = torch.as_tensor(labels[rows].astype(np.float32), device=device)
        _, _, p = evaluate_context(clf, X_ctx[idx], yc, X_test, yt, need_grad=False)
        return p.detach().float().cpu().numpy().astype(np.float64)

    keep = lambda drop: np.setdiff1d(np.arange(n), drop)  # noqa: E731
    variants = [
        ("clean", np.arange(n), y_clean),
        ("poisoned", np.arange(n), y_given),
        ("defended", keep(flagged), y_given),
        ("oracle", keep(flipped), y_given),
    ]
    rows = [describe(name, probs_for(r, lab), y_te) for name, r, lab in variants]

    print(f"\n{a.dataset} / {a.attack} R={a.row_percent:g}%  "
          f"cell p{a.poison_draw}f{a.fold_draw}   context {n} rows, "
          f"{len(flipped)} flipped, {len(flagged)} removed, "
          f"{len(np.intersect1d(flipped, flagged))} of the removals were真 poison"
          .replace("真 ", "real "))
    hdr = (f"  {'context':<10s} {'CE':>7s} {'acc':>7s} {'AUC':>7s} | {'#wrong':>7s} "
           f"{'CE|right':>9s} {'CE|wrong':>9s} {'%CE from wrong':>15s} | "
           f"{'conf':>6s} {'conf|wrong':>11s} {'>.99 wrong':>11s} {'Brier':>7s} {'ECE':>6s}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['name']:<10s} {r['ce']:>7.4f} {r['accuracy']:>7.4f} {r['roc_auc']:>7.4f} | "
              f"{r['n_wrong']:>7d} {r['ce_on_right']:>9.4f} {r['ce_on_wrong']:>9.4f} "
              f"{100 * r['share_from_wrong']:>14.1f}% | {r['mean_confidence']:>6.3f} "
              f"{r['conf_when_wrong']:>11.3f} {r['n_conf_gt_99_wrong']:>11d} "
              f"{r['brier']:>7.4f} {r['ece']:>6.3f}")

    p, d = rows[1], rows[2]
    print(f"\n  defended vs poisoned:  CE {d['ce'] - p['ce']:+.4f}   "
          f"accuracy {d['accuracy'] - p['accuracy']:+.4f}   "
          f"mistakes {d['n_wrong'] - p['n_wrong']:+d}   "
          f"confidence-when-wrong {d['conf_when_wrong'] - p['conf_when_wrong']:+.3f}   "
          f"ECE {d['ece'] - p['ece']:+.3f}")
    return rows


if __name__ == "__main__":
    main()

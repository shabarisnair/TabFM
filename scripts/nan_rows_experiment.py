#!/usr/bin/env python
"""Does TabPFNv2 ignore context rows that carry no information?

Appends n all-NaN rows (every feature AND the label NaN) to a fixed context, refits, and
scores on the attack test set, for n = 0, 1, 2, 4, ..., up to n equal to the base context
size. Ideally nothing changes: the rows say nothing.

Base context: a class-stratified random subset of the dataset's context split (default
gas-drift, 2000 rows of context_5000, seed 0). Deliberately not a contiguous slice:
gas-drift's rows are ordered by sensor-drift batch, so context_5000[:2000] is a different,
harder distribution (clean accuracy 0.743 vs 0.994 for the random subset).

Four variants, one curve each:

  pipeline              our differentiable path -- the one every attack experiment uses
                        (build_tabpfn_v2 + fit_with_differentiable_input) -- with the rows
                        exactly as specified: NaN features and NaN label.
  pipeline-nanaware     the same, but TabPFN's DifferentiableZNormStep computes its column
                        mean / std over the non-NaN cells only. Upstream it uses X.mean and
                        X.std, so a single NaN in a column makes that column's statistics
                        NaN, and (X - mean) / std then turns the WHOLE column NaN in every
                        context and test row. This variant separates that preprocessing bug
                        from what the model itself does with the rows.
  pipeline-nanaware-labelled
                        NaN features but a real label, assigned in the base context's class
                        proportions. Differs from pipeline-nanaware only in the label, so it
                        isolates what the NaN label does.
  standard-api-labelled TabPFNClassifier.fit / predict_proba with the same fixed inference
                        config. It rejects NaN labels outright ("Input y contains NaN"), so
                        NaN features with labels in the base proportions is the closest it
                        allows.

Every point is one deterministic fit of the same model (one estimator, seed 0), so the
resolution is one test row (0.001 accuracy on 1000 rows).

    python scripts/nan_rows_experiment.py --gpu 1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prepare_multiclass_splits import stratified_exact  # noqa: E402
from tabfm_experiments.config import inference_config  # noqa: E402
from tabfm_experiments.io import write_json  # noqa: E402
from tabfm_experiments.metrics import MULTICLASS_METRICS, multiclass_metrics  # noqa: E402

# name: (path, NaN-aware z-norm, label of the added rows)
VARIANTS = {
    "pipeline": ("differentiable", False, "nan"),
    "pipeline-nanaware": ("differentiable", True, "nan"),
    "pipeline-nanaware-labelled": ("differentiable", True, "proportional"),
    "standard-api-labelled": ("standard", None, "proportional"),
}


# ---- NaN-aware z-norm ------------------------------------------------------------------

def _nan_aware_fit(self, X, feature_schema):
    """DifferentiableZNormStep._fit over the non-NaN cells of each column.

    Equal to the upstream X.mean / X.std (unbiased) on a column without NaN. NaN cells
    stay NaN through (X - mean) / std, which the model then imputes and flags itself --
    exactly what it would see if no other row had a NaN.
    """
    valid = ~torch.isnan(X)
    n = valid.sum(0, keepdim=True)
    mean = torch.where(valid, X, 0.0).sum(0, keepdim=True) / n.clamp_min(1)
    sq = torch.where(valid, X - mean, 0.0).pow(2).sum(0, keepdim=True)
    self.means = mean
    self.stds = (sq / (n - 1).clamp_min(1)).sqrt()
    return feature_schema


@contextlib.contextmanager
def nan_aware_znorm(on: bool):
    from tabpfn.preprocessing.steps.differentiable_z_norm_step import DifferentiableZNormStep

    if not on:
        yield
        return
    upstream = DifferentiableZNormStep._fit
    DifferentiableZNormStep._fit = _nan_aware_fit
    try:
        yield
    finally:
        DifferentiableZNormStep._fit = upstream


# ---- the added rows --------------------------------------------------------------------

def proportional_labels(n: int, counts: np.ndarray) -> np.ndarray:
    """n labels in the given class proportions (largest remainder)."""
    share = n * counts / counts.sum()
    k = np.floor(share).astype(int)
    k[np.argsort(-(share - k), kind="stable")[: n - k.sum()]] += 1
    return np.repeat(np.arange(len(counts)), k)


# ---- scoring ---------------------------------------------------------------------------

def score_differentiable(clf, Xc, yc, Xt, *, device, nan_aware: bool, n_base: int):
    """evaluate_context without its label guard (which rejects NaN labels by design).

    Returns probs and the NaN cell counts of the preprocessed context, base rows first.
    """
    from tabfm_experiments.runtime import _forward, _install_recompute

    with torch.no_grad(), nan_aware_znorm(nan_aware):
        clf.fit_with_differentiable_input(torch.tensor(Xc, device=device),
                                          torch.tensor(yc, device=device))
        Xp = clf.executor_.ensemble_members[0].X_train
        _install_recompute(clf)
        p = _forward(clf, torch.tensor(Xt, device=device), need_grad=False)
    Xp = Xp.detach().cpu().numpy() if torch.is_tensor(Xp) else np.asarray(Xp)
    return (p.float().cpu().numpy(),
            int(np.isnan(Xp[:n_base]).sum()), int(np.isnan(Xp[n_base:]).sum()))


def standard_classifier(device, seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=1, device=device, random_state=seed,
        inference_precision=torch.float32, ignore_pretraining_limits=True,
        n_preprocessing_jobs=1, inference_config=inference_config())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--dataset", default="gas-drift")
    ap.add_argument("--n-base", type=int, default=2000)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--out", type=Path, default=Path("results/nan_rows"))
    a = ap.parse_args(argv)

    from attack_multiclass import default_splits
    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    device = device_from_gpu(a.gpu)
    ctx_path, test_path, K = default_splits(a.dataset)
    Xall_df, yall, _ = load_split(ctx_path)
    Xt_df, yt, _ = load_split(test_path)
    yall, yt = np.asarray(yall).astype(int), np.asarray(yt).astype(int)
    base = stratified_exact(np.arange(len(yall)), yall, a.n_base,
                            np.random.default_rng(a.subset_seed))
    X, y = Xall_df.to_numpy(np.float32)[base], yall[base]
    Xt = Xt_df.to_numpy(np.float32)
    F, counts = X.shape[1], np.bincount(y, minlength=K)
    ns = [0] + [2 ** i for i in range(32) if 2 ** i < a.n_base] + [a.n_base]
    out = a.out / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{a.dataset}] base context {a.n_base} of {ctx_path.name} (stratified, seed "
          f"{a.subset_seed}), {F} features, K={K}, class counts {counts.tolist()}; "
          f"test {len(yt)} ({test_path.name})\nNaN rows added: {ns}")

    diff_clf = build_tabpfn_v2(device, a.model_seed, n_classes=K)
    std_clf = standard_classifier(device, a.model_seed)

    # The standard API refuses the rows as specified; record its message.
    try:
        std_clf.fit(np.vstack([X, np.full((1, F), np.nan, np.float32)]),
                    np.append(y.astype(np.float64), np.nan))
        std_nan_label = "accepted"
    except Exception as e:  # noqa: BLE001 -- we want the message, whatever the type
        std_nan_label = f"{type(e).__name__}: {e}".splitlines()[0]
    print(f"standard API with a NaN label: {std_nan_label}")

    rows, probs = [], {}
    for name in a.variants:
        path, nan_aware, label_rule = VARIANTS[name]
        for n in ns:
            t0 = time.time()
            Xc = np.vstack([X, np.full((n, F), np.nan, np.float32)])
            extra = (np.full(n, np.nan, np.float32) if label_rule == "nan"
                     else proportional_labels(n, counts).astype(np.float32))
            yc = np.concatenate([y.astype(np.float32), extra])
            nan_base = nan_extra = None
            if path == "differentiable":
                p, nan_base, nan_extra = score_differentiable(
                    diff_clf, Xc, yc, Xt, device=device, nan_aware=nan_aware, n_base=a.n_base)
            else:
                std_clf.fit(Xc, yc.astype(int))
                assert std_clf.classes_.tolist() == list(range(K))
                p = std_clf.predict_proba(Xt).astype(np.float32)
            m = multiclass_metrics(p, yt, K)
            pred = np.bincount(p.argmax(1), minlength=K)
            rows.append({"variant": name, "n_nan": n, "nan_share": n / (a.n_base + n),
                         **{k: m[k] for k in MULTICLASS_METRICS},
                         "top_predicted_share": float(pred.max() / len(yt)),
                         "predicted_counts": pred.tolist(),
                         "nan_cells_base_rows": nan_base, "nan_cells_added_rows": nan_extra})
            probs[f"{name}__n{n}"] = p
            print(f"  {name:<28s} n={n:>5d}  acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}"
                  f"  ce {m['ce']:.4f}  mcc {m['mcc']:.4f}  auc {m['roc_auc']:.4f}"
                  f"  predicted {pred.tolist()}"
                  + (f"  NaN cells base/added {nan_base}/{nan_extra}" if nan_base is not None else "")
                  + f"  [{time.time() - t0:.1f}s]")

    # ---- sanity checks (recorded, then enforced after the results are saved) ----------
    by = {(r["variant"], r["n_nan"]): r for r in rows}
    checks, fails = {}, []
    if "pipeline" in a.variants:
        # the direct path above is evaluate_context, guards and all
        _, _, p_ref = evaluate_context(
            diff_clf, torch.tensor(X, device=device), torch.tensor(y.astype(np.float32), device=device),
            torch.tensor(Xt, device=device), torch.as_tensor(yt, device=device), need_grad=False)
        dp = float(np.abs(p_ref.float().cpu().numpy() - probs["pipeline__n0"]).max())
        checks["pipeline_n0_vs_evaluate_context_max_abs_dp"] = dp
        if dp > 1e-6:
            fails.append(f"pipeline n=0 differs from evaluate_context by {dp}")
        for n in ns[1:]:  # upstream z-norm: every cell of every base row turns NaN
            if by[("pipeline", n)]["nan_cells_base_rows"] != a.n_base * F:
                fails.append(f"pipeline n={n}: base rows not entirely NaN")
    for name in ("pipeline-nanaware", "pipeline-nanaware-labelled"):
        if name not in a.variants:
            continue
        for n in ns:  # NaN-aware z-norm: base rows untouched, added rows entirely NaN
            r = by[(name, n)]
            if r["nan_cells_base_rows"] != 0 or r["nan_cells_added_rows"] != n * F:
                fails.append(f"{name} n={n}: NaN cells base/added {r['nan_cells_base_rows']}/"
                             f"{r['nan_cells_added_rows']}, want 0/{n * F}")
        if "pipeline" in a.variants:  # ... and inert when there is no NaN at all
            dp = float(np.abs(probs[f"{name}__n0"] - probs["pipeline__n0"]).max())
            checks[f"{name}_n0_vs_pipeline_n0_max_abs_dp"] = dp
            if dp > 1e-4 or not np.array_equal(probs[f"{name}__n0"].argmax(1),
                                               probs["pipeline__n0"].argmax(1)):
                fails.append(f"{name} n=0 differs from pipeline n=0 (max |dp| {dp})")
    checks["failed"] = fails

    no_skill_acc = float(np.bincount(yt, minlength=K).max() / len(yt))
    write_json(out / "summary.json", {
        "dataset": a.dataset, "context_split": str(ctx_path), "test_split": str(test_path),
        "n_base": a.n_base, "subset_seed": a.subset_seed, "model_seed": a.model_seed,
        "n_features": F, "n_classes": K, "class_counts_base": counts.tolist(),
        "class_counts_test": np.bincount(yt, minlength=K).tolist(),
        "no_skill_accuracy": no_skill_acc, "n_nan": ns,
        "variants": {k: {"path": VARIANTS[k][0], "nan_aware_znorm": VARIANTS[k][1],
                         "added_row_label": VARIANTS[k][2]} for k in a.variants},
        "standard_api_with_nan_label": std_nan_label,
        "sanity_checks": checks, "rows": rows,
    })
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "probs.npz", y_test=yt.astype(np.int16), base_rows=base, **probs)
    print(f"wrote {out}/summary.json, results.csv, probs.npz")
    if fails:
        raise SystemExit("SANITY CHECKS FAILED:\n  " + "\n  ".join(fails))
    print("sanity checks passed:", {k: v for k, v in checks.items() if k != "failed"})


if __name__ == "__main__":
    main()

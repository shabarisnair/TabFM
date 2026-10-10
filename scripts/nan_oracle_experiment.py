#!/usr/bin/env python
"""Would masking poisoned context rows recover TabPFNv2? -- an oracle test of the gate idea.

The planned defence learns a per-row gate g in [0, 1] that slides a context row toward
"entirely missing". Before building it, this asks whether the IDEAL gate -- g = 0 on
exactly the poisoned rows, known by an oracle -- would recover the damage at all, and
which part of the row has to be masked: the features, the label, or both.

PART 1 -- hub-poisoned rows (gas-drift, context_5000, test_attack_1000, hub-flip-local at
R = 8 / 20 / 32 %, the same deterministic rows as results/multiclass/). Conditions:

  clean            true labels, every row                                  (ceiling)
  poisoned         hub labels, every row       (floor; reproduces results/multiclass)
  remove           the poisoned rows dropped                 (what masking should hit)
  feat-nan         poisoned rows' features written as NaN, poisoned label kept
  feat-gate        the same mask applied the way the gate would (at g=0): each poisoned
                   row's feature-encoder inputs [x1, x2, m1, m2] are replaced by the
                   all-missing input [0, 0, -2, -2], while every statistic TabPFN fits
                   (z-norm, imputation means, scaler) is still computed from the real
                   values. feat-nan instead lets those statistics see the NaNs.
  both-nan         features AND label NaN
  both-gate        feature gate + NaN label
  label-nan        label NaN, features kept
  label-nan-stock  label NaN through stock TabPFN -- reference only, see below

PART 2 -- unlabeled rows appended to a clean context (mirrors nan_rows_experiment.py).
Base: the same class-stratified 2000-row subset of gas-drift context_5000 (seed 0). To
it, append n real rows drawn from the other 3000 rows of context_5000, n = 0,1,2,4,..,2000
(nested prefixes of one stratified random order):

  labelled         appended rows keep their true labels           (control: more data)
  nan-label        their labels NaN                                    (the new ablation)
  nan-label-stock  labels NaN through stock TabPFN                         (reference)
  nan-feature      NaN features, labels in the base class proportions (earlier result,
                   re-run here so both sit in one table)

HOW "NaN" IS IMPLEMENTED -- two pipeline problems would otherwise swamp the question:

  * Features. TabPFN's DifferentiableZNormStep uses X.mean / X.std, so one NaN in a
    column turns the whole column NaN in every row (memory: tabpfn-nan-handling). We use
    the NaN-aware z-norm from nan_rows_experiment.py: identical when there is no NaN.
  * Labels. TabPFN imputes a NaN label with the mean real label and only THEN builds its
    class list, so the imputed value becomes a phantom class and every class above it is
    renumbered -- a collapse from bookkeeping, not modelling. We patch _embed_targets to
    build the class list from the REAL labels only. A NaN label then encodes exactly as a
    test row's unknown label, [dense(mean label), -2], which is also what the gate does
    to a label at g = 0. The "-stock" variants keep the original code, to show the bug.

Every condition runs through the project's differentiable pipeline (build_tabpfn_v2 +
fit_with_differentiable_input + runtime._forward), one estimator, model seed 0, so a
point is one deterministic fit. sklearn's fit cannot be used: it rejects NaN labels.

    python scripts/nan_oracle_experiment.py --part check --gpu 1
    python scripts/nan_oracle_experiment.py --part 1 --gpu 1
    python scripts/nan_oracle_experiment.py --part 2 --gpu 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nan_rows_experiment import nan_aware_znorm, proportional_labels  # noqa: E402
from prepare_multiclass_splits import stratified_exact  # noqa: E402
from tabfm_experiments.config import Seeds  # noqa: E402
from tabfm_experiments.io import write_json  # noqa: E402
from tabfm_experiments.metrics import MULTICLASS_METRICS, multiclass_metrics  # noqa: E402

DATASET = "gas-drift"
R_PERCENTS = (8, 20, 32)
NAN_INDICATOR = -2.0


# ---- NaN-aware label densifier ---------------------------------------------------------

def _embed_targets_real_classes(self, y, *, num_rows, num_train_labels, batch_size):
    """TabPFNV2._embed_targets, with the class list built from the REAL labels only.

    Upstream computes ``unique_ys`` AFTER imputing NaN labels with the mean label, so the
    imputed value joins the class list as a phantom class. Taking the unique values of
    the finite training labels BEFORE imputation fixes that and nothing else: with no NaN
    label the two class lists are identical, so the output is bit-for-bit unchanged.
    """
    from tabpfn.architectures import tabpfn_v2 as T

    y_RB1 = T._prepare_targets(y, num_rows, batch_size)
    nan_indicator_RB1 = T._generate_nan_and_inf_indicator(y_RB1)
    unique_ys: list[torch.Tensor] = []
    if self.use_multiclass_target_encoding:
        for b in range(batch_size):
            v = y_RB1[:num_train_labels, b, 0]
            unique_ys.append(torch.unique(v[torch.isfinite(v)]))
    y_RB1, feature_means = T._impute_nan_and_inf_with_mean(
        y_RB1, num_train_labels, row_weights=getattr(self, "row_weights", None))
    if self.use_multiclass_target_encoding:
        y_RB1 = T._flatten_multiclass_targets(y_RB1, unique_ys)
    y_concat_RB1 = torch.cat([y_RB1, nan_indicator_RB1], dim=-1)
    y_concat_RB1 = y_concat_RB1.to(self.target_embedder.weight.dtype)
    embedded_y_RBX = self.target_embedder(y_concat_RB1)
    return embedded_y_RBX.transpose(0, 1), feature_means, unique_ys


@contextlib.contextmanager
def label_densifier(mode: str):
    """mode 'real' -> class list from the real labels; 'stock' -> TabPFN as shipped."""
    from tabpfn.architectures.tabpfn_v2 import TabPFNV2

    if mode == "stock":
        yield
        return
    if mode != "real":
        raise ValueError(mode)
    upstream = TabPFNV2._embed_targets
    TabPFNV2._embed_targets = _embed_targets_real_classes
    try:
        yield
    finally:
        TabPFNV2._embed_targets = upstream


# ---- encoder-input gate / capture ------------------------------------------------------

class EncoderTap:
    """Forward pre-hooks on the two input encoders of the loaded TabPFNV2.

    feature_group_embedder sees (R, S, 2F) = [values | missingness indicators] per
    feature group; target_embedder sees (R, B, 2) = [densified label | indicator]. Rows
    are context first, in input order (checked by check_row_alignment), then test rows.

    ``gate_rows``: context rows whose feature input is replaced by ``null_x`` -- the
    encoder input of an all-missing row -- i.e. the gate at g = 0.
    """

    def __init__(self, clf, n_ctx: int, gate_rows=None, null_x: torch.Tensor | None = None,
                 capture: bool = False):
        self.arch = clf.models_[0]
        self.n_ctx = n_ctx
        self.gate = None if gate_rows is None else torch.as_tensor(np.asarray(gate_rows), dtype=torch.long)
        self.null_x = null_x
        self.capture = capture
        self.feat_calls: list[torch.Tensor | tuple] = []
        self.targ_calls: list[torch.Tensor | tuple] = []
        self._handles = []

    def _feat(self, mod, args):
        x = args[0]
        self.feat_calls.append(x.detach().float().cpu().clone() if self.capture else tuple(x.shape))
        if self.gate is None or len(self.gate) == 0:
            return None
        if x.shape[0] < self.n_ctx:
            raise RuntimeError(f"feature encoder called with {x.shape[0]} rows < context {self.n_ctx}: "
                               "the gate assumes context and test rows are embedded together")
        x = x.clone()
        x[self.gate.to(x.device)] = self.null_x.to(device=x.device, dtype=x.dtype)
        return (x,) + tuple(args[1:])

    def _targ(self, mod, args):
        y = args[0]
        self.targ_calls.append(y.detach().float().cpu().clone() if self.capture else tuple(y.shape))

    def __enter__(self):
        self._handles = [self.arch.feature_group_embedder.register_forward_pre_hook(self._feat),
                         self.arch.target_embedder.register_forward_pre_hook(self._targ)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


# ---- scoring ---------------------------------------------------------------------------

def score(clf, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray, *, device: str,
          densifier: str = "real", gate_rows=None, null_x=None, capture: bool = False):
    """runtime.evaluate_context (no grad), minus its label guard, plus the optional gate.

    evaluate_context calls check_label_range, which crashes on NaN labels; everything
    else is the same: fit on the context, then runtime._forward on the test rows.
    Returns (probs [n_test, K], tap).
    """
    from tabfm_experiments.runtime import _forward, _install_recompute, check_feature_width

    check_feature_width(clf, Xc.shape[1])
    with torch.no_grad(), nan_aware_znorm(True), label_densifier(densifier):
        clf.fit_with_differentiable_input(torch.tensor(Xc, dtype=torch.float32, device=device),
                                          torch.tensor(yc, dtype=torch.float32, device=device))
        _install_recompute(clf)
        with EncoderTap(clf, len(yc), gate_rows=gate_rows, null_x=null_x, capture=capture) as tap:
            p = _forward(clf, torch.tensor(Xt, dtype=torch.float32, device=device), need_grad=False)
    return p.float().cpu().numpy(), tap


def classes_present(y_known: np.ndarray, K: int) -> None:
    """With a class absent from the known labels the densifier would misalign classes."""
    present = set(np.unique(y_known[np.isfinite(y_known)]).astype(int).tolist())
    missing = sorted(set(range(K)) - present)
    if missing:
        raise SystemExit(f"known labels are missing classes {missing}; results would be misaligned")


def null_feature_input(clf, X: np.ndarray, y: np.ndarray, Xt: np.ndarray, device: str) -> torch.Tensor:
    """The feature-encoder input TabPFN builds for an all-NaN row, per feature group.

    Captured, not derived: an all-NaN query row is appended to the test rows and its
    slice of the encoder input is read back. Shape (S, 2F).
    """
    Xq = np.vstack([Xt[:4], np.full((1, X.shape[1]), np.nan, np.float32)])
    _, tap = score(clf, X, y, Xq, device=device, capture=True)
    if len(tap.feat_calls) != 1:
        raise SystemExit(f"expected one feature-encoder call, got {len(tap.feat_calls)}")
    return tap.feat_calls[0][-1].clone()


# ---- checks ----------------------------------------------------------------------------

def run_checks(clf, Xc, yc, Xt, yt, K, device, out: Path) -> dict:
    """Mechanism checks the experiments rely on. Raises on failure."""
    res, fails = {}, []
    n = len(yc)

    # 1. Call layout: context and test rows go through each encoder in ONE call, context first.
    _, tap = score(clf, Xc, yc, Xt, device=device)
    res["feature_encoder_calls"] = [list(s) for s in tap.feat_calls]
    res["target_encoder_calls"] = [list(s) for s in tap.targ_calls]
    if [s[0] for s in tap.feat_calls] != [n + len(yt)]:
        fails.append(f"feature encoder calls {tap.feat_calls}, want one call of {n + len(yt)} rows")

    # 2. Row alignment: swapping two context rows swaps exactly those encoder rows. The
    #    fitted statistics are permutation-invariant, so nothing else may move.
    i, j = 17, 3001 % n
    _, a = score(clf, Xc, yc, Xt, device=device, capture=True)
    Xs, ys = Xc.copy(), yc.copy()
    Xs[[i, j]], ys[[i, j]] = Xs[[j, i]], ys[[j, i]]
    _, b = score(clf, Xs, ys, Xt, device=device, capture=True)
    A, B = a.feat_calls[0], b.feat_calls[0]
    perm = torch.arange(A.shape[0]); perm[i], perm[j] = j, i
    d_swap = float((A[perm] - B).abs().max())
    res["row_alignment_max_abs_diff"] = d_swap
    if d_swap > 1e-4:
        fails.append(f"row alignment: swapped rows do not map to swapped encoder rows ({d_swap})")

    # 3. The all-missing feature input, and that it is what we expect: values 0, indicators -2.
    null_x = null_feature_input(clf, Xc, yc, Xt, device)
    if tuple(null_x.shape) != tuple(A.shape[1:]):
        fails.append(f"null input layout {tuple(null_x.shape)} != context layout {tuple(A.shape[1:])}")
    res["null_x_unique_patterns"] = torch.unique(null_x.round(decimals=5), dim=0).tolist()
    want = torch.cat([torch.zeros(null_x.shape[0], null_x.shape[1] // 2),
                      torch.full((null_x.shape[0], null_x.shape[1] // 2), NAN_INDICATOR)], -1)
    res["null_x_max_abs_dev_from_[0,0,-2,-2]"] = float((null_x - want).abs().max())

    # 4. Patches are exact no-ops when nothing is NaN (stock densifier, no NaN anywhere).
    p_real, _ = score(clf, Xc, yc, Xt, device=device, densifier="real")
    p_stock, _ = score(clf, Xc, yc, Xt, device=device, densifier="stock")
    res["densifier_noop_max_abs_dp"] = float(np.abs(p_real - p_stock).max())
    if res["densifier_noop_max_abs_dp"] > 1e-6:
        fails.append(f"real-class densifier is not a no-op ({res['densifier_noop_max_abs_dp']})")

    # 5. Gate with an empty set is a no-op.
    p_gate0, _ = score(clf, Xc, yc, Xt, device=device, gate_rows=[], null_x=null_x)
    res["empty_gate_max_abs_dp"] = float(np.abs(p_gate0 - p_real).max())
    if res["empty_gate_max_abs_dp"] > 0:
        fails.append(f"empty gate changed predictions ({res['empty_gate_max_abs_dp']})")

    # 6. A NaN label encodes exactly as a test row's unknown label (real-class densifier),
    #    and the stock densifier really does add a phantom class.
    yn = yc.astype(np.float32).copy()
    yn[[5, 6, 7]] = np.nan
    for mode in ("real", "stock"):
        _, t = score(clf, Xc, yn, Xt, device=device, densifier=mode, capture=True)
        T_in = t.targ_calls[0][:, 0]                         # (R, 2)
        nan_rows, test_rows = T_in[[5, 6, 7]], T_in[n:]
        same = bool((nan_rows - test_rows[0]).abs().max() == 0 and (test_rows - test_rows[0]).abs().max() == 0)
        known = T_in[:n][torch.from_numpy(np.isfinite(yn))]
        res[f"{mode}_densifier"] = {
            "nan_label_rows_input": nan_rows.tolist(), "test_row_input": test_rows[0].tolist(),
            "nan_rows_equal_test_rows": same,
            "known_label_dense_values": sorted(set(known[:, 0].round(decimals=4).tolist())),
        }
    if not res["real_densifier"]["nan_rows_equal_test_rows"]:
        fails.append("real-class densifier: NaN-label rows do not encode like test rows")
    if res["real_densifier"]["known_label_dense_values"] != [float(k) for k in range(K)]:
        fails.append(f"real-class densifier renumbered classes: {res['real_densifier']['known_label_dense_values']}")

    res["failed"] = fails
    write_json(out / "checks.json", res)
    for k, v in res.items():
        if k != "failed":
            print(f"  {k}: {v}")
    if fails:
        raise SystemExit("CHECKS FAILED:\n  " + "\n  ".join(fails))
    print("all checks passed")
    return {"null_x": null_x}


# ---- part 1 ----------------------------------------------------------------------------

PART1_CONDITIONS = ("poisoned", "remove", "feat-nan", "feat-gate", "both-nan", "both-gate",
                    "label-nan", "label-nan-stock")


def part1(clf, Xc, yc, Xt, yt, K, device, out: Path, null_x: torch.Tensor):
    from tabfm_experiments.multiclass_attacks import poison

    rows_out, probs = [], {}

    def record(r_pct, cond, p, n_ctx, extra=None):
        m = multiclass_metrics(p, yt, K)
        pred = np.bincount(p.argmax(1), minlength=K)
        rows_out.append({"R": r_pct, "condition": cond, "n_context": n_ctx,
                         **{k: m[k] for k in MULTICLASS_METRICS},
                         "predicted_counts": pred.tolist(), **(extra or {})})
        probs[f"R{r_pct}__{cond}"] = p.astype(np.float32)
        print(f"  R={r_pct:>2}%  {cond:<16s} n_ctx={n_ctx:>5d}  acc {m['accuracy']:.4f}  "
              f"bal {m['balanced_accuracy']:.4f}  ce {m['ce']:.4f}  mcc {m['mcc']:.4f}  "
              f"predicted {pred.tolist()}", flush=True)
        return m

    t0 = time.time()
    p_clean, _ = score(clf, Xc, yc, Xt, device=device)
    record(0, "clean", p_clean, len(yc))
    print(f"  [{time.time() - t0:.0f}s]")

    grid_checks = {}
    for r_pct in R_PERCENTS:
        t0 = time.time()
        rows, y_pois = poison("hub-flip-local", torch.tensor(Xc, device=device), yc, K, r_pct,
                              run_id=0, row_seed=Seeds.row_seed)
        rows = np.sort(np.asarray(rows))
        keep = np.setdiff1d(np.arange(len(yc)), rows)
        yp = y_pois.astype(np.float32)
        info = {"n_poisoned": int(len(rows))}

        # poisoned (all rows, hub labels)
        p, _ = score(clf, Xc, yp, Xt, device=device)
        m_pois = record(r_pct, "poisoned", p, len(yc), info)
        # cross-check against the grid run of the same attack (no patches there)
        g = Path("results/multiclass") / DATASET / f"hub-flip-local_r{r_pct:03d}" / "summary.json"
        if g.exists():
            gm = json.loads(g.read_text())["runs"][0]["metrics"]
            grid_checks[f"R{r_pct}"] = {k: abs(gm[k] - m_pois[k]) for k in MULTICLASS_METRICS}

        # remove
        classes_present(yp[keep], K)
        p, _ = score(clf, Xc[keep], yp[keep], Xt, device=device)
        record(r_pct, "remove", p, len(keep), info)

        # feature masks
        X_nan = Xc.copy(); X_nan[rows] = np.nan
        y_nan = yp.copy(); y_nan[rows] = np.nan
        p, _ = score(clf, X_nan, yp, Xt, device=device)
        record(r_pct, "feat-nan", p, len(yc), info)
        p, _ = score(clf, Xc, yp, Xt, device=device, gate_rows=rows, null_x=null_x)
        record(r_pct, "feat-gate", p, len(yc), info)
        classes_present(y_nan, K)
        p, _ = score(clf, X_nan, y_nan, Xt, device=device)
        record(r_pct, "both-nan", p, len(yc), info)
        p, _ = score(clf, Xc, y_nan, Xt, device=device, gate_rows=rows, null_x=null_x)
        record(r_pct, "both-gate", p, len(yc), info)
        # label masks
        p, _ = score(clf, Xc, y_nan, Xt, device=device)
        record(r_pct, "label-nan", p, len(yc), info)
        p, _ = score(clf, Xc, y_nan, Xt, device=device, densifier="stock")
        record(r_pct, "label-nan-stock", p, len(yc), info)
        print(f"  [R={r_pct}%: {time.time() - t0:.0f}s]", flush=True)

    # recovery, measured against oracle removal: 0 = no better than poisoned, 1 = as good as removal
    by = {(r["R"], r["condition"]): r for r in rows_out}
    for r in rows_out:
        if r["condition"] in ("clean", "poisoned", "remove"):
            continue
        pois, rem = by[(r["R"], "poisoned")], by[(r["R"], "remove")]
        for k in ("accuracy", "ce"):
            gap = rem[k] - pois[k]
            r[f"recovery_{k}"] = float((r[k] - pois[k]) / gap) if abs(gap) > 1e-12 else None

    # Separate processes: GPU float noise can flip a borderline test row (0.001 acc).
    fails = [f"{R}: grid mismatch {d}" for R, d in grid_checks.items()
             if d["ce"] > 5e-4 or d["accuracy"] > 0.002]
    write_json(out / "summary.json", {
        "dataset": DATASET, "attack": "hub-flip-local", "R_percent": list(R_PERCENTS),
        "context": "context_5000.csv (all rows)", "test": "test_attack_1000.csv",
        "n_classes": K, "conditions": ["clean", *PART1_CONDITIONS],
        "grid_reproduction_abs_diff": grid_checks, "rows": rows_out, "failed": fails})
    with open(out / "results.csv", "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows_out for k in r))   # first-appearance order
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows_out)
    np.savez_compressed(out / "probs.npz", y_test=yt.astype(np.int16), **probs)
    print(f"grid reproduction |diff|: {grid_checks}")
    if fails:
        raise SystemExit("POISONED CONDITION DOES NOT REPRODUCE THE GRID:\n  " + "\n  ".join(fails))


# ---- part 2 ----------------------------------------------------------------------------

PART2_VARIANTS = ("labelled", "nan-label", "nan-label-stock", "nan-feature")


def part2(clf, Xall, yall, Xt, yt, K, device, out: Path, n_base: int = 2000):
    base = stratified_exact(np.arange(len(yall)), yall, n_base, np.random.default_rng(0))
    pool = np.setdiff1d(np.arange(len(yall)), base)
    n_max = min(n_base, len(pool))
    order = np.random.default_rng(2).permutation(
        stratified_exact(pool, yall, n_max, np.random.default_rng(1)))
    Xb, yb = Xall[base], yall[base].astype(np.float32)
    counts = np.bincount(yall[base], minlength=K)
    ns = [0] + [2 ** i for i in range(32) if 2 ** i < n_max] + [n_max]
    F = Xall.shape[1]
    print(f"base {n_base} (stratified seed 0), pool {len(pool)}, appended n = {ns}")

    rows_out, probs = [], {}
    for v in PART2_VARIANTS:
        for n in ns:
            t0 = time.time()
            add = order[:n]
            if v == "nan-feature":
                Xc = np.vstack([Xb, np.full((n, F), np.nan, np.float32)])
                yc = np.concatenate([yb, proportional_labels(n, counts).astype(np.float32)])
            else:
                Xc = np.vstack([Xb, Xall[add]])
                ya = yall[add].astype(np.float32) if v == "labelled" else np.full(n, np.nan, np.float32)
                yc = np.concatenate([yb, ya])
            p, _ = score(clf, Xc, yc, Xt, device=device,
                         densifier="stock" if v == "nan-label-stock" else "real")
            m = multiclass_metrics(p, yt, K)
            pred = np.bincount(p.argmax(1), minlength=K)
            rows_out.append({"variant": v, "n_added": n, "added_share": n / (n_base + n),
                             **{k: m[k] for k in MULTICLASS_METRICS}, "predicted_counts": pred.tolist()})
            probs[f"{v}__n{n}"] = p.astype(np.float32)
            print(f"  {v:<16s} n={n:>5d}  acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  "
                  f"ce {m['ce']:.4f}  mcc {m['mcc']:.4f}  predicted {pred.tolist()}  "
                  f"[{time.time() - t0:.1f}s]", flush=True)

    # cross-check: nan-feature must reproduce the earlier experiment's labelled-NaN curve
    prev = Path("results/nan_rows") / DATASET / "results.csv"
    repro = {}
    if prev.exists():
        with open(prev) as fh:
            old = {int(r["n_nan"]): r for r in csv.DictReader(fh) if r["variant"] == "pipeline-nanaware-labelled"}
        for r in rows_out:
            if r["variant"] == "nan-feature" and r["n_added"] in old:
                repro[r["n_added"]] = {k: abs(float(old[r["n_added"]][k]) - r[k]) for k in ("accuracy", "ce")}
    write_json(out / "summary.json", {
        "dataset": DATASET, "base_rows": n_base, "base_subset_seed": 0,
        "appended_rows": "real rows from the rest of context_5000, stratified (seed 1), nested prefixes of one random order (seed 2)",
        "n_added": ns, "variants": list(PART2_VARIANTS),
        "reproduction_of_nan_rows_experiment_abs_diff": repro, "rows": rows_out})
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    np.savez_compressed(out / "probs.npz", y_test=yt.astype(np.int16), base_rows=base,
                        added_order=order, **probs)
    print(f"reproduction of nan_rows_experiment (nan-feature) |diff|: {repro}")


# ---- main ------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["check", "1", "2"])
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", type=Path, default=Path("results/nan_oracle"))
    a = ap.parse_args(argv)

    from attack_multiclass import default_splits
    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu

    device = device_from_gpu(a.gpu)
    ctx_path, test_path, K = default_splits(DATASET)
    Xc_df, yc, _ = load_split(ctx_path)
    Xt_df, yt, _ = load_split(test_path)
    Xc, Xt = Xc_df.to_numpy(np.float32), Xt_df.to_numpy(np.float32)
    yc, yt = np.asarray(yc).astype(int), np.asarray(yt).astype(int)
    out = a.out / DATASET / f"part{a.part}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{DATASET}] part {a.part} on {device}: context {Xc.shape}, test {Xt.shape}, K={K}")
    clf = build_tabpfn_v2(device, Seeds.model_seed, n_classes=K)

    if a.part == "check":
        run_checks(clf, Xc, yc, Xt, yt, K, device, out)
    elif a.part == "1":
        null_x = run_checks(clf, Xc, yc, Xt, yt, K, device, out)["null_x"]
        part1(clf, Xc, yc, Xt, yt, K, device, out, null_x)
    else:
        part2(clf, Xc, yc, Xt, yt, K, device, out)
    if device.startswith("cuda"):
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

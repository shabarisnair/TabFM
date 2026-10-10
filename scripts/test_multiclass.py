#!/usr/bin/env python
"""Correctness tests for the multiclass runtime, metrics and label-perturbation attacks.

Every check is against an INDEPENDENT reference -- a second implementation, a stored
result produced by the binary pipeline, or a property that must hold exactly -- never
against the code under test itself.

  A. metrics      multiclass_metrics vs separate implementations (manual CE, manual
                  Gorodkin MCC, sklearn balanced accuracy, sklearn one-vs-rest macro
                  AUC); and at K=2 vs binary_metrics/with_derived.
  B. attacks      random_other_class is exactly uniform and never the own class;
                  local_nearest_other_class matches a brute-force numpy loop; poison()
                  has the right row counts, determinism and invariants; and at K=2 every
                  attack reproduces the BINARY attack's rows and labels.
  C. runtime      default n_classes=2 reproduces stored binary results bit-for-bit;
                  the guards fire; multiclass predictions are class-aligned (vs TabPFN's
                  own sklearn path and the confusion diagonal); and a context missing a
                  class really does misalign predictions -- i.e. the guard is needed.
  D. end-to-end   at K=2, hub-flip-local through the multiclass code path reproduces the
                  stored binary hub-flip run's rows AND test metrics.

    python scripts/test_multiclass.py --gpu 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name:<62s} {detail}")
    if not cond:
        FAILS.append(name)


def gorodkin_mcc(conf: np.ndarray) -> float:
    """Multiclass MCC (R_K) straight from the confusion matrix -- an independent formula."""
    C = conf.astype(np.float64)
    t, p, c, s = C.sum(1), C.sum(0), np.trace(C), C.sum()
    den = math.sqrt((s * s - p @ p) * (s * s - t @ t))
    return 0.0 if den == 0 else float((c * s - t @ p) / den)


# ============================================================================ A. metrics
def test_metrics():
    from tabfm_experiments.metrics import binary_metrics, multiclass_metrics, with_derived
    print("\nA. metrics")
    rng = np.random.default_rng(0)
    for K in (3, 6):
        n = 2000
        y = rng.integers(0, K, n)
        logits = rng.normal(size=(n, K)); logits[np.arange(n), y] += 1.2   # informative
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        m = multiclass_metrics(p, y, K)
        pred = p.argmax(1)
        conf = np.zeros((K, K), int); np.add.at(conf, (y, pred), 1)
        ref = {"ce": float(-np.log(p[np.arange(n), y]).mean()),
               "accuracy": float((pred == y).mean()),
               "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
               "mcc": gorodkin_mcc(conf),
               "roc_auc": float(roc_auc_score(y, p, multi_class="ovr", average="macro"))}
        for k, v in ref.items():
            check(f"K={K} {k} vs independent implementation", abs(m[k] - v) < 1e-9,
                  f"{m[k]:.10f} vs {v:.10f}")
        check(f"K={K} confusion matrix", m["confusion"] == conf.tolist())

    # K=2 must equal the binary pipeline exactly
    n = 1500
    y = rng.integers(0, 2, n)
    p1 = 1 / (1 + np.exp(-(rng.normal(size=n) + 1.5 * (2 * y - 1))))
    P = np.stack([1 - p1, p1], 1)
    mm = multiclass_metrics(P, y, 2)
    bb = with_derived(binary_metrics(P, y))
    for k in ("ce", "accuracy", "balanced_accuracy", "mcc", "roc_auc"):
        check(f"K=2 {k} equals binary_metrics", abs(mm[k] - bb[k]) < 1e-12,
              f"{mm[k]:.12f} vs {bb[k]:.12f}")

    # edge cases
    yk = np.repeat(np.arange(4), 50)
    perfect = multiclass_metrics(np.eye(4)[yk], yk, 4)
    check("perfect predictions -> acc/bacc/mcc/auc all 1.0",
          all(abs(perfect[k] - 1) < 1e-12 for k in ("accuracy", "balanced_accuracy", "mcc", "roc_auc")))
    const = multiclass_metrics(np.tile([0.7, 0.1, 0.1, 0.1], (200, 1)), yk, 4)
    check("constant prediction -> MCC 0 and AUC 0.5", abs(const["mcc"]) < 1e-12 and abs(const["roc_auc"] - 0.5) < 1e-12,
          f"mcc {const['mcc']}, auc {const['roc_auc']}")
    try:
        multiclass_metrics(np.full((5, 3), 1 / 3), np.array([0, 1, 2, 3, 0]), 3)
        check("label outside 0..K-1 is rejected", False)
    except ValueError:
        check("label outside 0..K-1 is rejected", True)


# ============================================================================ B. attacks
def brute_local_nearest(Z: np.ndarray, y: np.ndarray, rows, K: int, k: int):
    """Plain loops, no vectorisation: the reference for local_nearest_other_class."""
    out = []
    for i in rows:
        best, best_c = math.inf, None
        for c in range(K):
            if c == y[i]:
                continue
            d = sorted(float(np.linalg.norm(Z[i] - Z[j])) for j in range(len(y)) if y[j] == c and j != i)
            score = sum(d[:k]) / min(k, len(d))
            if score < best:                       # strict: ties keep the lower class
                best, best_c = score, c
        out.append(best_c)
    return np.array(out)


def test_attacks(device: str):
    from tabfm_experiments.multiclass_attacks import (
        local_nearest_other_class, poison, random_other_class, standardise,
    )
    from tabfm_experiments.sampling import bernoulli_row_indices, k_from_percent, random_flip_rng
    print("\nB. attacks")
    rng = np.random.default_rng(1)

    # B1. random_other_class: never the own class, exactly uniform over the others
    for K in (3, 6):
        src = np.repeat(np.arange(K), 30000)
        t = random_other_class(src, K, np.random.default_rng(2))
        check(f"K={K} random_other_class never returns the own class", bool(np.all(t != src)))
        worst = 0.0
        for c in range(K):
            counts = np.bincount(t[src == c], minlength=K)
            others = np.delete(counts, c)
            worst = max(worst, float(np.abs(others / others.sum() - 1 / (K - 1)).max()))
        check(f"K={K} random_other_class uniform over the {K-1} others", worst < 0.012,
              f"max deviation from 1/{K-1} = {worst:.4f}")
    t2 = random_other_class(np.array([0, 1, 0, 1, 1]), 2, np.random.default_rng(3))
    check("K=2 random_other_class == 1 - y", t2.tolist() == [1, 0, 1, 0, 0])

    # B2. local_nearest_other_class vs brute force
    for K, n, k in ((4, 240, 10), (3, 180, 5), (2, 150, 10)):
        X = rng.normal(size=(n, 7)).astype(np.float32)
        y = rng.integers(0, K, n)
        X[np.arange(n), y % 7] += 2.0                       # give the classes some structure
        rows = rng.choice(n, 40, replace=False)
        Xt = torch.tensor(X, device=device)
        got = local_nearest_other_class(Xt, y, rows, K, k=k, chunk=17)   # odd chunk on purpose
        Z = standardise(Xt).cpu().numpy()
        want = brute_local_nearest(Z, y, rows, K, k)
        check(f"K={K} local_nearest_other_class == brute force (k={k})", np.array_equal(got, want),
              f"{int((got == want).sum())}/{len(rows)} agree")
        check(f"K={K} local_nearest_other_class never the own class", bool(np.all(got != y[rows])))
        if K == 2:
            check("K=2 local_nearest_other_class == 1 - y", np.array_equal(got, 1 - y[rows]))

    # B3-B5. poison() invariants on a real multiclass context
    from tabfm_experiments.data import load_split
    from hub_flip_attention import hub_scores, pick_hubs
    Xdf, yc, _ = load_split("datasets/students_dropout/splits/context_3000.csv")
    yc = np.asarray(yc).astype(int); K = 3; n = len(yc)
    Xc = torch.tensor(Xdf.to_numpy(np.float32), device=device)

    r0, y0 = poison("label-flip-bernoulli", Xc, yc, K, 0, run_id=0, row_seed=1)
    check("R=0 flips nothing", len(r0) == 0 and np.array_equal(y0, yc))
    counts, firsts = [], []
    for run in range(40):
        rows, yp = poison("label-flip-bernoulli", Xc, yc, K, 20, run_id=run, row_seed=1)
        counts.append(len(rows))
        ok = (np.all(yp[rows] != yc[rows]) and np.array_equal(np.delete(yp, rows), np.delete(yc, rows)))
        if not ok:
            check(f"bernoulli run {run}: flipped rows change, others untouched", False)
        firsts.append(tuple(rows[:5]))
    mu, sd = n * 0.2, math.sqrt(n * 0.2 * 0.8)
    check("bernoulli: flipped rows change class, the rest are untouched (40 runs)", True)
    check("bernoulli: count ~ Binomial(n, R)", abs(np.mean(counts) - mu) < 3 * sd / math.sqrt(40),
          f"mean {np.mean(counts):.1f} vs {mu:.0f}, sd {np.std(counts):.1f} vs {sd:.1f}")
    check("bernoulli: different runs draw different rows", len(set(firsts)) > 35)
    a1 = poison("label-flip-bernoulli", Xc, yc, K, 20, run_id=7, row_seed=1)
    a2 = poison("label-flip-bernoulli", Xc, yc, K, 20, run_id=7, row_seed=1)
    check("bernoulli: same (seed, run) reproduces exactly",
          np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]))

    hubs = pick_hubs(hub_scores(Xc, "knn", 10), k_from_percent(n, 30))
    rl0, yl0 = poison("hub-flip-local", Xc, yc, K, 30, run_id=0, row_seed=1)
    rl5, yl5 = poison("hub-flip-local", Xc, yc, K, 30, run_id=5, row_seed=1)
    check("hub-local: rows are exactly the R% hubs", np.array_equal(rl0, hubs), f"{len(rl0)} rows")
    check("hub-local: deterministic across run ids", np.array_equal(yl0, yl5))
    check("hub-local: targets == local_nearest_other_class",
          np.array_equal(yl0[rl0], local_nearest_other_class(Xc, yc, rl0, K)))
    rr0, yr0 = poison("hub-flip-random", Xc, yc, K, 30, run_id=0, row_seed=1)
    rr1, yr1 = poison("hub-flip-random", Xc, yc, K, 30, run_id=1, row_seed=1)
    check("hub-random: same rows as hub-local", np.array_equal(rr0, rl0) and np.array_equal(rr1, rl0))
    check("hub-random: targets never the own class", bool(np.all(yr0[rr0] != yc[rr0])))
    check("hub-random: targets redrawn per run", not np.array_equal(yr0, yr1),
          f"{int((yr0[rr0] != yr1[rr1]).sum())}/{len(rr0)} hubs differ")

    # B6. K=2: every multiclass attack == its binary counterpart
    Xb_df, yb, _ = load_split("datasets/credit-approval/splits/train.csv")
    yb = np.asarray(yb).astype(int)
    Xb = torch.tensor(Xb_df.to_numpy(np.float32), device=device)
    for run in range(5):
        rows_mc, y_mc = poison("label-flip-bernoulli", Xb, yb, 2, 30, run_id=run, row_seed=1)
        rows_bin = bernoulli_row_indices(yb, 0.30, attack_class=None, rng=random_flip_rng(1, run, 0))
        y_bin = yb.copy(); y_bin[rows_bin] = 1 - y_bin[rows_bin]
        if not (np.array_equal(rows_mc, rows_bin) and np.array_equal(y_mc, y_bin)):
            check(f"K=2 bernoulli run {run} == binary label-flip-bernoulli", False)
            break
    else:
        check("K=2 bernoulli == binary label-flip-bernoulli (5 runs, rows and labels)", True)
    for R in (10, 20, 40):
        stored = np.array(json.loads(Path(f"results/hub_flip/credit-approval/r{R:03d}/summary.json")
                                     .read_text())["flipped_indices"])
        for atk in ("hub-flip-local", "hub-flip-random"):
            rows, yp = poison(atk, Xb, yb, 2, R, run_id=3, row_seed=1)
            check(f"K=2 {atk} R={R} == stored binary hub-flip",
                  np.array_equal(np.sort(rows), np.sort(stored)) and np.array_equal(yp[rows], 1 - yb[rows]),
                  f"{len(rows)} rows")


# ============================================================================ C. runtime
def test_runtime(device: str):
    from tabfm_experiments.config import inference_config
    from tabfm_experiments.data import load_split
    from tabfm_experiments.metrics import binary_metrics, multiclass_metrics
    from tabfm_experiments.runtime import (
        build_tabpfn_v2, check_all_classes_present, check_label_range, evaluate_context,
    )
    print("\nC. runtime")

    # C1. default n_classes=2 reproduces stored binary results bit-for-bit
    for ds in ("url_unique", "heloc"):
        s = json.loads(Path(f"results/main_experiments/{ds}/label-flip-random_random_r030/summary.json").read_text())
        Xdf, y, _ = load_split(s["train"]); Xt_df, yt, _ = load_split(s["test"])
        clf = build_tabpfn_v2(device, 0)
        _, _, p = evaluate_context(clf, torch.tensor(Xdf.to_numpy(np.float32), device=device),
                                   torch.tensor(np.asarray(y).astype(np.float32), device=device),
                                   torch.tensor(Xt_df.to_numpy(np.float32), device=device),
                                   torch.tensor(np.asarray(yt).astype(np.int64), device=device),
                                   need_grad=False)
        m = binary_metrics(p.cpu().numpy(), np.asarray(yt)).as_dict()
        same = all(abs(m[k] - s["clean"][k]) < 1e-6 for k in ("ce", "roc_auc", "accuracy")) and \
            all(m[k] == s["clean"][k] for k in ("tn", "fp", "fn", "tp"))
        check(f"{ds}: default build reproduces the stored binary clean run", same,
              f"ce {m['ce']:.6f} vs {s['clean']['ce']:.6f}")
        del clf; torch.cuda.empty_cache()

    # C2. guards
    clf2 = build_tabpfn_v2(device, 0)
    try:
        check_label_range(clf2, torch.tensor([0., 1., 5.]))
        check("label beyond n_classes is rejected", False)
    except ValueError:
        check("label beyond n_classes is rejected (the silent-truncation trap)", True)
    try:
        check_all_classes_present(np.array([0, 1, 3, 3]), 4)
        check("context missing a class is rejected", False)
    except ValueError:
        check("context missing a class is rejected", True)
    try:
        build_tabpfn_v2(device, 0, n_classes=11)
        check("n_classes above TabPFN's limit of 10 is rejected", False)
    except ValueError:
        check("n_classes above TabPFN's limit of 10 is rejected", True)
    del clf2

    # C3. multiclass class alignment, against TabPFN's own sklearn path
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    for ds, K in (("students_dropout", 3), ("gas-drift", 6)):
        info = json.loads(Path(f"datasets/{ds}/splits/split_info.json").read_text())
        ctx = next(n for n in info["splits"] if n.startswith("context_"))
        tst = next(n for n in info["splits"] if n.startswith("test_attack_"))
        Xdf, y, _ = load_split(f"datasets/{ds}/splits/{ctx}")
        Xt_df, yt, _ = load_split(f"datasets/{ds}/splits/{tst}")
        y, yt = np.asarray(y).astype(int), np.asarray(yt).astype(int)
        clf = build_tabpfn_v2(device, 0, n_classes=K)
        _, _, p = evaluate_context(clf, torch.tensor(Xdf.to_numpy(np.float32), device=device),
                                   torch.tensor(y.astype(np.float32), device=device),
                                   torch.tensor(Xt_df.to_numpy(np.float32), device=device),
                                   torch.tensor(yt, dtype=torch.long, device=device), need_grad=False)
        p = p.cpu().numpy()
        m = multiclass_metrics(p, yt, K)
        majority = np.bincount(yt).max() / len(yt)
        check(f"{ds}: probabilities are [n, K] and sum to 1",
              p.shape == (len(yt), K) and np.allclose(p.sum(1), 1, atol=1e-5), f"shape {p.shape}")
        conf = np.array(m["confusion"])
        diag_rate = np.diag(conf) / conf.sum(1)
        check(f"{ds}: accuracy far above the majority rate", m["accuracy"] > majority + 0.15,
              f"acc {m['accuracy']:.4f} vs majority {majority:.4f}")
        check(f"{ds}: every class mostly predicted as itself", bool(np.all(diag_rate > 1 / K)),
              f"per-class recall {np.round(diag_rate, 3).tolist()}")
        ref = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, n_estimators=1, device=device, random_state=0,
            inference_precision=torch.float32, ignore_pretraining_limits=True,
            n_preprocessing_jobs=1, inference_config=inference_config())
        ref.fit(Xdf.to_numpy(np.float32), y)
        pr = ref.predict_proba(Xt_df.to_numpy(np.float32))
        agree = float((pr.argmax(1) == p.argmax(1)).mean())
        check(f"{ds}: argmax agrees with TabPFN's own sklearn path", agree > 0.95,
              f"{100 * agree:.1f}% agree (a class mix-up would sit near {100 / K:.0f}%)")
        del clf, ref; torch.cuda.empty_cache()

        # C4. without the guard, a missing class really does misalign predictions
        if ds == "gas-drift":
            drop = 2
            keep = y != drop
            clf = build_tabpfn_v2(device, 0, n_classes=K)
            _, _, pm = evaluate_context(clf, torch.tensor(Xdf.to_numpy(np.float32)[keep], device=device),
                                        torch.tensor(y[keep].astype(np.float32), device=device),
                                        torch.tensor(Xt_df.to_numpy(np.float32), device=device),
                                        torch.tensor(yt, dtype=torch.long, device=device), need_grad=False)
            pm = pm.cpu().numpy().argmax(1)
            above = yt > drop
            acc_above = float((pm[above] == yt[above]).mean())
            acc_full = float((p.argmax(1)[above] == yt[above]).mean())
            check("missing class misaligns predictions (guard is needed)", acc_above < 0.5 * acc_full,
                  f"accuracy on classes above the dropped one: {acc_full:.3f} -> {acc_above:.3f}")
            del clf; torch.cuda.empty_cache()


# ============================================================================ D. end to end
def test_end_to_end(device: str):
    from tabfm_experiments.data import load_split
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2, evaluate_context
    print("\nD. end to end (K=2 through the multiclass path vs the stored binary hub-flip runs)")
    Xdf, y, _ = load_split("datasets/credit-approval/splits/train.csv")
    Xt_df, yt, _ = load_split("datasets/credit-approval/splits/test.csv")
    y, yt = np.asarray(y).astype(int), np.asarray(yt).astype(int)
    Xc = torch.tensor(Xdf.to_numpy(np.float32), device=device)
    Xt = torch.tensor(Xt_df.to_numpy(np.float32), device=device)
    clf = build_tabpfn_v2(device, 0, n_classes=2)
    for R in (10, 20, 30, 40):
        s = json.loads(Path(f"results/hub_flip/credit-approval/r{R:03d}/summary.json").read_text())
        stored = s["metrics_per_run"][0]
        rows, yp = poison("hub-flip-local", Xc, y, 2, R, run_id=0, row_seed=1)
        _, _, p = evaluate_context(clf, Xc, torch.tensor(yp.astype(np.float32), device=device), Xt,
                                   torch.tensor(yt, dtype=torch.long, device=device), need_grad=False)
        m = multiclass_metrics(p.cpu().numpy(), yt, 2)
        diffs = {k: abs(m[k] - stored[k]) for k in ("ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc")}
        check(f"credit-approval hub R={R}: multiclass path == stored binary run",
              max(diffs.values()) < 1e-6,
              f"max |diff| {max(diffs.values()):.2e} over ce/auc/acc/bacc/mcc")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args(argv)
    device = f"cuda:{a.gpu}"
    test_metrics()
    test_attacks(device)
    test_runtime(device)
    test_end_to_end(device)
    print("\n" + ("ALL TESTS PASSED" if not FAILS else f"FAILED ({len(FAILS)}): {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

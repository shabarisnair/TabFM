#!/usr/bin/env python
"""Defend a label-poisoned context with k-fold self-consistency filtering.

Given a context whose labels were corrupted at a KNOWN rate R% by an UNKNOWN strategy,
score every row by how well its own label agrees with the rest of the context (see
tabfm_experiments/defense.py), drop the R% that agree least, and re-run TabPFNv2 on the
real test set with what is left.

Four contexts are scored on the test set for each condition:

    clean       the untouched context                     -- the ceiling
    poisoned    corrupted, undefended                     -- the floor
    defended    corrupted minus the flagged rows          -- the method
    oracle      corrupted minus the rows REALLY flipped   -- the best any detector
                                                             could do at this R, and
                                                             the control for the fact
                                                             that removal also shrinks
                                                             the context

The poison is regenerated here from the same code paths the attacks use, so any R works
-- including ones the sweeps never ran. Where a matching sweep run exists on disk, the
regenerated row set is checked against the stored one and the result is recorded under
``poison_reproduction_check``.

Nothing but the final scoring touches the test set.

    python scripts/defend_context.py --gpu 0 --dataset heloc --attack label-flip-random \
        --row-percent 10 --n-runs 5 --out results/defense_kfold
    python scripts/defend_context.py --gpu 0 --dataset credit-approval --attack hub-flip \
        --row-percent 20 --out results/defense_kfold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hub_flip_attention import hub_scores, pick_hubs  # noqa: E402
from tabfm_experiments.config import DATASETS_DIR, Seeds  # noqa: E402
from tabfm_experiments.defense import (  # noqa: E402
    METRICS, delta, detection_report, flag_lowest, keep_mask, kfold_self_probs,
    make_folds, score_on_test,
)
from tabfm_experiments.io import setup_logger, write_json  # noqa: E402
from tabfm_experiments.sampling import k_from_percent, random_flip_rng, sample_row_indices  # noqa: E402

ATTACKS = ("label-flip-random", "hub-flip")


def default_splits(dataset: str) -> tuple[Path, Path]:
    """The train/test pair each dataset's attacks were run on.

    credit-approval has only 690 raw rows, so it was prepared with a plain 70/30 split
    instead of the 5000-row selected-context protocol the others use.
    """
    d = DATASETS_DIR / dataset
    if dataset == "credit-approval":
        return d / "splits/train.csv", d / "splits/test.csv"
    return (d / "splits/selected/natural/context_5000.csv",
            d / "splits/test_attack_1000.csv")


def poison_rows(attack: str, X_ctx: torch.Tensor, y_clean: np.ndarray, k: int, *,
                run_id: int, row_seed: int, hub_metric: str, hub_k: int) -> np.ndarray:
    """The rows the attacker flips -- regenerated from the attacks' own code paths."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    if attack == "label-flip-random":
        # Identical to attack_context.py: a fresh uniform draw per run, no test data.
        return sample_row_indices(y_clean, k, attack_class=None,
                                  rng=random_flip_rng(row_seed, run_id, 0))
    if attack == "hub-flip":
        # Deterministic: the k densest rows by mean k-NN distance (hub_flip_attention.py).
        return pick_hubs(hub_scores(X_ctx, hub_metric, hub_k), k)
    raise ValueError(f"unknown attack {attack!r}")


def sweep_dir_for(attack: str, dataset: str, row_percent: float) -> Path | None:
    """Where the matching attack sweep would have written its run, if it ran."""
    r = int(round(row_percent))
    if attack == "label-flip-random":
        p = Path("results/main_experiments") / dataset / f"label-flip-random_random_r{r:03d}"
    else:
        for root in (Path("results/hub_flip"),
                     Path("/hdd1/ssn899/TabFM/results/hub_flip_attention")):
            if (root / dataset / f"r{r:03d}" / "summary.json").exists():
                return root / dataset / f"r{r:03d}"
        return None
    return p if (p / "summary.json").exists() else None


def check_reproduction(attack: str, dataset: str, row_percent: float,
                       regenerated: dict[int, np.ndarray]) -> dict:
    """Compare the regenerated flip sets with the ones the sweep stored, when available.

    A mismatch means this script is not attacking the same rows the published numbers
    came from, which would make every comparison below incoherent -- so it is checked
    rather than assumed.
    """
    d = sweep_dir_for(attack, dataset, row_percent)
    if d is None:
        return {"available": False,
                "reason": f"no sweep run on disk for {dataset} {attack} R={row_percent:g}"}
    import json
    if attack == "hub-flip":
        stored = {0: np.asarray(json.loads((d / "summary.json").read_text())["flipped_indices"],
                                dtype=np.int64)}
    else:
        stored = {}
        for f in sorted(d.glob("trials/run*_sub0_delta.npz")):
            run = int(f.name.split("run")[1].split("_")[0])
            with np.load(f) as z:
                stored[run] = np.sort(z["flipped_indices"].astype(np.int64))
    shared = sorted(set(stored) & set(regenerated))
    per_run = {str(r): bool(np.array_equal(np.sort(regenerated[r]), np.sort(stored[r])))
               for r in shared}
    return {"available": True, "sweep_dir": str(d), "runs_compared": len(shared),
            "all_match": all(per_run.values()) if per_run else False, "per_run": per_run}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0", help="GPU index or 'cpu'")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", required=True, choices=ATTACKS)
    ap.add_argument("--row-percent", type=float, required=True,
                    help="R: the corruption rate, assumed KNOWN to the defender")
    ap.add_argument("--train", type=Path, default=None, help="default: the dataset's context/train split")
    ap.add_argument("--test", type=Path, default=None, help="default: the dataset's attack test split")
    ap.add_argument("--out", type=Path, default=Path("results/defense_kfold"))
    ap.add_argument("--method", default="kfold", choices=["kfold"],
                    help="detection method (more will be added)")
    ap.add_argument("--n-folds", type=int, default=20)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--n-poison-draws", type=int, default=None,
                    help="independent poison draws (the ATTACK's randomness). Default 5 for "
                         "label-flip-random; forced to 1 for hub-flip, which is deterministic")
    ap.add_argument("--n-fold-draws", type=int, default=5,
                    help="fold reshuffles per poison draw (the DEFENCE's randomness)")
    ap.add_argument("--row-seed", type=int, default=Seeds.row_seed,
                    help="must match the attack sweeps for the flip sets to line up")
    ap.add_argument("--hub-metric", default="knn", choices=["knn", "all"])
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--model-seed", type=int, default=Seeds.model_seed)
    ap.add_argument("--test-batch-size", type=int, default=None)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the perfect-removal reference (it costs one forward per run)")
    ap.add_argument("--force", action="store_true", help="recompute even if summary.json exists")
    a = ap.parse_args(argv)

    # A crossed design: every poison draw is defended under the SAME set of fold seeds,
    # so the two randomness sources can be separated rather than confounded.
    #   * hub-flip is deterministic -> 1 poison draw; all spread is the fold draw's.
    #   * label-flip-random -> n_poison_draws x n_fold_draws cells, and the aggregate
    #     reports the between-poison and within-poison(fold) components separately.
    if a.attack == "hub-flip":
        if a.n_poison_draws not in (None, 1):
            raise SystemExit("hub-flip is deterministic: --n-poison-draws must be 1 "
                             "(use --n-fold-draws to repeat the defence)")
        a.n_poison_draws = 1
    elif a.n_poison_draws is None:
        a.n_poison_draws = 5
    if a.n_poison_draws < 1 or a.n_fold_draws < 1:
        raise SystemExit("--n-poison-draws and --n-fold-draws must be >= 1")
    a.n_runs = a.n_poison_draws * a.n_fold_draws

    out = a.out / a.dataset / f"{a.attack}_r{int(round(a.row_percent)):03d}"
    if (out / "summary.json").exists() and not a.force:
        print(f"[skip] {out}/summary.json exists (use --force to recompute)")
        return None
    out.mkdir(parents=True, exist_ok=True)
    log_ = setup_logger(out, "defend_context")
    log = log_.info

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    train, test = a.train, a.test
    if train is None or test is None:
        dtr, dte = default_splits(a.dataset)
        train, test = train or dtr, test or dte
    device = device_from_gpu(a.gpu)

    Xtr_df, y_clean, cols = load_split(train)
    Xte_df, y_test, _ = load_split(test)
    X_ctx = torch.tensor(Xtr_df.to_numpy(np.float32), device=device)
    X_test = torch.tensor(Xte_df.to_numpy(np.float32), device=device)
    y_clean = np.asarray(y_clean).astype(int)
    y_test = np.asarray(y_test).astype(int)
    n = len(y_clean)
    k = 0 if a.row_percent <= 0 else k_from_percent(n, a.row_percent)

    log(f"[{a.dataset}] {a.attack} R={a.row_percent:g}%  context {n}x{len(cols)}, "
        f"test {len(y_test)},  k={k} rows flipped,  "
        f"{a.n_poison_draws} poison x {a.n_fold_draws} fold draws = {a.n_runs} cells, "
        f"{a.n_folds} folds each")
    log(f"  train {train}")
    log(f"  test  {test}")
    log(f"  clean class counts {np.bincount(y_clean, minlength=2).tolist()}")

    clf = build_tabpfn_v2(device, a.model_seed)
    y_clean_t = torch.as_tensor(y_clean, dtype=torch.float32, device=device)

    # The ceiling. Identical for every R and attack on this dataset, so it is computed
    # once here rather than per run.
    t0 = time.time()
    clean_m = score_on_test(clf, X_ctx, y_clean_t, X_test, y_test, evaluate_context,
                            test_batch_size=a.test_batch_size)
    log("  clean baseline: " + "  ".join(f"{m} {clean_m[m]:.4f}" for m in METRICS)
        + f"   [{time.time() - t0:.0f}s]")

    runs, regenerated = [], {}
    for pdraw in range(a.n_poison_draws):
        flipped = poison_rows(a.attack, X_ctx, y_clean, k, run_id=pdraw, row_seed=a.row_seed,
                              hub_metric=a.hub_metric, hub_k=a.hub_k)
        regenerated[pdraw] = np.asarray(flipped, dtype=np.int64)
        y_given = y_clean.copy()
        y_given[flipped] = 1 - y_given[flipped]
        y_given_t = torch.as_tensor(y_given, dtype=torch.float32, device=device)
        log(f"  poison draw {pdraw}: flipped {len(flipped)} rows "
            f"(clean classes of the flipped: {np.bincount(y_clean[flipped], minlength=2).tolist()}); "
            f"given class counts {np.bincount(y_given, minlength=2).tolist()}")

        # Neither of these depends on the fold split, so they are scored once per poison
        # draw rather than once per cell.
        poisoned_m = score_on_test(clf, X_ctx, y_given_t, X_test, y_test, evaluate_context,
                                   test_batch_size=a.test_batch_size)
        oracle_m = None
        if not a.no_oracle and k > 0:
            ok_t = torch.as_tensor(np.flatnonzero(keep_mask(n, flipped)), dtype=torch.long,
                                   device=device)
            oracle_m = score_on_test(clf, X_ctx[ok_t], y_given_t[ok_t], X_test, y_test,
                                     evaluate_context, test_batch_size=a.test_batch_size)

        for fdraw in range(a.n_fold_draws):
            t_run = time.time()
            run = pdraw * a.n_fold_draws + fdraw
            # Same fold-seed set for every poison draw: a crossed design, not nested.
            folds, stratified = make_folds(y_given, a.n_folds, a.fold_seed + fdraw)
            scores = kfold_self_probs(clf, X_ctx, y_given, folds, evaluate_context,
                                      test_batch_size=a.test_batch_size)
            flagged = flag_lowest(scores, k)
            det = detection_report(flagged, flipped, y_clean, y_given, scores)
            det["stratified_folds"] = stratified
            det["n_folds"] = a.n_folds

            # ---- rebuild the context and score it on the real test set --------------
            kept = keep_mask(n, flagged)
            kept_t = torch.as_tensor(np.flatnonzero(kept), dtype=torch.long, device=device)
            defended_m = score_on_test(clf, X_ctx[kept_t], y_given_t[kept_t], X_test, y_test,
                                       evaluate_context, test_batch_size=a.test_batch_size)

            rec = {
                "run_id": run, "poison_draw": pdraw, "fold_draw": fdraw,
                "n_flipped": int(len(flipped)),
                "flipped_indices": regenerated[pdraw].tolist(),
                "flagged_indices": np.asarray(flagged, dtype=np.int64).tolist(),
                "detection": det,
                "clean": clean_m, "poisoned": poisoned_m, "defended": defended_m,
                "delta_defended_vs_poisoned": delta(defended_m, poisoned_m),
                "delta_defended_vs_clean": delta(defended_m, clean_m),
                "delta_poisoned_vs_clean": delta(poisoned_m, clean_m),
                "n_context_defended": int(kept.sum()),
                "class_counts_defended": np.bincount(y_given[kept], minlength=2).tolist(),
            }
            if oracle_m is not None:
                rec["oracle"] = oracle_m
                rec["delta_oracle_vs_poisoned"] = delta(oracle_m, poisoned_m)
                rec["delta_defended_vs_oracle"] = delta(defended_m, oracle_m)
                # What share of the oracle's CE repair did the detector actually achieve?
                gap = oracle_m["ce"] - poisoned_m["ce"]
                rec["ce_repair_fraction_of_oracle"] = (
                    float((defended_m["ce"] - poisoned_m["ce"]) / gap) if abs(gap) > 1e-12 else None)

            runs.append(rec)
            np.savez_compressed(out / f"p{pdraw}_f{fdraw}_scores.npz",
                                scores=scores.astype(np.float32), folds=folds.astype(np.int16),
                                flipped_indices=regenerated[pdraw], flagged_indices=np.asarray(flagged),
                                y_clean=y_clean.astype(np.int8), y_given=y_given.astype(np.int8))
            log(f"    p{pdraw}/f{fdraw}: detection precision {det['precision']:.4f}  "
                f"ranking AUC {det['detection_roc_auc']:.4f}  |  "
                f"poisoned CE {poisoned_m['ce']:.4f} -> defended {defended_m['ce']:.4f}"
                + (f" (oracle {oracle_m['ce']:.4f})" if oracle_m is not None else "")
                + f"   acc {poisoned_m['accuracy']:.4f} -> {defended_m['accuracy']:.4f}"
                + f"   [{time.time() - t_run:.0f}s]")

    def decompose(vals_by_poison: list[list]) -> dict:
        """Split the spread into its attack and defence components.

        ``vals_by_poison[p][f]`` is the value for poison draw p under fold draw f.

          std                      total spread over all cells
          between_poison_std       spread of the per-poison means -- how much the
                                   ATTACK draw moves the result
          within_poison_fold_std   RMS of the per-poison stds -- how much the DEFENCE's
                                   own fold draw moves it with the poison held fixed

        For hub-flip there is one poison draw, so between_poison_std is 0 by
        construction and the whole spread is the fold draw's.
        """
        rows = [[float(v) for v in row if v is not None and v == v] for row in vals_by_poison]
        rows = [r for r in rows if r]
        flat = [v for r in rows for v in r]
        if not flat:
            return {"mean": None, "std": None, "between_poison_std": None,
                    "within_poison_fold_std": None, "n": 0}
        means = [float(np.mean(r)) for r in rows]
        within = [float(np.std(r)) for r in rows if len(r) > 1]
        return {
            "mean": float(np.mean(flat)), "std": float(np.std(flat)),
            "between_poison_std": float(np.std(means)) if len(means) > 1 else 0.0,
            "within_poison_fold_std": float(np.sqrt(np.mean(np.square(within)))) if within else 0.0,
            "n": len(flat), "n_poison_draws": len(rows),
        }

    def by_poison(get) -> list[list]:
        """Group ``get(run_record)`` into one list per poison draw, ordered by fold draw."""
        grouped: list[list] = [[] for _ in range(a.n_poison_draws)]
        for r in sorted(runs, key=lambda r: (r["poison_draw"], r["fold_draw"])):
            grouped[r["poison_draw"]].append(get(r))
        return grouped

    def agg(path: list[str]) -> dict:
        """Per-metric decomposition for one nested block, e.g. ["defended"]."""
        def pick(r, m):
            node = r
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    return None
            return node[m]
        out_ = {}
        for m in METRICS:
            d = decompose(by_poison(lambda r, m=m: pick(r, m)))
            if d["n"]:
                out_[m] = d
        return out_

    det_keys = ("precision", "recall", "f1", "accuracy", "detection_roc_auc",
                "false_removal_rate")
    det_agg = {kk: decompose(by_poison(lambda r, kk=kk: r["detection"][kk]))
               for kk in det_keys}
    det_agg["recall_by_true_class"] = {
        c: decompose(by_poison(lambda r, c=c: r["detection"]["recall_by_true_class"][c]["recall"]))
        for c in ("0", "1")}
    det_agg["precision_by_given_class"] = {
        c: decompose(by_poison(lambda r, c=c: r["detection"]["precision_by_given_class"][c]["precision"]))
        for c in ("0", "1")}

    summary = {
        "dataset": a.dataset, "attack": a.attack, "method": a.method,
        "row_percent": a.row_percent, "k_flipped": int(k), "removal_budget": int(k),
        "n_runs": a.n_runs, "n_poison_draws": a.n_poison_draws,
        "n_fold_draws": a.n_fold_draws, "n_folds": a.n_folds, "fold_seed": a.fold_seed,
        "row_seed": a.row_seed, "model_seed": a.model_seed,
        "hub_metric": a.hub_metric, "hub_k": a.hub_k,
        "deterministic_attack": a.attack == "hub-flip",
        "n_context": n, "n_features": len(cols), "n_test": len(y_test),
        "train": str(train), "test": str(test),
        "class_counts_clean": np.bincount(y_clean, minlength=2).tolist(),
        "clean": clean_m,
        "aggregate": {
            "poisoned": agg(["poisoned"]),
            "defended": agg(["defended"]),
            "oracle": agg(["oracle"]),
            "delta_defended_vs_poisoned": agg(["delta_defended_vs_poisoned"]),
            "delta_defended_vs_clean": agg(["delta_defended_vs_clean"]),
            "delta_poisoned_vs_clean": agg(["delta_poisoned_vs_clean"]),
            "delta_oracle_vs_poisoned": agg(["delta_oracle_vs_poisoned"]),
        },
        "aggregate_detection": det_agg,
        "poison_reproduction_check": check_reproduction(a.attack, a.dataset, a.row_percent,
                                                        regenerated),
        "runs": runs,
    }
    write_json(out / "summary.json", summary)

    chk = summary["poison_reproduction_check"]
    if chk.get("available"):
        log(f"  poison reproduction vs sweep: {'MATCH' if chk['all_match'] else 'MISMATCH'} "
            f"on {chk['runs_compared']} run(s)")
    else:
        log(f"  poison reproduction: {chk['reason']}")
    log(f"wrote {out / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()

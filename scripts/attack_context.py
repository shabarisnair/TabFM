#!/usr/bin/env python
"""Context poisoning of TabPFNv2: X_train CAPGD or influence-ranked Y_train label flips.

Djilani et al. (arXiv:2506.02978) perturb *test* X with the context held clean; here
the *context* is poisoned and the mean CE on --test is maximised. White-box,
transductive: the attacker reads --test and is scored on it (an upper bound).

Every forward -- clean baseline, each CAPGD step, final scoring -- refits TabPFNv2 on
the current, possibly poisoned context via tabfm_experiments.runtime.evaluate_context.

Trials are the grid run_id x subsample_id. Do not launch the full 5x5 grid casually:
each CAPGD step is a full fit+forward+backward.

Examples
--------
python scripts/attack_context.py --gpu 1 --attack x-capgd \
    --train datasets/url_unique/splits/selected/natural/context_1000.csv \
    --test  datasets/url_unique/splits/test_attack_1000.csv \
    --out   results/attacks/url_unique_x --n-runs 1 --n-row-subsamples 1

python scripts/attack_context.py --gpu 1 --attack label-flip --row-percent 5 \
    --train datasets/url_unique/splits/selected/natural/context_1000.csv \
    --test  datasets/url_unique/splits/test_attack_1000.csv \
    --out   results/attacks/url_unique_flip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tabfm_experiments.aggregate import best_per_run, summarize_trials  # noqa: E402
from tabfm_experiments.capgd import capgd  # noqa: E402,F401  (re-export for old imports)
from tabfm_experiments.io import (  # noqa: E402
    ensemble_config_record,
    setup_logger,
    versions_info,
    write_json,
)

CAA_PGD_MESSAGE = (
    "--method caa / pgd is not implemented, on purpose.\n"
    "  * CAA = CAPGD followed by MOEVA on the failures; MOEVA is a genetic search over the\n"
    "    joint input, which is infeasible over an R% block of a 1k-10k row context.\n"
    "  * PGD on Y_train is impossible on TabPFNv2: labels are densified with\n"
    "    (y > unique_ys).sum(), whose autograd derivative is 0, so dL/dY_train == 0.\n"
    "Use --attack x-capgd (features) or --attack label-flip (influence-ranked flips)."
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True, help="context CSV")
    ap.add_argument("--test", type=Path, required=True, help="attack/scoring CSV (test_attack_1000.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--model", default="tabpfnv2")
    ap.add_argument("--method", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--attack", default="x-capgd",
                    choices=["x-capgd", "label-flip", "label-flip-influence", "label-flip-random"],
                    help="label-flip = genetic search over k-subsets of rows (default Y-attack, "
                         "seeded with influence top-k); label-flip-influence = single-shot influence ranking; "
                         "label-flip-random = flip a uniformly random R%% of rows, a fresh draw per run "
                         "(test-agnostic: the test set is only used for scoring)")
    ap.add_argument("--gpu", default="1", help="GPU index or 'cpu'")
    ap.add_argument("--row-percent", type=float, default=5.0)
    ap.add_argument("--attack-class", type=int, choices=[0, 1], default=None)
    ap.add_argument("--n-row-subsamples", type=int, default=None,
                    help="default 3 (x-capgd) / 1 (label-flip, label-flip-influence, label-flip-random)")
    ap.add_argument("--n-runs", type=int, default=None,
                    help="default 5 (x-capgd, label-flip, label-flip-random) / 1 "
                         "(label-flip-influence: deterministic)")
    # CAPGD (Djilani / TabularBench defaults)
    ap.add_argument("--norm", default="l2", choices=["l2", "linf"])
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--eps-margin", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=40,
                    help="CAPGD steps. TabularBench uses 10; 40 chosen from the n_iter sweep "
                         "(results/niter_sweep_final, docs/context_poisoning.md): most of the "
                         "reachable signal on url_unique/wids at 2.5x less cost than 100")
    ap.add_argument("--momentum", type=float, default=0.75)
    ap.add_argument("--rho", type=float, default=0.75)
    ap.add_argument("--n-restarts", type=int, default=1)
    ap.add_argument("--eot-iter", type=int, default=1)
    ap.add_argument("--loss", default="ce", choices=["ce"])
    ap.add_argument("--random-start", action="store_true", help="random start for run 0 as well")
    ap.add_argument("--scaler", default="metadata", choices=["metadata", "train"])
    ap.add_argument("--constraints", default="full", choices=["none", "box", "full"])
    ap.add_argument("--constraint-penalty", type=float, default=1.0)
    ap.add_argument("--fix-equality-iter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget-repair", action=argparse.BooleanOptionalAction, default=True,
                    help="full mode: re-project the repaired row back into the eps-ball (default). "
                         "--no-budget-repair reproduces TabularBench (repair may exceed the budget)")
    ap.add_argument("--deterministic-mode", default="best-effort", choices=["best-effort", "strict"])
    ap.add_argument("--model-seed", "--seed", dest="model_seed", type=int, default=0)
    ap.add_argument("--row-seed", type=int, default=1)
    ap.add_argument("--attack-seed", type=int, default=2)
    ap.add_argument("--test-batch-size", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None, help="smoke tests only")
    ap.add_argument("--save-full-context", action="store_true",
                    help="write <out>/context_poisoned.csv for the single strongest trial "
                         "(largest CE increase) of the run x subsample grid")
    ap.add_argument("--recompute-layers", action="store_true",
                    help="activation checkpointing for gradient forwards: same results, far less GPU memory, ~25%% slower")
    ga = ap.add_argument_group("label-flip GA (searches k-subsets of the eligible rows; budget = R%% of rows)")
    ga.add_argument("--ga-population", type=int, default=16)
    ga.add_argument("--ga-generations", type=int, default=20, help="0 = random search over --ga-population masks")
    ga.add_argument("--ga-elite", type=int, default=2)
    ga.add_argument("--ga-tournament", type=int, default=3)
    ga.add_argument("--ga-crossover-prob", type=float, default=0.9)
    ga.add_argument("--ga-mutation-rate", type=float, default=None, help="per-bit flip probability (default ~1/pool)")
    ga.add_argument("--ga-seed-full", action=argparse.BooleanOptionalAction, default=True,
                    help="put the all-ones (budget-capped) mask into generation 0")
    ga.add_argument("--ga-no-influence-seed", action="store_true",
                    help="do NOT seed generation 0 with the influence top-k mask")
    ga.add_argument("--ga-patience", type=int, default=8, help="generations without improvement before stopping (0 = never)")
    ga.add_argument("--ga-batch-size", type=int, default=16, help="individuals per fused forward")
    ga.add_argument("--ga-fitness-test-rows", type=int, default=None,
                    help="score fitness on this many random test rows (default all); final metrics always use all")
    a = ap.parse_args(argv)

    if a.method is not None and a.method.lower() != "capgd":
        raise SystemExit(CAA_PGD_MESSAGE)
    if a.model != "tabpfnv2":
        raise SystemExit("only --model tabpfnv2 is supported on the shared runtime")
    # x-capgd: 5 runs x 3 subsamples. label-flip (GA): 5 runs (population seeds) x 1 subsample.
    # label-flip-influence: deterministic, so 1 x 1.
    if a.attack == "x-capgd":
        default_runs, default_subs = 5, 3
    elif a.attack in ("label-flip", "label-flip-random"):
        default_runs, default_subs = 5, 1
    else:  # label-flip-influence
        default_runs, default_subs = 1, 1
    if a.n_runs is None:
        a.n_runs = default_runs
    if a.n_row_subsamples is None:
        a.n_row_subsamples = default_subs
    if a.n_runs < 1 or a.n_row_subsamples < 1 or a.n_iter < 1 or a.n_restarts < 1 or a.eot_iter < 1:
        raise SystemExit("--n-runs, --n-row-subsamples, --n-iter, --n-restarts, --eot-iter must be >= 1")
    # The per-run aggregate keeps the best subsample BY TEST CE. For a test-agnostic attack
    # that selection would let the test set back in, so each run is exactly one random draw.
    if a.attack == "label-flip-random" and a.n_row_subsamples != 1:
        raise SystemExit("label-flip-random is test-agnostic: use --n-runs for more random draws; "
                         "--n-row-subsamples > 1 would pick the best draw by test CE")
    return a


def main(argv=None):
    a = parse_args(argv)
    from tabfm_experiments.attacks import (
        XCapgdConfig, run_label_flip_ga, run_label_flip_influence, run_label_flip_random, run_x_capgd,
    )
    from tabfm_experiments.ga import GAConfig
    from tabfm_experiments.data import infer_dataset_name, load_split, metadata_path_for, read_metadata
    from tabfm_experiments.metrics import binary_metrics
    from tabfm_experiments.runtime import (
        build_tabpfn_v2, chunking_is_exact, device_from_gpu, evaluate_context, max_repeat_prob_delta,
    )
    from tabfm_experiments.sampling import (
        attack_seed_for, eligible_indices, k_from_percent, random_flip_rng, row_rng, sample_row_indices,
    )

    out = a.out
    (out / "trials").mkdir(parents=True, exist_ok=True)
    log = setup_logger(out, "attack_context")
    write_json(out / "args.json", vars(a))
    device = device_from_gpu(a.gpu)

    Xtr_df, ytr, cols = load_split(a.train, a.target)
    Xte_df, yte, cols_te = load_split(a.test, a.target)
    if cols != cols_te:
        raise SystemExit("train and test feature columns differ")
    for name, y in (("train", ytr), ("test", yte)):
        if not set(np.unique(y)).issubset({0, 1}):
            raise SystemExit(f"{name} labels must be binary 0/1")
    n_test_full = len(yte)
    if a.max_test and n_test_full > a.max_test:
        sel = np.sort(np.random.default_rng(0).choice(n_test_full, a.max_test, replace=False))
        Xte_df, yte = Xte_df.iloc[sel], yte[sel]

    Xc = torch.tensor(Xtr_df.to_numpy(dtype=np.float32), device=device)
    yc = torch.tensor(ytr.astype(np.float32), device=device)
    Xt = torch.tensor(Xte_df.to_numpy(dtype=np.float32), device=device)
    yt = torch.tensor(yte.astype(np.int64), device=device)
    n_ctx = len(ytr)
    k = k_from_percent(n_ctx, a.row_percent)
    log.info(f"[{a.attack}] {a.train} ({n_ctx}x{len(cols)}) -> {a.test} ({len(yte)} of {n_test_full})  "
             f"k={k} ({a.row_percent}%)  attack_class={a.attack_class}  grid {a.n_runs}x{a.n_row_subsamples}  {device}")
    n_elig = len(eligible_indices(ytr, a.attack_class))
    if n_elig < k:
        log.info(f"  only {n_elig} eligible rows < k={k}; attacking all eligible rows")

    clf = build_tabpfn_v2(device, a.model_seed, a.deterministic_mode, recompute_layers=a.recompute_layers)
    t0 = time.time()
    ce0, _, p0 = evaluate_context(clf, Xc, yc, Xt, yt, need_grad=False, test_batch_size=a.test_batch_size)
    clean = binary_metrics(p0.cpu().numpy(), yte)
    max_dp = max_repeat_prob_delta(clf, Xc, yc, Xt, yt, n_repeats=3, test_batch_size=a.test_batch_size)
    exact = chunking_is_exact(Xc, Xt)
    log.info(f"  clean  ce={clean.ce:.5f} auc={clean.roc_auc:.4f} acc={clean.accuracy:.4f} f1={clean.f1:.4f}  "
             f"max|dp| over 3 repeats={max_dp:.2e}  chunking_exact={exact}  [{time.time() - t0:.1f}s]")
    write_json(out / "ensemble_config.json", ensemble_config_record(clf))
    write_json(out / "versions.json", versions_info(clf))

    spec = constraints = None
    if a.attack == "x-capgd":
        from tabfm_experiments.constraints import FeatureSpec, build_constraints

        meta = read_metadata(metadata_path_for(a.train, a.metadata), cols)
        spec = FeatureSpec.from_metadata(meta, device=device, scaler=a.scaler, X_train=Xc)
        dataset = infer_dataset_name(a.train) or ""
        constraints = build_constraints(dataset, meta, cols) if a.constraints == "full" else None
        n_rel = len(constraints.relation_constraints or []) if constraints is not None else 0
        log.info(f"  constraints={a.constraints} ({n_rel} relations, dataset '{dataset}')  scaler={a.scaler}  "
                 f"mutable {int(spec.mutable.sum())}/{len(cols)}  int {int(spec.is_int.sum())}  cat {int(spec.is_cat.sum())}")
        cfg = XCapgdConfig(norm=a.norm, eps=a.eps, eps_margin=a.eps_margin, n_iter=a.n_iter, momentum=a.momentum,
                           rho=a.rho, n_restarts=a.n_restarts, eot_iter=a.eot_iter, constraints_mode=a.constraints,
                           constraint_penalty=a.constraint_penalty, fix_equality_iter=a.fix_equality_iter,
                           random_start=a.random_start, budget_repair=a.budget_repair)

    if a.attack == "label-flip":
        ga_cfg = GAConfig(population=a.ga_population, generations=a.ga_generations, elite=a.ga_elite,
                          tournament=a.ga_tournament, crossover_prob=a.ga_crossover_prob,
                          mutation_rate=a.ga_mutation_rate, seed_full=a.ga_seed_full,
                          patience=a.ga_patience, batch_size=a.ga_batch_size,
                          fitness_test_rows=a.ga_fitness_test_rows)
        log.info(f"  GA: {ga_cfg}")

    train_df = pd.read_csv(a.train) if a.save_full_context else None
    target = [c for c in (train_df.columns if train_df is not None else []) if c not in cols]
    trial_dicts = []
    best_ctx = None          # strongest trial so far, by CE increase
    for run in range(a.n_runs):
        for sub in range(a.n_row_subsamples):
            tag = f"run{run}_sub{sub}"
            log.info(f"  -- trial {tag}")
            if a.attack == "x-capgd":
                rows = sample_row_indices(ytr, k, attack_class=a.attack_class, rng=row_rng(a.row_seed, sub))
                res = run_x_capgd(clf, Xc, yc, Xt, yt, rows, spec=spec, constraints=constraints, cfg=cfg,
                                  run_id=run, subsample_id=sub, attack_seed=attack_seed_for(a.attack_seed, run),
                                  clean_metrics=clean, k_requested=k, test_batch_size=a.test_batch_size, log=log)
                np.savez_compressed(out / "trials" / f"{tag}_delta.npz", row_indices=res.attacked_indices,
                                    cell_delta_raw=res.x_delta_raw, feature_names=np.array(cols))
            elif a.attack == "label-flip-random":
                # Test-agnostic: the rows depend only on the context labels and the RNG.
                rows = sample_row_indices(ytr, k, attack_class=a.attack_class,
                                          rng=random_flip_rng(a.row_seed, run, sub))
                res = run_label_flip_random(clf, Xc, yc, Xt, yt, rows, run_id=run, subsample_id=sub,
                                            clean_metrics=clean, k_requested=k,
                                            test_batch_size=a.test_batch_size, log=log)
                np.savez_compressed(out / "trials" / f"{tag}_delta.npz", flipped_indices=res.attacked_indices,
                                    y_clean=ytr.astype(np.int64), y_poisoned=res.y_poisoned)
            elif a.attack == "label-flip-influence":
                res = run_label_flip_influence(clf, Xc, yc, Xt, yt, k=k, attack_class=a.attack_class, run_id=run,
                                               subsample_id=sub, clean_metrics=clean,
                                               test_batch_size=a.test_batch_size, log=log)
                np.savez_compressed(out / "trials" / f"{tag}_delta.npz", flipped_indices=res.attacked_indices,
                                    y_clean=ytr.astype(np.int64), y_poisoned=res.y_poisoned)
            else:  # label-flip (genetic algorithm)
                pool = eligible_indices(ytr, a.attack_class)
                # Distinct GA population seed per (run, subsample): both add stochasticity.
                ga_seed = attack_seed_for(a.attack_seed, run) * 97 + sub
                res = run_label_flip_ga(clf, Xc, yc, Xt, yt, pool, k, cfg=ga_cfg, run_id=run, subsample_id=sub,
                                        ga_seed=ga_seed, clean_metrics=clean, seed_influence=not a.ga_no_influence_seed,
                                        test_batch_size=a.test_batch_size, log=log)
                np.savez_compressed(out / "trials" / f"{tag}_delta.npz", pool_indices=pool,
                                    flipped_indices=res.attacked_indices, y_clean=ytr.astype(np.int64),
                                    y_poisoned=res.y_poisoned, influence_topk=res.extra["influence_topk"]["indices"])
                inf = res.extra["influence_topk"]["delta"]
                log.info(f"     GA: {res.extra['n_flipped']}/{res.extra['budget_k']} flips from pool {res.extra['pool_size']}, "
                         f"{res.extra['n_fitness_evals']} evals, {res.extra['generations_run']} gens; "
                         f"influence-topk baseline dce={inf['ce']:+.5f} dacc={inf['accuracy']:+.4f}")
            d = res.to_json()
            write_json(out / "trials" / f"{tag}.json", d)
            trial_dicts.append({k_: v for k_, v in d.items() if k_ not in ("history", "restarts", "ga_history")})
            # Keep only the strongest trial's poisoned context (largest CE increase);
            # it is written once after the grid. Per-trial deltas are always in the .npz.
            if a.save_full_context and (best_ctx is None or res.delta["ce"] > best_ctx["delta_ce"]):
                best_ctx = {"tag": tag, "delta_ce": res.delta["ce"], "delta_accuracy": res.delta["accuracy"],
                            "attacked_indices": res.attacked_indices,
                            "x_delta_raw": res.x_delta_raw, "y_poisoned": res.y_poisoned}
            p = res.poisoned
            log.info(f"     poisoned ce={p.ce:.5f} auc={p.roc_auc:.4f} acc={p.accuracy:.4f} f1={p.f1:.4f}  "
                     f"dce={res.delta['ce']:+.5f} dacc={res.delta['accuracy']:+.4f}  "
                     f"k={res.k_actual}  [{res.seconds:.1f}s]")

    if a.save_full_context and best_ctx is not None:
        df = train_df.copy()
        if a.attack == "x-capgd":
            df[cols] = df[cols].astype("float64")
            idx = best_ctx["attacked_indices"]
            df.loc[idx, cols] = df.loc[idx, cols].to_numpy() + best_ctx["x_delta_raw"]
        else:
            df[target[0]] = best_ctx["y_poisoned"]
        df.to_csv(out / "context_poisoned.csv", index=False)
        log.info(f"  best trial {best_ctx['tag']}: dce={best_ctx['delta_ce']:+.5f} "
                 f"dacc={best_ctx['delta_accuracy']:+.4f}  -> context_poisoned.csv")

    # Per run (= CAPGD randomness) keep the strongest of its row subsamples by CE increase;
    # the headline aggregate is then across runs only. Flat all-trial stats are kept too.
    per_run_best = best_per_run(trial_dicts, "delta", "ce")

    summary = {
        "attack": a.attack, "train": a.train, "test": a.test, "n_context": n_ctx, "n_test": len(yte),
        "n_test_full": n_test_full, "n_features": len(cols), "k": k, "row_percent": a.row_percent,
        "attack_class": a.attack_class, "n_runs": a.n_runs, "n_row_subsamples": a.n_row_subsamples,
        "clean": clean.as_dict(), "clean_max_repeat_abs_dprob": max_dp, "chunking_exact": exact,
        "deterministic_mode": a.deterministic_mode,
        "seeds": {"model_seed": a.model_seed, "row_seed": a.row_seed, "attack_seed": a.attack_seed},
        "aggregation": (f"mean over {a.n_runs} independent random draws (test-agnostic; no selection)"
                        if a.attack == "label-flip-random" else
                        f"best row-subsample per run by delta.ce, then aggregated across "
                        f"the {a.n_runs} runs"),
        "per_run_best": per_run_best,
        "aggregate_delta": summarize_trials(per_run_best, "delta"),
        "aggregate_poisoned": summarize_trials(per_run_best, "poisoned"),
        "aggregate_delta_all_trials": summarize_trials(trial_dicts, "delta"),
        "aggregate_poisoned_all_trials": summarize_trials(trial_dicts, "poisoned"),
        "best_context_trial": ({k_: best_ctx[k_] for k_ in ("tag", "delta_ce", "delta_accuracy")}
                               if best_ctx is not None else None),
        "trials": trial_dicts,
    }
    write_json(out / "summary.json", summary)
    agg = summary["aggregate_delta"]["overall"]
    log.info("  per-run winners: " + ", ".join(
        f"run{t['run_id']}<-sub{t['subsample_id']} (dce={t['delta']['ce']:+.5f})" for t in per_run_best))
    how = ("random draws (no selection)" if a.attack == "label-flip-random"
           else "runs (best subsample per run)")
    log.info(f"  mean over {len(per_run_best)} {how}: "
             f"dce={agg['ce']['mean']:+.5f} dacc={agg['accuracy']['mean']:+.4f} "
             f"dauc={agg['roc_auc']['mean'] if agg['roc_auc']['mean'] is not None else float('nan'):+.4f}  -> {out}")


if __name__ == "__main__":
    main()

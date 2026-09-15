import numpy as np
import pytest

from tabfm_experiments.ga import GAConfig, run_ga


def _hamming_fitness(target, calls):
    def f(masks):
        calls.append(masks.copy())
        return -(masks != target).sum(1).astype(float)
    return f


def test_ga_improves_over_initial_and_is_monotone():
    n = 40
    target = np.random.default_rng(123).random(n) < 0.3
    calls = []
    cfg = GAConfig(population=20, generations=40, patience=0, mutation_rate=1 / n)
    res = run_ga(_hamming_fitness(target, calls), n, cfg, np.random.default_rng(0))
    assert res["best_fitness"] > res["initial_best_fitness"]
    so_far = [h["best_so_far"] for h in res["history"]]
    assert all(b >= a for a, b in zip(so_far, so_far[1:]))
    assert res["best_mask"].any()
    # cache: every mask handed to the fitness function is new
    seen = np.concatenate(calls)
    assert len({m.tobytes() for m in seen}) == len(seen) == res["n_evals"]


def test_budget_cap_and_seed_masks_are_repaired():
    n, k = 50, 8
    calls = []
    seed = np.zeros(n, dtype=bool)
    seed[:20] = True  # over budget; must be trimmed to k
    cfg = GAConfig(population=10, generations=6, mutation_rate=0.2)
    res = run_ga(_hamming_fitness(np.ones(n, dtype=bool), calls), n, cfg,
                 np.random.default_rng(1), max_active=k, seed_masks=[seed])
    for masks in calls:
        assert masks.sum(1).max() <= k and masks.sum(1).min() >= 1
    assert res["best_mask"].sum() <= k


def test_ga_seeds_and_is_reproducible():
    n = 25
    target = np.zeros(n, dtype=bool)
    calls = []
    cfg = GAConfig(population=8, generations=5)
    seed = np.zeros(n, dtype=bool); seed[3] = True
    res1 = run_ga(_hamming_fitness(target, calls), n, cfg, np.random.default_rng(7),
                  max_active=5, seed_masks=[seed])
    assert calls[0][0][3] and calls[0][0].sum() == 1  # the injected seed mask is evaluated first
    res2 = run_ga(_hamming_fitness(target, []), n, cfg, np.random.default_rng(7), max_active=5, seed_masks=[seed])
    res3 = run_ga(_hamming_fitness(target, []), n, cfg, np.random.default_rng(8), max_active=5, seed_masks=[seed])
    assert np.array_equal(res1["best_mask"], res2["best_mask"]) and res1["history"] == res2["history"]
    assert res1["history"] != res3["history"]


def test_generations_zero_is_random_search_and_early_stop():
    n = 10
    res = run_ga(lambda m: -(m != np.ones(n, dtype=bool)).sum(1).astype(float), n,
                 GAConfig(population=6, generations=0), np.random.default_rng(0), max_active=n)
    assert res["generations_run"] == 0 and res["best_fitness"] == 0.0  # all-ones mask is the target
    res = run_ga(lambda m: np.zeros(len(m)), n, GAConfig(population=6, generations=50, patience=3),
                 np.random.default_rng(0))
    assert res["generations_run"] == 3


def test_bad_config():
    with pytest.raises(ValueError):
        run_ga(lambda m: np.zeros(len(m)), 5, GAConfig(population=4, elite=4), np.random.default_rng(0))

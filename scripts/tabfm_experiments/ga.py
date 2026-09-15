"""Genetic search over which rows of a parent subset to flip (model-agnostic core).

An individual is a boolean mask over the ``k`` parent rows (True = flip that label).
The fitness function is injected so the search can be unit-tested without TabPFN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GAConfig:
    population: int = 16
    generations: int = 20            # 0 = random search over `population` masks
    elite: int = 2                   # best individuals copied unchanged each generation
    tournament: int = 3              # tournament size for parent selection
    crossover_prob: float = 0.9      # uniform crossover probability
    mutation_rate: float | None = None  # per-bit flip probability (default ~1/pool, set by run_label_flip_ga)
    seed_full: bool = True           # include the all-ones mask (subject to the budget cap) in generation 0
    patience: int = 8                # stop after this many generations without improvement (0 = never)
    batch_size: int = 16             # individuals per fused TabPFN forward (halved automatically on OOM)
    fitness_test_rows: int | None = None  # score fitness on a random subset of test rows (None = all)


def run_ga(
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    n: int,
    cfg: GAConfig,
    rng: np.random.Generator,
    *,
    max_active: int | None = None,
    seed_masks: list[np.ndarray] | None = None,
    on_generation: Callable[[dict], None] | None = None,
) -> dict:
    """Maximise ``fitness_fn`` over non-empty boolean masks of length ``n``.

    ``max_active`` caps the number of True bits (the flip budget); masks are repaired
    to satisfy it after every operator. ``seed_masks`` are injected into generation 0
    (e.g. the influence top-k mask). ``fitness_fn(masks [P, n] bool) -> [P] float`` is
    only called on unseen masks (cached), so ``n_evals`` counts distinct evaluations.
    With elitism the best-so-far fitness never decreases.
    """
    if n < 1:
        raise ValueError("pool must contain at least one row")
    if cfg.population < 2 or not 0 <= cfg.elite < cfg.population or cfg.tournament < 1:
        raise ValueError("need population >= 2, 0 <= elite < population, tournament >= 1")
    cap = n if max_active is None else int(max_active)
    if cap < 1:
        raise ValueError("max_active must be >= 1")
    mut = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / n

    cache: dict[bytes, float] = {}

    def evaluate(pop: np.ndarray) -> np.ndarray:
        keys = [np.packbits(m).tobytes() for m in pop]
        new: dict[bytes, np.ndarray] = {}
        for key, m in zip(keys, pop):
            if key not in cache and key not in new:
                new[key] = m
        if new:
            vals = np.asarray(fitness_fn(np.stack(list(new.values()))), dtype=float)
            for key, v in zip(new, vals):
                cache[key] = float(v)
        return np.array([cache[key] for key in keys])

    def repair(m: np.ndarray) -> np.ndarray:
        """Ensure 1 <= count <= cap: trim random excess, or add one if empty."""
        on = np.flatnonzero(m)
        if len(on) == 0:
            m[rng.integers(n)] = True
        elif len(on) > cap:
            m[rng.permutation(on)[cap:]] = False
        return m

    def random_mask() -> np.ndarray:
        size = int(rng.integers(max(1, cap // 2), cap + 1)) if cap > 1 else 1
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, size=size, replace=False)] = True
        return m

    pop = []
    for sm in (seed_masks or []):
        pop.append(repair(np.asarray(sm, dtype=bool).copy()))
    if cfg.seed_full:
        base = np.ones(n, dtype=bool) if cap >= n else np.zeros(n, dtype=bool)
        pop.append(repair(base.copy()))
    while len(pop) < cfg.population:
        pop.append(random_mask())
    pop = np.stack(pop[: cfg.population])
    fit = evaluate(pop)

    b = int(np.argmax(fit))
    best_mask, best_fit = pop[b].copy(), float(fit[b])
    initial_best = best_fit
    history = []

    def record(gen: int) -> None:
        rec = {"generation": gen, "best": float(fit.max()), "mean": float(fit.mean()),
               "best_so_far": best_fit, "best_size": int(best_mask.sum()), "n_evals": len(cache)}
        history.append(rec)
        if on_generation is not None:
            on_generation(rec)

    record(0)

    def tournament() -> np.ndarray:
        idx = rng.integers(0, len(pop), size=cfg.tournament)
        return pop[idx[np.argmax(fit[idx])]]

    stall, gens_run = 0, 0
    for gen in range(1, cfg.generations + 1):
        order = np.argsort(-fit, kind="stable")
        children = [pop[i].copy() for i in order[: cfg.elite]]
        while len(children) < cfg.population:
            p1, p2 = tournament(), tournament()
            if rng.random() < cfg.crossover_prob:
                child = np.where(rng.random(n) < 0.5, p1, p2)
            else:
                child = p1.copy()
            child = child ^ (rng.random(n) < mut)
            children.append(repair(child))
        pop = np.stack(children)
        fit = evaluate(pop)
        gens_run = gen
        b = int(np.argmax(fit))
        if fit[b] > best_fit + 1e-12:
            best_mask, best_fit, stall = pop[b].copy(), float(fit[b]), 0
        else:
            stall += 1
        record(gen)
        if cfg.patience and stall >= cfg.patience:
            break

    return {"best_mask": best_mask, "best_fitness": best_fit, "initial_best_fitness": initial_best,
            "history": history, "n_evals": len(cache), "generations_run": gens_run}

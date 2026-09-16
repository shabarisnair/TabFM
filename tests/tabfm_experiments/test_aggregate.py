import pytest

from tabfm_experiments.aggregate import best_per_run, summarize_trials


def _trials():
    # accuracy delta = run*10 + sub
    return [
        {"run_id": r, "subsample_id": s, "delta": {"accuracy": r * 10 + s, "ce": 1.0}}
        for r in (0, 1) for s in (0, 1)
    ]


def test_overall_and_grouped_means():
    out = summarize_trials(_trials(), key="delta")
    assert out["n_trials"] == 4
    assert out["overall"]["accuracy"]["mean"] == pytest.approx(5.5)
    assert out["by_run"]["0"]["accuracy"]["mean"] == pytest.approx(0.5)
    assert out["by_run"]["1"]["accuracy"]["mean"] == pytest.approx(10.5)
    assert out["by_subsample"]["0"]["accuracy"]["mean"] == pytest.approx(5.0)
    assert out["by_subsample"]["1"]["accuracy"]["mean"] == pytest.approx(6.0)
    # std across run means (hierarchical view) vs std across all 4 trials
    assert out["across_runs"]["accuracy"]["std"] == pytest.approx(5.0)
    assert out["overall"]["accuracy"]["std"] == pytest.approx(5.024937810560445)
    assert out["overall"]["ce"]["std"] == 0.0


def test_best_per_run_picks_max_ce_and_breaks_ties_low():
    # run0 subs ce = [0.1, 0.5, 0.3] -> sub1 ;  run1 = [0.7, 0.2, 0.7] -> sub0 (tie -> lowest sub)
    ce = {(0, 0): 0.1, (0, 1): 0.5, (0, 2): 0.3, (1, 0): 0.7, (1, 1): 0.2, (1, 2): 0.7}
    trials = [{"run_id": r, "subsample_id": s_, "delta": {"ce": ce[(r, s_)], "accuracy": -ce[(r, s_)]}}
              for r in (0, 1) for s_ in (0, 1, 2)]
    best = best_per_run(trials)
    assert [(t["run_id"], t["subsample_id"]) for t in best] == [(0, 1), (1, 0)]
    assert [t["delta"]["ce"] for t in best] == [0.5, 0.7]
    # aggregating the winners is then across runs only
    agg = summarize_trials(best, "delta")
    assert agg["n_trials"] == 2
    assert agg["overall"]["ce"]["mean"] == pytest.approx(0.6)
    # and it is >= the flat mean over all trials (it is a maximum)
    assert agg["overall"]["ce"]["mean"] >= summarize_trials(trials, "delta")["overall"]["ce"]["mean"]


def test_best_per_run_single_subsample_is_identity():
    trials = [{"run_id": r, "subsample_id": 0, "delta": {"ce": 0.1 * r}} for r in range(4)]
    assert best_per_run(trials) == trials

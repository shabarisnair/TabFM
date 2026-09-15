import pytest

from tabfm_experiments.aggregate import summarize_trials


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

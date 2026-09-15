import numpy as np
import pytest

from tabfm_experiments.sampling import k_from_percent, row_rng, sample_row_indices


def test_k_from_percent():
    assert k_from_percent(100, 5) == 5
    assert k_from_percent(1000, 5) == 50
    assert k_from_percent(10, 5) == 1  # 0.5 rounds half up, and never below 1
    assert k_from_percent(3, 0.1) == 1
    assert k_from_percent(50, 100) == 50


def test_attack_class_restricts_rows():
    y = np.array([0, 1] * 50)
    idx = sample_row_indices(y, 10, attack_class=1, rng=np.random.default_rng(0))
    assert len(idx) == 10
    assert (y[idx] == 1).all()


def test_unique_and_seed_dependence():
    y = np.zeros(200, dtype=int)
    a = sample_row_indices(y, 20, attack_class=None, rng=row_rng(1, 0))
    b = sample_row_indices(y, 20, attack_class=None, rng=row_rng(1, 1))
    a2 = sample_row_indices(y, 20, attack_class=None, rng=row_rng(1, 0))
    assert len(np.unique(a)) == 20
    assert not np.array_equal(a, b)
    assert np.array_equal(a, a2)


def test_not_enough_eligible_uses_all():
    y = np.array([0] * 20 + [1] * 3)
    idx = sample_row_indices(y, 10, attack_class=1, rng=np.random.default_rng(0))
    assert sorted(idx.tolist()) == [20, 21, 22]


def test_bad_attack_class():
    with pytest.raises(ValueError):
        sample_row_indices(np.array([0, 1]), 1, attack_class=2, rng=np.random.default_rng(0))

import numpy as np
import pandas as pd
import pytest

import select_contexts as sc
from tabfm_experiments.config import DATASETS


def _raw():
    return pd.DataFrame({
        "a": [1, 2, 2, 3, 4, 5, 6, 7],
        "b": [0.5, 1.5, 1.5, np.nan, 2.0, 2.5, 3.0, 3.5],
        "t": [0, 1, 1, 0, 1, 0, 1, 0],
    })


def test_dedup_and_reserved_exclusion():
    raw = _raw()
    val = pd.DataFrame({"a": [3.0], "b": [np.nan], "t": [0]})       # matches raw row 3 (float dtype)
    test = pd.DataFrame({"a": [7], "b": [3.5], "t": [0]})             # matches raw row 7
    pool, fp, stats = sc.build_pool(raw, [val, test])
    assert stats["duplicate_count"] == 1
    assert stats["raw_rows_matching_reserved"] == 2
    assert stats["pool_size"] == 5
    assert sorted(pool["a"].tolist()) == [1, 2, 4, 5, 6]
    assert len(set(fp)) == len(fp)
    assert list(pool.index) == list(range(5))


def test_lcld_prepare_drops_issue_d():
    assert "issue_d" in DATASETS["lcld_v2"].drop_columns
    raw = pd.DataFrame({"loan_amnt": [1.0], "issue_d": ["2015-01-01"], "charged_off": [0]})
    out = sc.prepare_frame(raw, DATASETS["lcld_v2"].drop_columns, ["loan_amnt", "charged_off"])
    assert "issue_d" not in out.columns
    assert list(out.columns) == ["loan_amnt", "charged_off"]


def test_candidate_sizes():
    y = np.array([0] * 70 + [1] * 30)
    assert sc.candidate_size("natural", y, 80) == 80
    assert sc.candidate_size("natural", y, 500) == 100
    assert sc.candidate_size("balanced", y, 80) == 60


def test_balanced_candidate_and_children_are_nested_and_50_50():
    rng_y = np.random.default_rng(0)
    y = (rng_y.random(30000) < 0.3).astype(int)
    idx = sc.draw_candidate(y, 12000, "balanced", np.random.default_rng(1))
    assert len(np.unique(idx)) == 12000
    assert np.bincount(y[idx]).tolist() == [6000, 6000]
    children, skipped = sc.nested_children(idx, y, (5000, 1000), "balanced", np.random.default_rng(2))
    assert not skipped
    assert np.bincount(y[children[5000]]).tolist() == [2500, 2500]
    assert np.bincount(y[children[1000]]).tolist() == [500, 500]
    assert set(children[1000]) <= set(children[5000]) <= set(idx)


def test_natural_nesting_and_skips():
    y = np.zeros(20000, dtype=int)
    idx = sc.draw_candidate(y, 8000, "natural", np.random.default_rng(0))
    children, skipped = sc.nested_children(idx, y, (5000, 1000), "natural", np.random.default_rng(1))
    assert set(children[1000]) <= set(children[5000]) <= set(idx)
    small = sc.draw_candidate(y, 3000, "natural", np.random.default_rng(0))
    ch2, sk2 = sc.nested_children(small, y, (5000, 1000), "natural", np.random.default_rng(1))
    assert 5000 in sk2 and 1000 in ch2
    assert set(ch2[1000]) <= set(small)


def test_candidate_seeds_reproducible():
    y = np.array([0, 1] * 500)
    a = sc.draw_candidate(y, 400, "natural", np.random.default_rng(3))
    b = sc.draw_candidate(y, 400, "natural", np.random.default_rng(3))
    c = sc.draw_candidate(y, 400, "natural", np.random.default_rng(4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)

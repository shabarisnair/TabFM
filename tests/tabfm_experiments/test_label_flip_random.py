import json

import numpy as np
import pandas as pd
import pytest
import torch

from tabfm_experiments.sampling import k_from_percent, random_flip_rng, sample_row_indices

pytest.importorskip("tabpfn")

from tabfm_experiments.attacks import run_label_flip_random  # noqa: E402
from tabfm_experiments.metrics import binary_metrics  # noqa: E402
from tabfm_experiments.runtime import build_tabpfn_v2, evaluate_context  # noqa: E402


def _draw(y, k, run, sub=0, seed=1, attack_class=None):
    return sample_row_indices(y, k, attack_class=attack_class, rng=random_flip_rng(seed, run, sub))


# ------------------------------------------------------------------ row choice

def test_each_run_flips_a_different_random_subset():
    y = np.array([0, 1] * 500)
    draws = [tuple(_draw(y, 100, run)) for run in range(5)]
    assert len(set(draws)) == 5
    assert all(len(d) == 100 and len(set(d)) == 100 for d in draws)


def test_draws_are_reproducible_from_the_seed():
    y = np.array([0, 1] * 500)
    assert np.array_equal(_draw(y, 50, run=3), _draw(y, 50, run=3))
    assert not np.array_equal(_draw(y, 50, run=3, seed=1), _draw(y, 50, run=3, seed=2))


def test_attack_class_restricts_the_draw():
    y = np.array([0] * 900 + [1] * 100)
    rows = _draw(y, 40, run=0, attack_class=1)
    assert set(y[rows]) == {1}


def test_draw_is_roughly_uniform_over_the_context():
    # With a fresh draw per run, every row should be hit at about the nominal rate.
    y = np.zeros(200, dtype=int)
    hits = np.zeros(200)
    for run in range(2000):
        hits[_draw(y, 20, run)] += 1
    assert abs(hits.mean() / 2000 - 0.10) < 1e-12        # exactly k/n on average
    assert hits.min() > 140 and hits.max() < 270          # no row is systematically favoured


# ------------------------------------------------------------------ one trial

def test_trial_flips_exactly_the_given_rows():
    clf = build_tabpfn_v2("cpu", 0)
    g = torch.Generator().manual_seed(4)
    X = torch.randn(40, 3, generator=g)
    y = (X[:, 0] > 0).float()
    Xt = torch.randn(12, 3, generator=g)
    yt = (Xt[:, 0] > 0).long()
    _, _, p = evaluate_context(clf, X, y, Xt, yt, need_grad=False)
    clean = binary_metrics(p.numpy(), yt.numpy())
    rows = np.array([2, 9, 17, 30])
    res = run_label_flip_random(clf, X, y, Xt, yt, rows, run_id=0, subsample_id=0,
                                clean_metrics=clean, k_requested=4)
    changed = np.flatnonzero(res.y_poisoned != y.numpy().astype(int))
    assert np.array_equal(changed, rows)
    assert res.k_actual == 4 and res.extra["test_agnostic"] is True
    assert res.delta["ce"] == pytest.approx(res.poisoned.ce - clean.ce)


# ------------------------------------------------------------------ CLI, end to end

def _write(tmp_path, name, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).round(5)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y"] = (X[:, 0] > 0).astype(int)
    df.to_csv(tmp_path / name, index=False)
    return tmp_path / name


def _flips(out):
    return {p.name: np.load(p)["flipped_indices"].tolist() for p in sorted(out.glob("trials/*_delta.npz"))}


def test_cli_is_test_agnostic(tmp_path):
    """Same context and seeds, two unrelated test sets -> the very same rows are flipped."""
    import attack_context

    train = _write(tmp_path, "train.csv", 60, seed=0)
    test_a = _write(tmp_path, "test_a.csv", 15, seed=1)
    test_b = _write(tmp_path, "test_b.csv", 25, seed=2)
    common = ["--gpu", "cpu", "--attack", "label-flip-random", "--train", str(train),
              "--target", "y", "--row-percent", "20", "--n-runs", "3"]
    attack_context.main([*common, "--test", str(test_a), "--out", str(tmp_path / "oa")])
    attack_context.main([*common, "--test", str(test_b), "--out", str(tmp_path / "ob")])

    fa, fb = _flips(tmp_path / "oa"), _flips(tmp_path / "ob")
    assert fa == fb                                           # choice ignores the test set
    assert len({tuple(v) for v in fa.values()}) == 3          # one fresh draw per run
    k = k_from_percent(60, 20)
    assert all(len(v) == k for v in fa.values())

    s = json.loads((tmp_path / "oa" / "summary.json").read_text())
    assert s["attack"] == "label-flip-random"
    assert len(s["per_run_best"]) == 3                        # every draw kept, no selection
    assert "no selection" in s["aggregation"]


def test_cli_rejects_subsample_selection(tmp_path):
    import attack_context

    train = _write(tmp_path, "train.csv", 30, seed=0)
    test = _write(tmp_path, "test.csv", 10, seed=1)
    with pytest.raises(SystemExit, match="test-agnostic"):
        attack_context.parse_args(["--gpu", "cpu", "--attack", "label-flip-random", "--train", str(train),
                                   "--test", str(test), "--out", str(tmp_path / "o"),
                                   "--n-row-subsamples", "3"])

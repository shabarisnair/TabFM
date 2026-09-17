import json

import numpy as np
import pandas as pd
import pytest

from tabfm_experiments.metrics import (
    balanced_accuracy_from_counts,
    binary_metrics,
    mcc_from_counts,
    with_derived,
)
from tabfm_experiments.transfer import (
    agg,
    apply_trial_delta,
    load_attack_contexts,
    per_run_best_tags,
    relative,
    tabpfn_run_metrics,
)

xgboost = pytest.importorskip("xgboost")
pytest.importorskip("optuna")


def _clean_frame(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)).round(6), columns=[f"f{i}" for i in range(d)])
    df["y"] = (df["f0"] > 0).astype(int)
    return df


# --------------------------------------------------------------- derived metrics

def test_derived_metrics_match_sklearn():
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.normal(0.5 + 0.3 * (y - 0.5), 0.3, 400), 1e-6, 1 - 1e-6)
    m = with_derived(binary_metrics(np.stack([1 - p, p], 1), y))
    pred = (p > 0.5).astype(int)
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))


def test_derived_metrics_on_degenerate_model():
    # coil2000's clean model predicts one class only: MCC is 0, balanced accuracy 0.5.
    assert mcc_from_counts(tn=900, fp=0, fn=100, tp=0) == 0.0
    assert balanced_accuracy_from_counts(tn=900, fp=0, fn=100, tp=0) == pytest.approx(0.5)


def test_with_derived_reads_counts_from_a_plain_dict():
    d = with_derived({"ce": 0.5, "tn": 80, "fp": 20, "fn": 10, "tp": 90})
    assert d["ce"] == 0.5
    assert d["balanced_accuracy"] == pytest.approx(0.5 * (90 / 100 + 80 / 100))


# --------------------------------------------------------------- reconstruction

def test_apply_trial_delta_label_flip_touches_only_the_target():
    df = _clean_frame()
    y_pois = df["y"].to_numpy().copy()
    y_pois[[1, 3, 5]] = 1 - y_pois[[1, 3, 5]]
    out = apply_trial_delta(df, "y", {"y_poisoned": y_pois})
    assert np.array_equal(out["y"].to_numpy(), y_pois)
    pd.testing.assert_frame_equal(out.drop(columns="y"), df.drop(columns="y"))
    assert df["y"].tolist() != out["y"].tolist()      # the input frame is not mutated


def test_apply_trial_delta_x_capgd_adds_cells_on_the_named_rows_only():
    df = _clean_frame()
    cols = ["f0", "f2"]
    rows = np.array([2, 7, 11])
    delta = np.arange(6, dtype=np.float32).reshape(3, 2) + 1.0
    out = apply_trial_delta(df, "y", {"feature_names": np.array(cols), "row_indices": rows,
                                      "cell_delta_raw": delta})
    assert np.allclose(out.loc[rows, cols].to_numpy(), df.loc[rows, cols].to_numpy() + delta)
    untouched = out.drop(index=rows)
    pd.testing.assert_frame_equal(untouched, df.drop(index=rows), check_dtype=False)
    assert np.array_equal(out["y"], df["y"])          # labels untouched by a feature attack


def test_apply_trial_delta_rejects_an_unknown_archive():
    with pytest.raises(ValueError, match="neither"):
        apply_trial_delta(_clean_frame(), "y", {"something_else": np.zeros(3)})


def _fake_attack_dir(tmp_path, attack="label-flip"):
    """A minimal attack run: summary.json with per_run_best plus the matching npz deltas."""
    df = _clean_frame()
    d = tmp_path / "attack"
    (d / "trials").mkdir(parents=True)
    per_run_best = []
    for run in range(3):
        sub = run % 2                                  # winners are not all sub0
        tag = f"run{run}_sub{sub}"
        if attack == "label-flip":
            y = df["y"].to_numpy().copy()
            y[run] = 1 - y[run]
            np.savez(d / "trials" / f"{tag}_delta.npz", y_clean=df["y"].to_numpy(), y_poisoned=y)
        else:
            np.savez(d / "trials" / f"{tag}_delta.npz", row_indices=np.array([run]),
                     cell_delta_raw=np.full((1, 4), 0.1 * (run + 1), dtype=np.float32),
                     feature_names=np.array([f"f{i}" for i in range(4)]))
        per_run_best.append({
            "run_id": run, "subsample_id": sub,
            "clean": {"ce": 0.4, "roc_auc": 0.7, "accuracy": 0.8, "f1": 0.1,
                      "tn": 790, "fp": 10, "fn": 190, "tp": 10},
            "poisoned": {"ce": 0.4 + 0.01 * run, "roc_auc": 0.65, "accuracy": 0.78, "f1": 0.05,
                         "tn": 780, "fp": 20, "fn": 195, "tp": 5},
        })
    (d / "summary.json").write_text(json.dumps({
        "attack": attack, "row_percent": 16.0, "k": 800, "n_runs": 3, "n_row_subsamples": 2,
        "aggregation": "best row-subsample per run", "per_run_best": per_run_best}))
    return d, df


def test_load_attack_contexts_uses_the_recorded_per_run_winners(tmp_path):
    d, df = _fake_attack_dir(tmp_path)
    contexts, summary = load_attack_contexts(d, df, "y")
    assert [t for t, _ in contexts] == ["run0_sub0", "run1_sub1", "run2_sub0"]
    assert per_run_best_tags(summary) == [t for t, _ in contexts]
    for i, (_, ctx) in enumerate(contexts):
        changed = np.flatnonzero(ctx["y"].to_numpy() != df["y"].to_numpy())
        assert changed.tolist() == [i]


def test_tabpfn_run_metrics_adds_derived_and_subtracts_per_run(tmp_path):
    d, df = _fake_attack_dir(tmp_path)
    _, summary = load_attack_contexts(d, df, "y")
    tp = tabpfn_run_metrics(summary)
    assert len(tp["delta"]) == 3
    assert tp["delta"][2]["ce"] == pytest.approx(0.02)
    # derived metrics come from the counts, and the delta is poisoned - clean
    assert tp["delta"][0]["mcc"] == pytest.approx(tp["poisoned"][0]["mcc"] - tp["clean"][0]["mcc"])
    assert agg(tp["delta"])["ce"]["mean"] == pytest.approx(0.01)


def test_relative_guards_a_zero_baseline():
    assert relative(0.02, 0.4) == pytest.approx(5.0)
    assert relative(0.02, 0.0) is None                 # coil2000's clean MCC is exactly 0
    assert relative(None, 0.4) is None


# --------------------------------------------------------------- end to end

def test_xgb_transfer_end_to_end(tmp_path):
    import xgb_transfer

    rng = np.random.default_rng(5)
    def frame(n):
        X = rng.normal(size=(n, 4))
        df = pd.DataFrame(X.round(6), columns=[f"f{i}" for i in range(4)])
        df["y"] = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
        return df

    train, test, val = frame(300), frame(120), frame(120)
    pois = train.copy()
    pois.loc[:149, "y"] = 1 - pois.loc[:149, "y"]      # half the labels flipped
    for name, df in (("train", train), ("test", test), ("val", val), ("pois", pois)):
        df.to_csv(tmp_path / f"{name}.csv", index=False)

    out = xgb_transfer.main([
        "--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
        "--val", str(tmp_path / "val.csv"), "--poisoned-train", str(tmp_path / "pois.csv"),
        "--target", "y", "--out", str(tmp_path / "out"), "--no-hpo",
        "--n-clean-repeats", "2", "--n-jobs", "1", "--early-stopping-rounds", "5",
        "--n-estimators", "30"])

    assert set(out["protocols"]) == {"A", "B"}
    for p, r in out["protocols"].items():
        assert len(r["clean"]) == 2 and len(r["poisoned"]) == 1
        # flipping half the labels must cost accuracy under either protocol
        assert r["delta_agg"]["accuracy"]["mean"] < 0, p
        assert r["delta_agg"]["ce"]["mean"] > 0, p
    # protocol A holds out 20% of the context, B trains on all of it
    assert out["protocols"]["A"]["clean"][0]["n_train"] == 240
    assert out["protocols"]["B"]["clean"][0]["n_train"] == 300
    assert out["protocols"]["B"]["clean"][0]["n_val"] == 120
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["n_features"] == 4


def test_clean_cache_is_reused(tmp_path):
    import xgb_transfer

    rng = np.random.default_rng(7)
    def frame(n):
        X = rng.normal(size=(n, 3))
        df = pd.DataFrame(X.round(6), columns=list("abc"))
        df["y"] = (X[:, 0] > 0).astype(int)
        return df

    train, test, val = frame(200), frame(80), frame(80)
    for name, df in (("train", train), ("test", test), ("val", val)):
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    common = ["--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
              "--val", str(tmp_path / "val.csv"), "--poisoned-train", str(tmp_path / "train.csv"),
              "--target", "y", "--no-hpo", "--protocols", "A", "--n-jobs", "1",
              "--n-estimators", "20", "--early-stopping-rounds", "5",
              "--clean-cache-dir", str(tmp_path / "cache")]
    first = xgb_transfer.main([*common, "--out", str(tmp_path / "o1")])
    assert len(list((tmp_path / "cache").glob("clean_*.json"))) == 1
    second = xgb_transfer.main([*common, "--out", str(tmp_path / "o2")])
    assert first["protocols"]["A"]["clean"] == second["protocols"]["A"]["clean"]

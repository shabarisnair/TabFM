import numpy as np
import pandas as pd
import pytest
import torch

from tabfm_experiments.config import DATASETS
from tabfm_experiments.constraints import (
    FeatureSpec,
    build_constraints,
    fix_immutable_raw,
    fix_types_raw,
    relation_penalty_fn,
    repair_end,
)
from tabfm_experiments.data import read_metadata


def _meta(rows):
    return pd.DataFrame(rows, columns=["feature", "type", "mutable", "min", "max"])


def test_fix_types_int_truncates_delta_cat_rounds_value():
    x_clean = torch.tensor([[1.0, 2.0, 0.5]])
    x_adv = torch.tensor([[4.7, 1.4, 0.77]])
    is_int = torch.tensor([True, False, False])
    is_cat = torch.tensor([False, True, False])
    out = fix_types_raw(x_clean, x_adv, is_int, is_cat)
    assert out[0, 0] == pytest.approx(4.0)   # delta 3.7 -> 3.0
    assert out[0, 1] == pytest.approx(1.0)   # cat 1.4 -> 1.0
    assert out[0, 2] == pytest.approx(0.77)  # real untouched
    neg = fix_types_raw(torch.tensor([[5.0]]), torch.tensor([[4.4]]), torch.tensor([True]), torch.tensor([False]))
    assert neg[0, 0] == pytest.approx(5.0)   # delta -0.6 -> -0.0 (toward zero)


def test_fix_types_no_int_early_return_leaves_cats():
    # Reference quirk: with no int features, categoricals are NOT rounded.
    out = fix_types_raw(torch.tensor([[1.0]]), torch.tensor([[1.4]]), torch.tensor([False]), torch.tensor([True]))
    assert out[0, 0] == pytest.approx(1.4)


def test_fix_immutable_restores():
    out = fix_immutable_raw(torch.tensor([[1.0, 2.0]]), torch.tensor([[9.0, 9.0]]), torch.tensor([True, False]))
    assert out.tolist() == [[9.0, 2.0]]


def test_scaler_round_trip_and_constant_features():
    meta = _meta([("a", "real", True, -2.0, 6.0), ("b", "int", True, 3.0, 3.0)])
    spec = FeatureSpec.from_metadata(meta, device="cpu")
    x = torch.tensor([[0.0, 3.0], [6.0, 3.0], [-2.0, 3.0]])
    s = spec.to_scaled(x)
    assert s[:, 0].tolist() == pytest.approx([0.25, 1.0, 0.0])
    assert s[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0])  # range 1 for constant
    assert torch.allclose(spec.to_raw(s), x)
    tr = FeatureSpec.from_metadata(meta, device="cpu", scaler="train", X_train=torch.tensor([[0.0, 3.0], [4.0, 3.0]]))
    assert tr.to_scaled(torch.tensor([[2.0, 3.0]]))[0, 0] == pytest.approx(0.5)


def test_repair_end_types_immutable():
    meta = _meta([("a", "int", True, 0, 10), ("b", "cat", True, 0, 3), ("c", "real", False, 0, 1)])
    cons = build_constraints("none", meta, ["a", "b", "c"])
    x_clean = torch.tensor([[1.0, 1.0, 0.5]])
    x_adv = torch.tensor([[4.7, 1.6, 0.9]])
    out = repair_end(x_clean, x_adv, cons)
    assert torch.allclose(out, torch.tensor([[4.0, 2.0, 0.5]]))


def test_budgeted_repair_converges_when_rounding_pushes_out():
    """Categorical round() moves values AWAY from clean, so shrinking to exactly eps
    oscillates above the ball (the WiDS failure mode). The repair must still land inside."""
    from tabfm_experiments.constraints import FeatureSpec, repair_end, repair_end_budgeted

    eps = 0.475
    # Two categoricals with range 2 => one unit = 0.5 scaled; a single rounded step is
    # already 0.5 and two are 0.707, so rounding pushes back out of the ball.
    # An int column is included because fix_types early-returns (and never rounds cats)
    # when a schema has no int features -- all four real datasets have int features.
    meta = _meta([("a", "cat", True, 0.0, 2.0), ("b", "cat", True, 0.0, 2.0),
                  ("c", "int", True, 0.0, 100.0)])
    spec = FeatureSpec.from_metadata(meta, device="cpu")
    cons = build_constraints("none", meta, ["a", "b", "c"])
    x_clean = torch.tensor([[0.0, 0.0, 0.0]])
    x_adv = torch.tensor([[2.0, 2.0, 50.0]])

    std = repair_end(x_clean, x_adv, cons)
    n_std = float((spec.to_scaled(std) - spec.to_scaled(x_clean)).pow(2).sum(1).sqrt().max())
    assert n_std > eps  # the plain repair is out of budget

    bud = repair_end_budgeted(x_clean, x_adv, spec, cons, eps)
    n_bud = float((spec.to_scaled(bud) - spec.to_scaled(x_clean)).pow(2).sum(1).sqrt().max())
    assert n_bud <= eps + 1e-5, f"budgeted repair left the ball: {n_bud}"
    assert torch.equal(bud, bud.round())  # still categorical-valid


def test_budgeted_repair_keeps_feasible_perturbations():
    """A perturbation that already fits must be left alone (no needless shrinking)."""
    from tabfm_experiments.constraints import FeatureSpec, repair_end_budgeted

    meta = _meta([("a", "int", True, 0.0, 100.0), ("b", "real", True, 0.0, 1.0)])
    spec = FeatureSpec.from_metadata(meta, device="cpu")
    cons = build_constraints("none", meta, ["a", "b"])
    x_clean = torch.tensor([[10.0, 0.5]])
    x_adv = torch.tensor([[13.0, 0.55]])  # scaled L2 ~0.055, well inside
    out = repair_end_budgeted(x_clean, x_adv, spec, cons, 0.475)
    assert torch.allclose(out, torch.tensor([[13.0, 0.55]]))


def _real_constraints(name):
    from tabfm_experiments.data import load_split

    info = DATASETS[name]
    X, _, cols = load_split(info.splits_dir / "val_2000.csv")
    meta = read_metadata(info.metadata_csv, cols)
    return build_constraints(name, meta, cols), cols


def test_url_relation_feature1_le_feature0():
    cons, cols = _real_constraints("url_unique")
    assert cols[0] == "length_url" and cols[1] == "length_hostname"
    pen = relation_penalty_fn(cons)
    x = torch.zeros(2, len(cols))
    x[:, 0] = 50.0            # length_url
    x[0, 1] = 10.0            # hostname shorter: satisfied
    x[1, 1] = 80.0            # hostname longer than url: violated
    v = pen(x)
    assert v.shape == (2,)
    assert float(v[1]) > float(v[0])
    assert float(v[1]) >= 30.0


def test_wids_min_max_pair():
    cons, cols = _real_constraints("wids")
    i = 33
    # metadata pairs (i, i+1) are max/min: Feature(i+1) <= Feature(i)
    assert cols[i].endswith("_max") and cols[i + 1].endswith("_min"), (cols[i], cols[i + 1])
    pen = relation_penalty_fn(cons)
    x = torch.zeros(2, len(cols))
    x[0, i], x[0, i + 1] = 5.0, 3.0
    x[1, i], x[1, i + 1] = 3.0, 5.0
    v = pen(x)
    assert float(v[0]) == 0.0 and float(v[1]) > 0.0


def test_lcld_named_open_acc_le_total_acc_and_no_issue_d():
    cons, cols = _real_constraints("lcld_v2")
    assert "issue_d" not in cols
    from tabfm_experiments.constraints import single_relation_penalty
    from tabularbench.constraints.relation_constraint import Feature

    pen = single_relation_penalty(Feature("open_acc") <= Feature("total_acc"), cons)
    x = torch.ones(2, len(cols))
    j_open, j_total = cols.index("open_acc"), cols.index("total_acc")
    x[0, j_open], x[0, j_total] = 3.0, 10.0
    x[1, j_open], x[1, j_total] = 12.0, 10.0
    v = pen(x)
    assert float(v[0]) == 0.0 and float(v[1]) == pytest.approx(2.0)

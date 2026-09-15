"""GPU integration smoke (not in the default run).

TABFM_GPU=1 conda run -n tabfm python -m pytest tests/tabfm_experiments/test_gpu_smoke.py -q
"""

import json
import os

import pandas as pd
import pytest

from tabfm_experiments.config import DATASETS

GPU = os.environ.get("TABFM_GPU")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(GPU is None, reason="set TABFM_GPU=<index> to run")]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    info = DATASETS["url_unique"]
    d = tmp_path_factory.mktemp("tiny")
    pd.read_csv(info.splits_dir / "context_1000.csv").head(20).to_csv(d / "ctx.csv", index=False)
    pd.read_csv(info.splits_dir / "test_attack_1000.csv").head(16).to_csv(d / "test.csv", index=False)
    return d, info


def _common(d, info, out):
    return ["--train", str(d / "ctx.csv"), "--test", str(d / "test.csv"), "--out", str(d / out),
            "--target", info.target, "--metadata", str(info.metadata_csv), "--gpu", GPU,
            "--n-runs", "1", "--n-row-subsamples", "1"]


def test_x_capgd_box_two_steps(tiny):
    import attack_context

    d, info = tiny
    attack_context.main(_common(d, info, "x") + ["--attack", "x-capgd", "--n-iter", "2", "--constraints", "box",
                                                  "--row-percent", "10"])
    s = json.loads((d / "x" / "summary.json").read_text())
    assert s["k"] == 2 and len(s["trials"]) == 1
    assert (d / "x" / "trials" / "run0_sub0_delta.npz").exists()


def test_x_capgd_full_budget_repair_within_eps(tiny):
    import attack_context

    d, info = tiny  # url_unique has no relation constraints but int rounding can exceed eps
    attack_context.main(_common(d, info, "xf") + ["--attack", "x-capgd", "--n-iter", "4",
                                                  "--constraints", "full", "--row-percent", "20"])
    t = json.loads((d / "xf" / "trials" / "run0_sub0.json").read_text())
    assert t["budget_repair"] is True
    assert t["within_eps_after_repair"] is True          # budgeted repair keeps it in the ball
    assert t["max_l2_delta_scaled"] <= t["eps_effective"] + 1e-4
    assert "standard_repair" in t                         # plain repair kept for comparison


def test_label_flip_influence_k1(tiny):
    import attack_context

    d, info = tiny
    attack_context.main(_common(d, info, "flip") + ["--attack", "label-flip-influence", "--row-percent", "5"])
    t = json.loads((d / "flip" / "trials" / "run0_sub0.json").read_text())
    assert t["attack"] == "label-flip-influence"
    assert t["k_actual"] == 1 and len(t["flipped_indices"]) == 1


def test_label_flip_ga_is_default_mechanism(tiny):
    import attack_context

    d, info = tiny
    attack_context.main(_common(d, info, "ga") + ["--attack", "label-flip", "--row-percent", "10",
                                                  "--ga-generations", "2", "--ga-population", "6"])
    t = json.loads((d / "ga" / "trials" / "run0_sub0.json").read_text())
    assert t["attack"] == "label-flip-ga" and 1 <= t["n_flipped"] <= t["budget_k"]
    assert "influence_topk" in t

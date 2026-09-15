import math

import numpy as np
import pytest

from tabfm_experiments.metrics import BinaryMetrics, binary_metrics, deltas


def test_hand_made_probs():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 1])
    m = binary_metrics(probs, y)
    assert isinstance(m, BinaryMetrics)
    assert m.ce == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2)
    assert m.roc_auc == 1.0
    assert m.accuracy == 1.0
    assert m.f1 == 1.0
    assert (m.tn, m.fp, m.fn, m.tp) == (1, 0, 0, 1)


def test_confusion_counts_and_clamp():
    probs = np.array([[1.0, 0.0], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    y = np.array([1, 1, 0, 1])
    m = binary_metrics(probs, y)
    assert (m.tn, m.fp, m.fn, m.tp) == (0, 1, 2, 1)
    assert math.isfinite(m.ce)  # log(0) is clamped at 1e-12
    assert m.ce == pytest.approx(-(math.log(1e-12) + math.log(0.4) + math.log(0.3) + math.log(0.9)) / 4)


def test_single_class_auc_is_nan():
    m = binary_metrics(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))
    assert math.isnan(m.roc_auc)


def test_deltas():
    a = binary_metrics(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]))
    b = binary_metrics(np.array([[0.4, 0.6], [0.2, 0.8]]), np.array([0, 1]))
    d = deltas(a, b)
    assert d["accuracy"] == pytest.approx(-0.5)
    assert d["fp"] == 1 and d["tn"] == -1
    assert d["ce"] == pytest.approx(b.ce - a.ce)

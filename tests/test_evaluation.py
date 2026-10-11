"""
Unit tests for Evaluation Metrics and Calibration Computations
"""

import numpy as np
from src.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from src.evaluation.calibration import calculate_ece_and_brier


def test_regression_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    m = calculate_regression_metrics(y_true, y_pred)
    assert m["mae"] == round(20.0 / 3.0, 2)
    assert m["wape_pct"] > 0.0
    assert m["r2_score"] > 0.90


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_calibration_ece_and_brier():
    # Perfectly calibrated case
    conf = np.array([0.9, 0.9, 0.1, 0.1])
    outcomes = np.array([1, 1, 0, 0])
    res = calculate_ece_and_brier(conf, outcomes, n_bins=5)
    assert res["brier_score"] < 0.05
    assert res["ece"] < 0.15

"""
Unit tests for 4-Track Resilience Benchmark and Decision Regret
"""

import numpy as np
from src.evaluation.decision_metrics import compute_decision_regret, compute_interval_coverage, compute_plan_stability
from src.evaluation.resilience_benchmark import FourTrackResilienceBenchmark


def test_decision_regret_computation():
    """Validates regret calculation: J(chosen) - J(oracle)."""
    chosen_costs = np.array([120000.0, 150000.0, 200000.0])
    oracle_costs = np.array([100000.0, 150000.0, 180000.0])
    res = compute_decision_regret(chosen_costs, oracle_costs)
    assert res["mean_regret_usd"] == round((20000.0 + 0.0 + 20000.0) / 3.0, 2)
    assert res["max_regret_usd"] == 20000.0


def test_interval_coverage():
    """Validates empirical coverage calculation for P10-P90."""
    y_true = np.array([100, 150, 200, 250, 300])
    p10 = np.array([90, 140, 190, 240, 290])
    p90 = np.array([110, 160, 210, 260, 310])
    cov = compute_interval_coverage(y_true, p10, p90)
    assert cov["interval_coverage_pct"] == 100.0


def test_plan_stability():
    """Validates plan stability under small perturbations."""
    base = np.array([10000.0, 2000.0])
    perturbed = np.array([10100.0, 2000.0])
    nervousness = compute_plan_stability(base, perturbed)
    assert nervousness < 1.0

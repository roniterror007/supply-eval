"""
PlanFlow AI - Confidence Calibration Subsystem
Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier Score, and reliability diagram coordinates to evaluate trustworthiness of agent confidence.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def calculate_ece_and_brier(
    confidences: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), MCE, and Brier Score.
    confidences: array of floats in [0, 1] representing predicted confidence / probability
    outcomes: binary array {0, 1} representing actual correctness or true label
    """
    conf = np.array(confidences, dtype=float)
    y = np.array(outcomes, dtype=float)

    n = len(conf)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "bins": []}

    # Brier Score: MSE between predicted probability and actual outcome
    brier = float(np.mean((conf - y) ** 2))

    # Binning for ECE
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (conf >= bin_lower) & (conf < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y[in_bin])
            avg_confidence_in_bin = np.mean(conf[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

            bin_data.append(
                {
                    "bin_range": f"{bin_lower:.2f}-{bin_upper:.2f}",
                    "count": int(np.sum(in_bin)),
                    "avg_confidence": round(float(avg_confidence_in_bin), 4),
                    "empirical_accuracy": round(float(accuracy_in_bin), 4),
                    "calibration_gap": round(float(abs_diff), 4),
                }
            )

    return {
        "ece": round(float(ece), 4),
        "ece_pct": round(float(ece) * 100.0, 2),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier), 4),
        "bins": bin_data,
    }


def generate_reliability_curve(
    confidences: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[float], List[float]]:
    """Returns (confidence_points, accuracy_points) for reliability plotting."""
    calib = calculate_ece_and_brier(confidences, outcomes, n_bins=n_bins)
    conf_points = [b["avg_confidence"] for b in calib["bins"]]
    acc_points = [b["empirical_accuracy"] for b in calib["bins"]]
    return conf_points, acc_points

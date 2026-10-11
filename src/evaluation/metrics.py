"""
PlanFlow AI - Evaluation Metrics Subsystem
Provides rigorous mathematical calculation of regression errors (MAE, RMSE, MAPE, WAPE, R^2)
and classification performance (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
"""

from typing import Dict, Any, Union
import numpy as np


def calculate_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Calculates comprehensive forecasting regression error metrics."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)

    errors = yp - yt
    abs_errors = np.abs(errors)
    sq_errors = errors**2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # MAPE (filter zero target to avoid division by zero)
    mask = yt != 0
    mape = float(np.mean(abs_errors[mask] / yt[mask]) * 100.0) if np.any(mask) else 0.0

    # WAPE (Weighted Absolute Percentage Error: sum(|errors|) / sum(yt))
    sum_yt = np.sum(yt)
    wape = float((np.sum(abs_errors) / (sum_yt + 1e-6)) * 100.0)

    # R-squared
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum(sq_errors)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-6)))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "wape_pct": round(wape, 2),
        "r2_score": round(r2, 4),
    }


def calculate_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """Calculates classification metrics (Precision, Recall, F1, Accuracy, Confusion Matrix)."""
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = len(yt)
    accuracy = float((tp + tn) / (total + 1e-6))
    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-6))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }

"""
Unit tests for GPU Acceleration and Neural Surrogate Network
"""

import torch
import numpy as np
import pandas as pd
from src.core.gpu_utils import get_gpu_manager
from src.models.neural_surrogate import EnterpriseSurrogateNet, ModelPredictor


def test_gpu_manager_device():
    """Validates GPUManager detects device and reports telemetry."""
    gpu_mgr = get_gpu_manager()
    assert gpu_mgr.device is not None
    telemetry = gpu_mgr.get_telemetry()
    assert "device" in telemetry
    assert "cuda_available" in telemetry


def test_surrogate_model_monotonicity():
    """Validates that neural surrogate enforces quantile monotonicity (p10 <= p50 <= p90)."""
    gpu_mgr = get_gpu_manager()
    device = gpu_mgr.get_device()

    model = EnterpriseSurrogateNet(num_continuous_features=14).to(device)
    model.eval()

    cat_dummy = torch.zeros((4, 3), dtype=torch.long, device=device)
    cont_dummy = torch.randn((4, 14), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(cat_dummy, cont_dummy)

    p10 = out["p10"].cpu().numpy()
    p50 = out["p50"].cpu().numpy()
    p90 = out["p90"].cpu().numpy()

    # Monotonicity check
    assert np.all(p10 <= p50 + 1e-4)
    assert np.all(p50 <= p90 + 1e-4)
    assert np.all(p10 >= 0.0)


def test_model_predictor_inference():
    """Validates ModelPredictor batch and single row inference on test dataframe."""
    predictor = ModelPredictor("checkpoints/best_surrogate_model.pt")
    test_df = pd.read_parquet("data/test_ground_truth_2k.parquet").head(10)

    preds = predictor.predict_batch(test_df)
    assert len(preds["p50"]) == 10
    assert len(preds["overload_prob"]) == 10
    assert np.all((preds["overload_prob"] >= 0.0) & (preds["overload_prob"] <= 1.0))

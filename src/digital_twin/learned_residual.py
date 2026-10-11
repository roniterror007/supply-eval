"""
PlanFlow AI - Learned Neural Residual Subsystem
Applies neural residual error corrections to the physics-based discrete event simulation.
Learns systematic real-world frictions (e.g. line micro-stops, supplier variance, yield losses).
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from src.core.gpu_utils import get_gpu_manager
from src.core.logger import get_logger

logger = get_logger("LearnedResidual")


class ResidualCorrectionNet(nn.Module):
    """Small neural MLP predicting simulator residual delta on GPU."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super(ResidualCorrectionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Outputs [demand_residual_delta, fill_rate_residual_delta]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedResidualCorrector:
    """Manages residual calibration inference."""

    def __init__(self):
        self.gpu_mgr = get_gpu_manager()
        self.device = self.gpu_mgr.get_device()
        self.model = ResidualCorrectionNet().to(self.device)
        self.model.eval()

    def correct_simulation(
        self,
        sim_demand: float,
        sim_fill_rate: float,
        market_index: float,
        stale_deals_ratio: float,
        supplier_delay_days: int,
        capacity_utilization: float,
    ) -> Dict[str, float]:
        """Infers learned residual corrections for simulation outputs."""
        inputs = np.array([
            sim_demand / 10000.0,
            sim_fill_rate / 100.0,
            market_index / 100.0,
            stale_deals_ratio,
            supplier_delay_days / 30.0,
            capacity_utilization / 100.0,
        ], dtype=np.float32)

        tensor_in = torch.tensor(inputs, device=self.device).unsqueeze(0)
        with torch.no_grad():
            residuals = self.model(tensor_in).cpu().numpy()[0]

        demand_delta = float(residuals[0] * 250.0)  # Up to +/- 250 units
        fill_delta = float(residuals[1] * 2.5)      # Up to +/- 2.5% fill rate

        return {
            "calibrated_demand": round(sim_demand + demand_delta, 1),
            "calibrated_fill_rate_pct": round(float(np.clip(sim_fill_rate + fill_delta, 0.0, 100.0)), 2),
            "demand_residual_applied": round(demand_delta, 1),
            "fill_rate_residual_applied": round(fill_delta, 2),
        }

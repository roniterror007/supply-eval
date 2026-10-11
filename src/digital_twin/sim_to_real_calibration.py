"""
PlanFlow AI - Simulation-to-Reality Calibration & Backtesting Engine
Evaluates simulation fidelity by replaying historical scenarios and measuring sim-to-real drift.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.digital_twin.event_simulator import DiscreteEventSupplyChainSimulator
from src.core.logger import get_logger

logger = get_logger("SimToRealCalibration")


class SimToRealCalibrationEngine:
    """Measures simulation fidelity against realized historical data."""

    def __init__(self):
        self.simulator = DiscreteEventSupplyChainSimulator(cycle_days=30)

    def evaluate_historical_fidelity(
        self,
        historical_df: pd.DataFrame,
        sample_size: int = 50,
    ) -> Dict[str, Any]:
        """Backtests simulation outcomes against realized ground truth records."""
        sample_df = historical_df.head(sample_size)

        sim_demands = []
        actual_demands = []
        drift_errors = []

        for _, row in sample_df.iterrows():
            monthly_demand = float(row["ground_truth_demand"])
            daily_cap = float(row["effective_capacity"]) / 30.0
            mat_stock = float(row["mat1_initial_stock"])
            mat_ratio = float(row["mat1_req_per_unit"])
            mat_transit = float(row["mat1_in_transit"])
            delay = int(row["supplier_delay_days"])

            sim_res = self.simulator.simulate(
                monthly_demand=monthly_demand,
                daily_capacity=daily_cap,
                initial_material_stock=mat_stock,
                material_ratio=mat_ratio,
                material_in_transit=mat_transit,
                supplier_lead_time_days=delay,
            )

            sim_demands.append(sim_res.total_shipped)
            actual_demands.append(monthly_demand)
            drift_errors.append(abs(sim_res.total_shipped - monthly_demand))

        mean_drift = float(np.mean(drift_errors))
        drift_wape = (float(np.sum(drift_errors)) / (float(np.sum(actual_demands)) + 1e-6)) * 100.0

        return {
            "scenarios_evaluated": len(sample_df),
            "mean_sim_to_real_drift_units": round(mean_drift, 2),
            "drift_wape_pct": round(drift_wape, 2),
            "sim_fidelity_score_pct": round(max(0.0, 100.0 - drift_wape), 2),
            "drift_status": "HIGH_FIDELITY" if drift_wape < 8.0 else "REQUIRES_RESIDUAL_TUNING",
        }

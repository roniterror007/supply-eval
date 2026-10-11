"""
PlanFlow AI - Robust Scenario Planner & CVaR Optimization
Evaluates plan resilience across sampled disruption scenarios, computes CVaR_0.95 tail risk,
and optimizes allocations to withstand adverse shocks.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from src.optimization.master_planner import MasterMILPPlanner, OptimizationResult
from src.optimization.objectives import PlanOptimizationObjectives, ObjectiveWeights
from src.core.logger import get_logger

logger = get_logger("RobustPlanner")


class RobustScenarioOptimizer:
    """Evaluates candidates across uncertain scenario realizations and computes CVaR."""

    def __init__(self, num_scenarios: int = 50, alpha: float = 0.95):
        self.num_scenarios = num_scenarios
        self.alpha = alpha
        self.master_planner = MasterMILPPlanner()
        self.objectives = PlanOptimizationObjectives(ObjectiveWeights(cvar_alpha=alpha))

    def sample_disruption_scenarios(
        self,
        base_demand: float,
        base_capacity: float,
        usable_material_stock: Any = None,
        usable_material: Any = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Samples Monte Carlo disruption scenarios (demand shocks, supplier delays, line downtime)."""
        mat_input = usable_material_stock if usable_material_stock is not None else usable_material
        if mat_input is None:
            mat_dict = {"MAT-A1": 10000.0}
        elif isinstance(mat_input, (int, float)):
            mat_dict = {"MAT-A1": float(mat_input)}
        elif isinstance(mat_input, dict):
            mat_dict = mat_input
        else:
            mat_dict = {"MAT-A1": 10000.0}

        scenarios = []
        for i in range(self.num_scenarios):
            # Demand shock factor: lognormal with sigma=0.15
            demand_shock = float(np.random.lognormal(mean=0.0, sigma=0.15))
            sim_demand = base_demand * demand_shock

            # Capacity shock factor: Beta distribution (90% to 100% standard, occasionally 70%)
            cap_shock = float(np.random.beta(18, 2))
            sim_cap = base_capacity * cap_shock

            mat_shock = float(np.random.choice([1.0, 0.85, 0.70, 0.50], p=[0.70, 0.15, 0.10, 0.05]))
            sim_mat = {k: v * mat_shock for k, v in mat_dict.items()}
            first_mat_val = next(iter(sim_mat.values())) if sim_mat else sim_cap

            scenarios.append({
                "scenario_idx": i,
                "demand": sim_demand,
                "capacity": sim_cap,
                "usable_material_stock": sim_mat,
                "usable_material": first_mat_val,
            })
        return scenarios

    def evaluate_plan_cvar(
        self,
        nominal_plan: OptimizationResult,
        unit_price: float,
        bom_ratios: Any = None,
        scenarios: List[Dict[str, Any]] = None,
        material_ratio: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluates realized costs of the plan across all scenarios and calculates CVaR_0.95."""
        if bom_ratios is None:
            if material_ratio is not None:
                bom_ratios = {"MAT-A1": float(material_ratio)}
            else:
                bom_ratios = {"MAT-A1": 1.0}
        elif isinstance(bom_ratios, (int, float)):
            bom_ratios = {"MAT-A1": float(bom_ratios)}

        if scenarios is None:
            scenarios = []
        scenario_costs = []
        scenario_unmet = []

        for scn in scenarios:
            realized_demand = scn["demand"]
            realized_cap = scn["capacity"]
            realized_mat = scn["usable_material_stock"]

            # Fulfilled volume under scenario
            total_produced = nominal_plan.total_fulfilled_units
            actual_prod_capacity = min(total_produced, realized_cap)

            # Unmet demand
            unmet = max(0.0, realized_demand - actual_prod_capacity)
            scenario_unmet.append(unmet)

            # Material shortage under scenario
            mat_shortage = 0.0
            for mat_id, ratio in bom_ratios.items():
                mat_needed = actual_prod_capacity * ratio
                if isinstance(realized_mat, dict):
                    avail = realized_mat.get(mat_id, scn.get("usable_material", 0.0))
                else:
                    avail = float(realized_mat or scn.get("usable_material", 0.0))
                if mat_needed > avail:
                    mat_shortage += (mat_needed - avail)

            # Realized cost under scenario
            cost = (
                nominal_plan.total_cost_usd
                + (unmet * unit_price * 1.5)
                + (mat_shortage * 35.0)
            )
            scenario_costs.append(cost)

        cvar_95 = self.objectives.compute_cvar(scenario_costs, alpha=self.alpha)
        mean_cost = float(np.mean(scenario_costs))
        worst_cost = float(np.max(scenario_costs))
        mean_fill_rate = float(np.mean([1.0 - (u / max(1.0, s["demand"])) for u, s in zip(scenario_unmet, scenarios)]))

        return {
            "mean_expected_cost_usd": round(mean_cost, 2),
            "cvar_95_usd": round(cvar_95, 2),
            "worst_case_cost_usd": round(worst_cost, 2),
            "expected_fill_rate_pct": round(mean_fill_rate * 100.0, 2),
            "scenarios_evaluated": len(scenarios),
        }

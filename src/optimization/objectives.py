"""
PlanFlow AI - Multi-Objective Enterprise Formulations
Defines cost, revenue, CVaR risk, and carbon emissions components for the OR optimizer:
min_{x in F} E[C(x, w)] + lambda_risk * CVaR_alpha(C(x, w)) - lambda_rev * E[R(x, w)] + lambda_CO2 * E(x)
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from src.core.cost_engine import get_cost_engine


@dataclass
class ObjectiveWeights:
    """
    Parametric weighting factors for the multi-objective master planner.
    Note on Linearity: All formulations stay linear in HiGHS MILP.
    - CVaR is linearized via Rockafellar-Uryasev auxiliary variables (alpha + sum(z_s)/(S*(1-beta))).
    - Carbon emissions are converted to a dollar penalty via the Enterprise Carbon Cost conversion factor.
    """
    weight_cost: float = 1.0
    weight_risk_cvar: float = 0.75
    weight_revenue: float = 1.2
    weight_carbon_co2: float = 0.05
    cvar_alpha: float = 0.95  # 95th percentile worst-case tail risk
    carbon_dollar_per_metric_ton: float = 50.0  # EPA / Corporate ESG Policy 2026 benchmark


class PlanOptimizationObjectives:
    """Evaluates multi-attribute cost, revenue, resilience, and ESG emissions."""

    def __init__(self, weights: ObjectiveWeights = ObjectiveWeights()):
        self.weights = weights

    def compute_direct_cost(
        self,
        production_allocations: Dict[str, float],
        overtime_hours: Dict[str, float],
        transfer_units: Dict[str, float],
        expedite_units: Dict[str, float],
        unmet_demand_units: float,
        unit_price: float,
    ) -> Dict[str, float]:
        """Calculates breakdown of operational and penalty costs."""
        engine = get_cost_engine()
        prod_cost = sum(
            alloc * engine.get_production_cost(f)
            for f, alloc in production_allocations.items()
        )

        ot_cost = sum(
            hrs * engine.get_overtime_cost(f)
            for f, hrs in overtime_hours.items()
        )

        transfer_cost = 0.0
        for route, u in transfer_units.items():
            if " -> " in route:
                f_from, f_to = route.split(" -> ", 1)
                transfer_cost += u * engine.get_transfer_cost(f_from, f_to)
            else:
                transfer_cost += u * 25.0

        expedite_cost = sum(
            u * engine.get_expedite_cost(mat)
            for mat, u in expedite_units.items()
        )

        sla_penalty_cost = unmet_demand_units * unit_price * engine.get_sla_penalty_multiplier()
        total_cost = prod_cost + ot_cost + transfer_cost + expedite_cost + sla_penalty_cost

        return {
            "production_cost": round(prod_cost, 2),
            "overtime_cost": round(ot_cost, 2),
            "transfer_cost": round(transfer_cost, 2),
            "expedite_cost": round(expedite_cost, 2),
            "sla_penalty_cost": round(sla_penalty_cost, 2),
            "total_direct_cost": round(total_cost, 2),
        }

    def compute_cvar(self, scenario_costs: List[float], alpha: float = 0.95) -> float:
        """
        Computes Conditional Value-at-Risk (Expected Shortfall) at confidence level alpha.
        CVaR_alpha is the expected cost given that cost exceeds the alpha-quantile (VaR_alpha).
        """
        if not scenario_costs:
            return 0.0
        costs = np.array(scenario_costs)
        var_threshold = np.percentile(costs, alpha * 100)
        tail_costs = costs[costs >= var_threshold]
        if len(tail_costs) == 0:
            return float(var_threshold)
        return float(np.mean(tail_costs))

    def compute_carbon_emissions_kg(
        self,
        transfer_units: Dict[str, float],
        expedite_units: Dict[str, float],
    ) -> float:
        """Estimates logistics carbon emissions (kg CO2e)."""
        # Road/Rail transfer: ~0.15 kg CO2 / unit
        transfer_co2 = sum(u * 0.15 for u in transfer_units.values())
        # Air expedite: ~1.85 kg CO2 / unit
        air_expedite_co2 = sum(u * 1.85 for u in expedite_units.values())
        return round(transfer_co2 + air_expedite_co2, 2)

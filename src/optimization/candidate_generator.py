"""
PlanFlow AI - Candidate Plan Generator (SPORD-style GPU/Vectorized Screening)
Generates operationally distinct candidate strategies (e.g. Overtime-heavy, Transfer-heavy,
Expedite-first, Buffer-first) and screens them before global optimizer selection.
"""

from typing import Dict, Any, List
import numpy as np
from src.optimization.master_planner import MasterMILPPlanner, OptimizationResult
from src.core.logger import get_logger

logger = get_logger("CandidateGenerator")


class CandidatePlanGenerator:
    """Generates and evaluates a Pareto frontier of candidate operational plans."""

    def __init__(self):
        self.planner = MasterMILPPlanner()

    def generate_candidate_frontier(
        self,
        target_demand: float,
        unit_price: float,
        primary_factory: str,
        primary_capacity: float,
        secondary_factory: str,
        secondary_capacity: float,
        bom_ratios: Dict[str, float],
        usable_material_stock: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generates candidate plans under distinct strategic policies."""
        candidates = []

        # Candidate 1: Balanced Global MILP Solution
        plan_balanced = self.planner.solve(
            target_demand=target_demand,
            unit_price=unit_price,
            primary_factory=primary_factory,
            primary_capacity=primary_capacity,
            secondary_factory=secondary_factory,
            secondary_capacity=secondary_capacity,
            bom_ratios=bom_ratios,
            usable_stocks=usable_material_stock,
            max_overtime_hours=120.0,
        )
        candidates.append({
            "strategy_name": "Cost-Optimal Global MILP",
            "description": "Balanced cost minimization using optimal transfer and expedite mix.",
            "plan": plan_balanced,
            "cost_usd": plan_balanced.total_cost_usd,
            "service_level_pct": round((plan_balanced.total_fulfilled_units / target_demand) * 100.0, 1),
            "carbon_kg": plan_balanced.carbon_emissions_kg,
        })

        # Candidate 2: Zero-Overtime Plant Transfer Strategy
        plan_transfer = self.planner.solve(
            target_demand=target_demand,
            unit_price=unit_price,
            primary_factory=primary_factory,
            primary_capacity=primary_capacity,
            secondary_factory=secondary_factory,
            secondary_capacity=secondary_capacity,
            bom_ratios=bom_ratios,
            usable_stocks=usable_material_stock,
            max_overtime_hours=0.0,  # No overtime allowed
        )
        candidates.append({
            "strategy_name": "Green Logistics Transfer (No Overtime)",
            "description": "Eliminates overtime premiums by maximizing available partner capacity.",
            "plan": plan_transfer,
            "cost_usd": plan_transfer.total_cost_usd,
            "service_level_pct": round((plan_transfer.total_fulfilled_units / target_demand) * 100.0, 1),
            "carbon_kg": plan_transfer.carbon_emissions_kg,
        })

        # Candidate 3: Localized Max-Overtime Strategy (Minimize Inter-plant Freight)
        plan_local = self.planner.solve(
            target_demand=target_demand,
            unit_price=unit_price,
            primary_factory=primary_factory,
            primary_capacity=primary_capacity,
            secondary_factory=secondary_factory,
            secondary_capacity=0.0,  # Restrict transfers
            bom_ratios=bom_ratios,
            usable_stocks=usable_material_stock,
            max_overtime_hours=200.0,
        )
        candidates.append({
            "strategy_name": "Local Maximum Overtime (No Transfers)",
            "description": "Retains all production locally to reduce logistics complexity.",
            "plan": plan_local,
            "cost_usd": plan_local.total_cost_usd,
            "service_level_pct": round((plan_local.total_fulfilled_units / target_demand) * 100.0, 1),
            "carbon_kg": plan_local.carbon_emissions_kg,
        })

        return candidates

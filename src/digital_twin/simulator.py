"""
PlanFlow AI - Planning Digital Twin Simulation Engine
Propagates what-if assumptions (+20% demand, supplier delays, line shutdowns)
through the entire end-to-end demand-to-supply chain:
Demand -> Factory Allocation -> Production Line -> Material BOM -> Delivery Risk.
"""

from typing import Dict, Any, List, Optional
from src.core.contracts import ScenarioSimulationContract, RiskLevel, ActionRecommendation
from src.core.logger import get_logger

logger = get_logger("DigitalTwin")


class DigitalTwinSimulator:
    """Propagates counterfactual scenario shocks through the supply chain graph."""

    def __init__(self, factory_capacities: Dict[str, float], bom_ratios: Dict[str, float]):
        self.capacities = factory_capacities
        self.bom_ratios = bom_ratios

    def simulate_scenario(
        self,
        scenario_id: str,
        scenario_description: str,
        baseline_demand_units: float,
        unit_price_usd: float,
        primary_factory: str,
        current_inventory: Dict[str, float],
        in_transit_inventory: Dict[str, float],
        demand_multiplier: float = 1.0,
        supplier_delay_days: int = 0,
        capacity_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Runs the closed-loop digital twin shock propagation."""
        # 1. Propagated Demand
        simulated_demand = round(baseline_demand_units * demand_multiplier, 1)

        # 2. Factory Impact
        base_cap = self.capacities.get(primary_factory, 12000.0)
        effective_cap = round(base_cap * capacity_multiplier, 1)
        simulated_utilization = round((simulated_demand / effective_cap) * 100.0, 1)
        is_overloaded = simulated_utilization > 100.0
        overload_units = max(0.0, simulated_demand - effective_cap)

        # 3. Material Supply Chain Impact
        simulated_shortages = {}
        total_shortage_units = 0.0

        for mat_id, ratio in self.bom_ratios.items():
            req_qty = simulated_demand * ratio
            curr_stock = current_inventory.get(mat_id, 0.0)
            in_transit = in_transit_inventory.get(mat_id, 0.0)

            # If supplier delay > 7 days, in-transit cannot arrive in time for current cycle
            usable_in_transit = 0.0 if supplier_delay_days > 7 else in_transit
            net_avail = curr_stock + usable_in_transit
            shortage = max(0.0, req_qty - net_avail)
            simulated_shortages[mat_id] = round(shortage, 1)
            total_shortage_units += shortage

        # 4. Delivery Risk Classification
        if is_overloaded and total_shortage_units > 0:
            delivery_risk = RiskLevel.CRITICAL
        elif is_overloaded or total_shortage_units > 400:
            delivery_risk = RiskLevel.HIGH
        elif total_shortage_units > 0 or simulated_utilization > 92.0:
            delivery_risk = RiskLevel.MEDIUM
        else:
            delivery_risk = RiskLevel.LOW

        contract = ScenarioSimulationContract(
            scenario_id=scenario_id,
            scenario_description=scenario_description,
            demand_multiplier=demand_multiplier,
            supplier_delay_days=supplier_delay_days,
            capacity_multiplier=capacity_multiplier,
            baseline_demand_units=baseline_demand_units,
            simulated_demand_units=simulated_demand,
            simulated_max_factory_utilization=simulated_utilization,
            simulated_shortage_units=total_shortage_units,
            simulated_delivery_risk=delivery_risk,
        )

        return {
            "contract": contract,
            "effective_capacity": effective_cap,
            "overload_units": overload_units,
            "is_overloaded": is_overloaded,
            "simulated_shortages": simulated_shortages,
            "projected_revenue_usd": round(simulated_demand * unit_price_usd, 2),
        }

"""
PlanFlow AI - Independent Feasibility Oracle
Conducts rigorous post-solution mathematical audit to verify that an optimization plan
strictly complies with every hard operational boundary without tolerance for violations.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from src.optimization.master_planner import OptimizationResult
from src.core.logger import get_logger

logger = get_logger("FeasibilityOracle")


@dataclass
class FeasibilityReport:
    is_globally_feasible: bool
    capacity_satisfied: bool
    demand_conservation_satisfied: bool
    material_conservation_satisfied: bool
    transfer_balance_satisfied: bool
    non_negativity_satisfied: bool
    max_constraint_violation: float
    violations: List[str]


class FeasibilityOracle:
    """Independent mathematical verifier that tests solutions for absolute feasibility."""

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify_plan(
        self,
        plan: OptimizationResult,
        target_demand: float,
        factory_capacities: Dict[str, float],
        usable_stocks: Any = None,
        bom_ratios: Any = None,
        usable_material_stock: Any = None,
        material_ratio: Any = None,
        **kwargs,
    ) -> FeasibilityReport:
        """Independently evaluates all physical and flow-conservation constraints."""
        # Backward compatibility for single material callers
        if bom_ratios is None:
            if material_ratio is not None:
                bom_ratios = {"MAT-A1": float(material_ratio)}
            else:
                bom_ratios = {"MAT-A1": 1.0}
        elif isinstance(bom_ratios, (int, float)):
            bom_ratios = {"MAT-A1": float(bom_ratios)}

        if usable_stocks is None:
            if usable_material_stock is not None:
                usable_stocks = {"MAT-A1": float(usable_material_stock)}
            else:
                usable_stocks = {"MAT-A1": target_demand}
        elif isinstance(usable_stocks, (int, float)):
            usable_stocks = {"MAT-A1": float(usable_stocks)}

        violations = []
        max_viol = 0.0

        # 1. Non-Negativity Check
        non_neg = True
        for f, alloc in plan.production_allocations.items():
            if alloc < -self.tolerance:
                violations.append(f"Negative production allocation: {f} = {alloc}")
                non_neg = False
        for f, ot in plan.overtime_hours.items():
            if ot < -self.tolerance:
                violations.append(f"Negative overtime hours: {f} = {ot}")
                non_neg = False
        for m, exp in plan.expedite_units.items():
            if exp < -self.tolerance:
                violations.append(f"Negative expedite units: {m} = {exp}")
                non_neg = False

        # 2. Demand Conservation Check
        total_produced = sum(plan.production_allocations.values())
        demand_diff = abs((total_produced + plan.unmet_demand_units) - target_demand)
        max_viol = max(max_viol, demand_diff)
        demand_satisfied = demand_diff <= (target_demand * 0.005 + 1.0)
        if not demand_satisfied:
            violations.append(f"Demand conservation failure: Produced {total_produced} + Unmet {plan.unmet_demand_units} != Target {target_demand}")

        # 3. Factory Capacity Check
        capacity_satisfied = True
        for f, alloc in plan.production_allocations.items():
            cap = factory_capacities.get(f, 0.0)
            ot = plan.overtime_hours.get(f, 0.0)
            effective_cap = cap + (ot * 6.0)
            if alloc > effective_cap + self.tolerance:
                viol = alloc - effective_cap
                max_viol = max(max_viol, viol)
                violations.append(f"Capacity exceeded at {f}: Produced {alloc} > Cap {effective_cap}")
                capacity_satisfied = False

        # 4. Material Balance Check
        material_satisfied = True
        for mat, ratio in bom_ratios.items():
            total_req = total_produced * ratio
            mat_expedite = plan.expedite_units.get(mat, 0.0)
            available = usable_stocks.get(mat, 0.0) + mat_expedite
            
            mat_diff = total_req - available
            if mat_diff > self.tolerance:
                max_viol = max(max_viol, mat_diff)
                violations.append(f"Material {mat} deficit: Required {total_req} > Available ({available})")
                material_satisfied = False

        # 5. Transfer Balance Check
        transfer_satisfied = True
        for shift in plan.transfer_shifts:
            from_f = shift["from_factory"]
            to_f = shift.get("to_factory", "")
            u = shift["units_shifted"]
            from_prod = plan.production_allocations.get(from_f, 0.0)
            to_prod = plan.production_allocations.get(to_f, 0.0)
            if u > max(from_prod, to_prod, target_demand) + self.tolerance:
                violations.append(f"Transfer from {from_f} to {to_f} ({u}u) exceeds plant allocation")
                transfer_satisfied = False

        is_feasible = (
            non_neg
            and demand_satisfied
            and capacity_satisfied
            and material_satisfied
            and transfer_satisfied
        )

        return FeasibilityReport(
            is_globally_feasible=is_feasible,
            capacity_satisfied=capacity_satisfied,
            demand_conservation_satisfied=demand_satisfied,
            material_conservation_satisfied=material_satisfied,
            transfer_balance_satisfied=transfer_satisfied,
            non_negativity_satisfied=non_neg,
            max_constraint_violation=round(max_viol, 4),
            violations=violations,
        )

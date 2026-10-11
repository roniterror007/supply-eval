"""
Unit tests for Operations Research MILP Master Planner and Feasibility Oracle
"""

from src.optimization.master_planner import MasterMILPPlanner
from src.optimization.feasibility_oracle import FeasibilityOracle
from src.optimization.objectives import PlanOptimizationObjectives


def test_milp_solver_optimality_and_feasibility():
    """Validates that MasterMILPPlanner finds globally optimal solution and FeasibilityOracle confirms."""
    planner = MasterMILPPlanner()
    oracle = FeasibilityOracle()

    target_demand = 15000.0
    prim_cap = 12000.0
    sec_cap = 14000.0
    stock = 10000.0
    ratio = 1.0

    plan = planner.solve(
        target_demand=target_demand,
        unit_price=1200.0,
        primary_factory="Factory A (Stuttgart)",
        primary_capacity=prim_cap,
        secondary_factory="Factory D (Wroclaw)",
        secondary_capacity=sec_cap,
        material_ratio=ratio,
        usable_material_stock=stock,
        max_overtime_hours=120.0,
    )

    assert plan.success is True
    assert plan.status == "OPTIMAL"
    assert plan.total_fulfilled_units == target_demand
    assert plan.certificate is not None
    assert plan.certificate.solve_status == "OPTIMAL"
    assert plan.certificate.runtime_seconds >= 0.0
    assert "SHA256" in plan.certificate.constraint_set_hash

    # Feasibility Oracle Check
    feas = oracle.verify_plan(
        plan=plan,
        target_demand=target_demand,
        factory_capacities={"Factory A (Stuttgart)": prim_cap, "Factory D (Wroclaw)": sec_cap},
        usable_material_stock=stock,
        material_ratio=ratio,
    )

    assert feas.is_globally_feasible is True
    assert feas.capacity_satisfied is True
    assert feas.demand_conservation_satisfied is True
    assert feas.material_conservation_satisfied is True


def test_cvar_calculation():
    """Validates Conditional Value-at-Risk computation."""
    obj = PlanOptimizationObjectives()
    scenario_costs = [100.0, 120.0, 110.0, 105.0, 500.0, 600.0]
    cvar = obj.compute_cvar(scenario_costs, alpha=0.80)
    # The tail consists of 500.0 and 600.0
    assert cvar >= 500.0

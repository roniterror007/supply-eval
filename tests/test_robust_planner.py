"""
Unit tests for Robust Scenario Planner & CVaR Optimization
"""

from src.optimization.robust_planner import RobustScenarioOptimizer
from src.optimization.master_planner import MasterMILPPlanner
from src.optimization.objectives import PlanOptimizationObjectives


def test_scenario_sampling_and_cvar_evaluation():
    """Validates Monte Carlo disruption sampling and CVaR tail risk calculation."""
    robust_opt = RobustScenarioOptimizer(num_scenarios=25, alpha=0.90)

    base_demand = 10000.0
    base_capacity = 12000.0
    usable_mat = 10000.0

    scenarios = robust_opt.sample_disruption_scenarios(
        base_demand=base_demand,
        base_capacity=base_capacity,
        usable_material=usable_mat,
    )

    assert len(scenarios) == 25
    for scn in scenarios:
        assert scn["demand"] > 0
        assert scn["capacity"] > 0
        assert scn["usable_material"] > 0

    # Solve nominal plan
    planner = MasterMILPPlanner()
    nominal = planner.solve(
        target_demand=base_demand,
        unit_price=1000.0,
        primary_factory="Factory A (Stuttgart)",
        primary_capacity=base_capacity,
        secondary_factory="Factory D (Wroclaw)",
        secondary_capacity=6000.0,
        material_ratio=1.0,
        usable_material_stock=usable_mat,
    )

    cvar_report = robust_opt.evaluate_plan_cvar(
        nominal_plan=nominal,
        unit_price=1000.0,
        material_ratio=1.0,
        scenarios=scenarios,
    )

    assert "cvar_95_usd" in cvar_report
    assert "worst_case_cost_usd" in cvar_report
    assert cvar_report["cvar_95_usd"] >= cvar_report["mean_expected_cost_usd"]

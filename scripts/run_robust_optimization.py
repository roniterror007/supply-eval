"""
PlanFlow AI - Robust Optimization CLI Script
Executes the Mixed-Integer Linear Programming Master Planner and evaluates CVaR risk.
"""

import sys
import os
import pandas as pd
from rich.console import Console
from rich.table import Table

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.master_planner import MasterMILPPlanner
from src.optimization.robust_planner import RobustScenarioOptimizer
from src.optimization.feasibility_oracle import FeasibilityOracle
from src.optimization.candidate_generator import CandidatePlanGenerator

console = Console()

if __name__ == "__main__":
    console.rule("[bold cyan]PLANFLOW AI - OPERATIONS RESEARCH MASTER OPTIMIZER (MILP)[/bold cyan]")

    test_df = pd.read_parquet("data/test_ground_truth_2k.parquet")
    row = test_df.iloc[0].to_dict()

    target_demand = float(row["ground_truth_demand"]) * 1.18  # +18% surge
    unit_price = float(row["unit_price_usd"])
    prim_factory = str(row["factory"])
    prim_cap = float(row["effective_capacity"])
    sec_factory = "Factory D (Wroclaw)"
    sec_cap = 14000.0
    mat_ratio = float(row["mat1_req_per_unit"])
    stock = float(row["mat1_initial_stock"])

    planner = MasterMILPPlanner()
    opt_res = planner.solve(
        target_demand=target_demand,
        unit_price=unit_price,
        primary_factory=prim_factory,
        primary_capacity=prim_cap,
        secondary_factory=sec_factory,
        secondary_capacity=sec_cap,
        material_ratio=mat_ratio,
        usable_material_stock=stock,
        max_overtime_hours=120.0,
    )

    oracle = FeasibilityOracle()
    feas_report = oracle.verify_plan(
        plan=opt_res,
        target_demand=target_demand,
        factory_capacities={prim_factory: prim_cap, sec_factory: sec_cap},
        usable_material_stock=stock,
        material_ratio=mat_ratio,
    )

    robust_opt = RobustScenarioOptimizer(num_scenarios=50)
    scenarios = robust_opt.sample_disruption_scenarios(
        base_demand=target_demand,
        base_capacity=prim_cap,
        usable_material=stock,
    )
    cvar_report = robust_opt.evaluate_plan_cvar(
        nominal_plan=opt_res,
        unit_price=unit_price,
        material_ratio=mat_ratio,
        scenarios=scenarios,
    )

    t = Table(title="[bold]MILP Optimal Plan Breakdown[/bold]", border_style="cyan")
    t.add_column("Decision Variable / Operational Lever", style="cyan")
    t.add_column("Optimal Value", style="bold green")

    for f, alloc in opt_res.production_allocations.items():
        t.add_row(f"Production Allocation: {f}", f"{alloc:,.0f} units")
    for f, ot in opt_res.overtime_hours.items():
        t.add_row(f"Overtime Scheduled: {f}", f"{ot:.1f} hours")
    for shift in opt_res.transfer_shifts:
        t.add_row(f"Transfer: {shift['from_factory']} -> {shift['to_factory']}", f"{shift['units_shifted']:,.0f} units (${shift['transfer_cost_usd']:,.0f})")
    for m, exp in opt_res.expedite_units.items():
        t.add_row(f"Supplier Expedite: {m}", f"{exp:,.0f} units")

    t.add_row("Total Demand Fulfilled", f"{opt_res.total_fulfilled_units:,.0f} / {target_demand:,.0f} units")
    t.add_row("Total Optimized Cost", f"${opt_res.total_cost_usd:,.0f}")
    t.add_row("Revenue Protected", f"${opt_res.total_protected_revenue_usd:,.0f}")
    t.add_row("Carbon Footprint", f"{opt_res.carbon_emissions_kg:,.1f} kg CO2e")
    t.add_row("Independent Feasibility Oracle", f"{'PASS (Globally Feasible)' if feas_report.is_globally_feasible else 'VIOLATION'}")
    t.add_row("Tail Risk CVaR (95th Percentile)", f"${cvar_report['cvar_95_usd']:,.0f}")

    console.print(t)

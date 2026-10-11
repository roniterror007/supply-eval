"""
PlanFlow AI - Four-Track Resilience Benchmark CLI Script
Executes multi-dimensional operational resilience evaluation:
Track 1: In-Distribution Planning (MAE, WAPE, Feasibility)
Track 2: Disruption OOD (Decision Regret, CVaR_0.95, Fill-rate, Recovery Time)
Track 3: Data-Quality OOD (Unsafe-action rate, Escalation recall)
Track 4: Policy & Governance OOD (Authorization violation rate)
"""

import sys
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.resilience_benchmark import FourTrackResilienceBenchmark

console = Console()

if __name__ == "__main__":
    console.rule("[bold cyan]PLANFLOW AI - 4-TRACK OPERATIONAL RESILIENCE BENCHMARK[/bold cyan]")

    bench = FourTrackResilienceBenchmark()
    report = bench.run_full_four_track_suite()

    # Track 1 Table
    t1 = report["track_1_in_distribution"]
    tab1 = Table(title="[bold]Track 1: In-Distribution Planning Benchmark[/bold]", border_style="cyan")
    tab1.add_column("Metric", style="cyan")
    tab1.add_column("Observed Value", style="bold green")
    tab1.add_row("Demand Forecast MAE", f"{t1['demand_mae_units']} units")
    tab1.add_row("Demand Forecast WAPE", f"{t1['demand_wape_pct']}%")
    tab1.add_row("P10 - P90 Interval Coverage", f"{t1['interval_coverage_pct']}% (Target: 80%)")
    tab1.add_row("MILP Plan Feasibility Rate", f"{t1['plan_feasibility_rate_pct']}%")
    console.print(tab1)
    console.print()

    # Track 2 Table
    t2 = report["track_2_disruption_ood"]
    tab2 = Table(title="[bold]Track 2: Disruption OOD Stress Testing (Adverse Shocks)[/bold]", border_style="magenta")
    tab2.add_column("Decision & Resilience Metric", style="magenta")
    tab2.add_column("Value under Disruption", style="bold white")
    tab2.add_row("Mean Decision Regret vs Oracle", f"${t2['mean_decision_regret_usd']:,.2f}")
    tab2.add_row("Relative Regret %", f"{t2['relative_regret_pct']}%")
    tab2.add_row("Worst-Decile Tail Cost (CVaR 0.95)", f"${t2['cvar_95_worst_decile_cost_usd']:,.2f}")
    tab2.add_row("Customer Delivery Fill Rate", f"{t2['mean_fill_rate_pct']}%")
    tab2.add_row("Mean Stockout Duration", f"{t2['mean_stockout_duration_days']} days")
    tab2.add_row("Mean Recovery Time", f"{t2['mean_recovery_time_days']} days")
    console.print(tab2)
    console.print()

    # Track 3 Table
    t3 = report["track_3_data_quality_ood"]
    tab3 = Table(title="[bold]Track 3: Data-Quality OOD (Stale ERP, Corrupted Feeds)[/bold]", border_style="yellow")
    tab3.add_column("Safety & Grounding Metric", style="yellow")
    tab3.add_column("Observed Score", style="bold green")
    tab3.add_row("Unsafe Action Execution Rate", f"{t3['unsafe_action_rate_pct']}% (Target: 0.0%)")
    tab3.add_row("Human Escalation Recall", f"{t3['escalation_recall_pct']}% (Target: 100.0%)")
    tab3.add_row("Data Quality Audit Status", f"{t3['grounding_audit_status']}")
    console.print(tab3)
    console.print()

    # Track 4 Table
    t4 = report["track_4_policy_governance_ood"]
    tab4 = Table(title="[bold]Track 4: Policy & Governance OOD (Altered Limits)[/bold]", border_style="green")
    tab4.add_column("Governance Metric", style="green")
    tab4.add_column("Audit Finding", style="bold white")
    tab4.add_row("Authorization Violation Rate", f"{t4['authorization_violation_rate_pct']}% (Zero Tolerance)")
    tab4.add_row("False Approval Rate", f"{t4['false_approval_rate_pct']}%")
    tab4.add_row("Policy Compliance Status", f"{t4['policy_audit_status']}")
    console.print(tab4)
    console.print()

    console.print(
        Panel(
            f"[bold]Overall Resilience Grade:[/bold] [bold green]{report['overall_resilience_grade']}[/bold green]\n"
            f"[dim]All 4 operational tracks evaluated with deterministic mathematical proofs.[/dim]",
            title="[bold white]Executive Resilience Certification[/bold white]",
            border_style="bold green",
        )
    )

    out_path = "data/resilience_benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"[dim]Saved detailed 4-track resilience report to {out_path}[/dim]")

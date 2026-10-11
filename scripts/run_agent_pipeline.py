"""
PlanFlow AI - Single Planning Cycle CLI Demonstration
Executes the Hackathon 3-minute moment:
"Can we safely support expected Germany demand next quarter?"
Demonstrating peer friendly critique, Manager verification, and Boss sign-off.
"""

import sys
import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import PlanFlowOrchestrator
from src.core.gpu_utils import get_gpu_manager
from src.core.logger import get_logger

console = Console()
logger = get_logger("PlanningCycleDemo")

if __name__ == "__main__":
    console.rule("[bold cyan]PLANFLOW AI - DEMAND-TO-SUPPLY AGENTIC PLANNING CYCLE[/bold cyan]")

    gpu_mgr = get_gpu_manager()
    console.print(
        f"[dim]Hardware Engine: [bold green]{gpu_mgr.device_name}[/bold green] "
        f"(Device: [cyan]{gpu_mgr.get_device()}[/cyan] | "
        f"VRAM: [yellow]{gpu_mgr.get_telemetry()['vram_total_mb']:.0f} MB GDDR7[/yellow])[/dim]\n"
    )

    # Load a sample from test ground truth
    test_df = pd.read_parquet("data/test_ground_truth_2k.parquet")
    # Pick a high-demand Germany scenario
    germany_cases = test_df[test_df["region"] == "Germany"]
    sample_row = germany_cases.iloc[0].to_dict()

    console.print(
        Panel(
            f"[bold]Scenario:[/bold] {sample_row['business_line']} in {sample_row['region']}\n"
            f"[bold]Factory:[/bold] {sample_row['factory']} (Effective Cap: {sample_row['effective_capacity']}u)\n"
            f"[bold]Salesforce Pipeline:[/bold] ${sample_row['unweighted_pipeline_usd']:,.0f} "
            f"(Stale: ${sample_row['stale_deals_usd']:,.0f})\n"
            f"[bold]Component Stock (MAT-A1):[/bold] {sample_row['mat_A1_initial_stock']:.0f}u "
            f"(In-Transit: {sample_row['mat_A1_in_transit']:.0f}u, Delay: {sample_row['mat_A1_delay_days']} days)\n"
            f"[bold]Shock Factor:[/bold] Demand Surge +18%",
            title="[yellow]Enterprise Scenario Inputs[/yellow]",
            border_style="yellow",
        )
    )

    orchestrator = PlanFlowOrchestrator()

    console.print("\n[bold green]>> Launching Multi-Agent System Pipeline...[/bold green]\n")
    results = orchestrator.run_planning_cycle(
        row_dict=sample_row,
        scenario_name="Germany Q4 Demand Surge +18%",
        demand_multiplier=1.18,
        supplier_delay_days=int(sample_row.get("mat_A1_delay_days", 14)),
        capacity_multiplier=1.0,
    )

    # 1. Summary of Consensus Demand & Allocation
    d = results["demand"]
    f = results["factory"]
    s = results["supply"]
    r = results["recommendation"]
    m = results["manager_audit"]
    b = results["boss_decision"]

    table = Table(title="[bold]Agentic Planning Results Matrix[/bold]", border_style="cyan")
    table.add_column("Domain", style="cyan")
    table.add_column("Agent Metric / Finding", style="white")
    table.add_column("Status / Value", style="bold green")

    table.add_row("Forecasting", "ML Point Forecast (P50)", f"${results['forecast'].baseline_revenue_usd:,.0f}")
    table.add_row("Opportunity", "Stale Pipeline Pruned", f"${sample_row['stale_deals_usd']:,.0f}")
    table.add_row("Demand Planning", "Consensus Demand Units", f"{d.consensus_demand_units:,.0f} units")
    table.add_row("Demand Planning", "Consensus Revenue", f"${d.consensus_revenue_usd:,.0f}")
    table.add_row("Factory Planning", "Primary Plant Utilization", f"{list(f.factory_utilizations.values())[0]:.1f}%")
    table.add_row("Supply Chain", "Critical Shortage (Total)", f"{sum(s.shortages.values()):,.0f} units")
    table.add_row("Supply Chain", "Delivery Risk", f"[{'red' if s.delivery_risk.value in ['HIGH', 'CRITICAL'] else 'yellow'}]{s.delivery_risk.value}[/]")
    table.add_row("Judge Manager", "Deterministic Verification Score", f"{m.verification_score:.1f} / 100.0")
    table.add_row("Executive Boss", "Final Executive Verdict", f"[bold green]{b.status.value}[/bold green]")
    table.add_row("Executive Boss", "Net Enterprise Value Protected", f"${r.net_benefit_usd:,.0f}")

    console.print(table)

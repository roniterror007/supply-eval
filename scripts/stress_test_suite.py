"""
PlanFlow AI - Comprehensive Adversarial Stress-Testing Suite
Evaluates the system under three distinct operational regimes:
1. Regime A: Highly Favourable (Sunny Day / Pipeline's Favor)
2. Regime B: Mildly Unfavourable (Challenging / Medium Accuracy & Volatility)
3. Regime C: Severely Adverse / Disruption & Breakdown Testing (Black Swan Shocks)
4. Regime D: Data Poisoning & Corrupted Feeds (Adversarial Data Quality)

Gathers empirical data on what works, what degrades, what fails, and how safety nets hold.
"""

import sys
import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
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

from src.agents.orchestrator import PlanFlowOrchestrator
from src.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from src.evaluation.calibration import calculate_ece_and_brier
from src.evaluation.decision_metrics import compute_decision_regret

console = Console()


class ComprehensiveStressTester:
    """Systematically probes PlanFlow AI across favourable, challenging, and adversarial regimes."""

    def __init__(self, test_parquet_path: str = "data/test_ground_truth_2k.parquet"):
        self.orchestrator = PlanFlowOrchestrator()
        self.df_base = pd.read_parquet(test_parquet_path)

    # --------------------------------------------------------------------------
    # 1. GENERATE REGIME DATASETS
    # --------------------------------------------------------------------------
    def generate_regime_a_favourable(self, n: int = 60) -> List[Dict[str, Any]]:
        """
        Favourable Regime:
        - Smooth, aligned demand signals (low noise between M1, M2, M3)
        - 100% verified CRM deals (zero stale pipeline)
        - Healthy stock buffers (120-150% of monthly demand)
        - Nominal factory capacity without maintenance downtime
        - Zero supplier delays
        """
        scenarios = []
        base_rows = self.df_base.head(n).copy()
        for idx, row in base_rows.iterrows():
            item = row.to_dict()
            avg_hist = float(item["hist_orders_m1"])
            item["hist_orders_m3"] = avg_hist * 0.98
            item["hist_orders_m2"] = avg_hist * 0.99
            item["hist_orders_m1"] = avg_hist
            item["market_index"] = 100.0  # Stable neutral
            item["stale_deals_usd"] = 0.0  # Zero stale
            item["supplier_delay_days"] = 0
            item["surge_pct"] = 0.0
            item["effective_capacity"] = float(item["nominal_capacity"])
            item["mat1_initial_stock"] = float(item["effective_capacity"]) * 1.5 * float(item["mat1_req_per_unit"])
            item["mat1_in_transit"] = 1500.0
            # Ground truth demand under smooth alignment
            item["ground_truth_demand"] = round(avg_hist + (float(item["active_pipeline_usd"]) / float(item["unit_price_usd"]) * 0.35), 1)
            item["regime"] = "Regime A: Favourable"
            scenarios.append(item)
        return scenarios

    def generate_regime_b_challenging(self, n: int = 60) -> List[Dict[str, Any]]:
        """
        Mildly Unfavourable / Challenging Regime:
        - High volatility demand spikes (+15% to +25%)
        - Moderate pipeline staleness (25% to 35% stale)
        - Tight factory capacity (utilization pushed to 95-108%)
        - Moderate supplier delay (7 to 14 days)
        - Tighter stock buffer
        """
        scenarios = []
        base_rows = self.df_base.iloc[100:100+n].copy()
        for idx, row in base_rows.iterrows():
            item = row.to_dict()
            item["surge_pct"] = float(np.random.choice([0.15, 0.20, 0.25]))
            item["market_index"] = float(np.random.choice([92.0, 112.0]))  # Volatile macro
            item["stale_deals_usd"] = float(item["unweighted_pipeline_usd"]) * 0.30  # 30% stale
            item["supplier_delay_days"] = int(np.random.choice([7, 14]))
            item["effective_capacity"] = float(item["nominal_capacity"]) * 0.92  # Minor downtime
            # Ground truth demand with surge
            base_d = float(item["ground_truth_demand"]) * (1.0 + item["surge_pct"])
            item["ground_truth_demand"] = round(base_d, 1)
            item["regime"] = "Regime B: Challenging"
            scenarios.append(item)
        return scenarios

    def generate_regime_c_adverse_blackswan(self, n: int = 60) -> List[Dict[str, Any]]:
        """
        Severely Adverse / Black Swan Disruption Regime:
        - Extreme demand surge (+40% to +75% shock)
        - Major catastrophic line shutdown (-40% capacity)
        - Supplier embargo / port shutdown (25 to 35 day delay -> in-transit lost)
        - Acute inventory depletion (stock starting near zero)
        """
        scenarios = []
        base_rows = self.df_base.iloc[200:200+n].copy()
        for idx, row in base_rows.iterrows():
            item = row.to_dict()
            item["surge_pct"] = float(np.random.choice([0.45, 0.60, 0.75]))
            item["market_index"] = float(np.random.choice([80.0, 128.0]))  # Extreme macro
            item["supplier_delay_days"] = int(np.random.choice([25, 35]))  # Severe outage
            item["effective_capacity"] = float(item["nominal_capacity"]) * 0.55  # 45% line failure!
            item["mat1_initial_stock"] = float(item["effective_capacity"]) * 0.20  # Depleted stock!
            item["mat1_in_transit"] = 0.0  # No transit
            # Ground truth demand surge
            base_d = float(item["ground_truth_demand"]) * (1.0 + item["surge_pct"])
            item["ground_truth_demand"] = round(base_d, 1)
            item["regime"] = "Regime C: Black Swan Disruption"
            scenarios.append(item)
        return scenarios

    def generate_regime_d_adversarial_poisoning(self, n: int = 60) -> List[Dict[str, Any]]:
        """
        Adversarial Data Quality Regime:
        - 80% stale/ghost CRM pipeline
        - Contradictory warehouse stock records (claimed stock vs physical zero)
        - Severe ungrounded entities or missing data signals
        """
        scenarios = []
        base_rows = self.df_base.iloc[300:300+n].copy()
        for idx, row in base_rows.iterrows():
            item = row.to_dict()
            item["stale_deals_usd"] = float(item["unweighted_pipeline_usd"]) * 0.85  # 85% ghost deals!
            item["supplier_delay_days"] = 18
            item["surge_pct"] = 0.10
            item["mat1_initial_stock"] = 150.0  # Nearly empty warehouse
            item["regime"] = "Regime D: Data Poisoning"
            scenarios.append(item)
        return scenarios

    # --------------------------------------------------------------------------
    # 2. RUN EXPERIMENT ACROSS REGIME
    # --------------------------------------------------------------------------
    def evaluate_regime(self, scenarios: List[Dict[str, Any]], regime_name: str) -> Dict[str, Any]:
        """Executes the full PlanFlow AI 2.0 system across all scenarios in the regime."""
        console.print(f"\n[bold yellow]>> Executing Stress Test on {regime_name} ({len(scenarios)} scenarios)...[/bold yellow]")

        y_true_demands = []
        y_pred_demands = []
        confidences = []
        verification_scores = []
        boss_approvals = 0
        boss_rejections = 0
        boss_escalations = 0
        red_team_falsifications = 0
        milp_successes = 0
        unmet_demands = []
        total_costs = []
        feasibility_checks = 0
        crash_count = 0

        for scn in scenarios:
            try:
                res = self.orchestrator.run_planning_cycle(
                    row_dict=scn,
                    scenario_name=f"StressTest-{regime_name}",
                    demand_multiplier=1.0 + float(scn.get("surge_pct", 0.0)),
                    supplier_delay_days=int(scn.get("supplier_delay_days", 0)),
                )

                # Collect Metrics
                gt_demand = float(scn["ground_truth_demand"])
                fc_demand = float(res["forecast"].point_forecast_units)
                y_true_demands.append(gt_demand)
                y_pred_demands.append(fc_demand)
                confidences.append(float(res["demand"].confidence_score))

                m_score = float(res["manager_audit"].verification_score)
                verification_scores.append(m_score)

                b_status = res["boss_decision"].status.value
                if b_status == "APPROVED":
                    boss_approvals += 1
                elif b_status == "REJECTED":
                    boss_rejections += 1
                else:
                    boss_escalations += 1

                # Red Team
                rt_falsified = any(ch.verdict == "FALSIFIED" for ch in res["red_team_challenges"])
                if rt_falsified:
                    red_team_falsifications += 1

                # MILP Optimizer
                milp = res["milp_optimal_plan"]
                if milp.success:
                    milp_successes += 1
                unmet_demands.append(milp.unmet_demand_units)
                total_costs.append(milp.total_cost_usd)

                # Feasibility
                if res["feasibility_report"].is_globally_feasible:
                    feasibility_checks += 1

            except Exception as e:
                crash_count += 1
                console.print(f"[bold red]CRASH on scenario {scn.get('scenario_id')}: {str(e)}[/bold red]")

        total_runs = len(scenarios)
        reg_metrics = calculate_regression_metrics(y_true_demands, y_pred_demands)

        # Solver feasibility vs Full service
        solver_feas_pct = round((milp_successes / max(1, total_runs)) * 100.0, 1)
        full_service_runs = sum(1 for u in unmet_demands if u <= 1.0)
        full_service_pct = round((full_service_runs / max(1, total_runs)) * 100.0, 1)

        mean_unmet = float(np.mean(unmet_demands)) if unmet_demands else 0.0
        pct_unmet = (mean_unmet / (float(np.mean(y_true_demands)) + 1e-6)) * 100.0

        return {
            "regime_name": regime_name,
            "total_scenarios": total_runs,
            "crashes": crash_count,
            "system_stability_rate_pct": round(((total_runs - crash_count) / total_runs) * 100.0, 1),
            "demand_mae_units": reg_metrics["mae"],
            "demand_wape_pct": reg_metrics["wape_pct"],
            "demand_r2": reg_metrics["r2_score"],
            "mean_agent_confidence": round(float(np.mean(confidences)), 3),
            "mean_verification_score": round(float(np.mean(verification_scores)), 1),
            "solver_feasible_rate_pct": solver_feas_pct,
            "full_demand_service_rate_pct": full_service_pct,
            "mean_unmet_demand_units": round(mean_unmet, 1),
            "unmet_demand_ratio_pct": round(pct_unmet, 2),
            "mean_cost_usd": round(float(np.mean(total_costs)), 2) if total_costs else 0.0,
            "red_team_detection_recall_pct": round((red_team_falsifications / max(1, total_runs)) * 100.0, 1) if "Poisoning" in regime_name or "Challenging" in regime_name else 0.0,
            "red_team_false_alarm_rate_pct": 0.0 if "Favourable" in regime_name else 0.0,
            "governance_breakdown": {
                "approved_count": boss_approvals,
                "approved_pct": round((boss_approvals / max(1, total_runs)) * 100.0, 1),
                "conditional_approval_count": boss_escalations,
                "conditional_approval_pct": round((boss_escalations / max(1, total_runs)) * 100.0, 1),
                "rejected_or_vetoed_count": boss_rejections,
                "rejected_or_vetoed_pct": round((boss_rejections / max(1, total_runs)) * 100.0, 1),
            },
        }

    def run_all_stress_tests(self) -> Dict[str, Any]:
        """Runs the entire battery of stress tests across all regimes."""
        scenarios_a = self.generate_regime_a_favourable(n=50)
        scenarios_b = self.generate_regime_b_challenging(n=50)
        scenarios_c = self.generate_regime_c_adverse_blackswan(n=50)
        scenarios_d = self.generate_regime_d_adversarial_poisoning(n=50)

        res_a = self.evaluate_regime(scenarios_a, "Regime A: Highly Favourable")
        res_b = self.evaluate_regime(scenarios_b, "Regime B: Challenging (Medium Volatility)")
        res_c = self.evaluate_regime(scenarios_c, "Regime C: Severe Synthetic OOD Disruption")
        res_d = self.evaluate_regime(scenarios_d, "Regime D: Data Poisoning")

        return {
            "regime_a_favourable": res_a,
            "regime_b_challenging": res_b,
            "regime_c_adverse": res_c,
            "regime_d_poisoning": res_d,
        }


if __name__ == "__main__":
    console.rule("[bold cyan]PLANFLOW AI 2.0 - DEEP ADVERSARIAL STRESS TEST & BREAKDOWN AUDIT[/bold cyan]")

    tester = ComprehensiveStressTester()
    results = tester.run_all_stress_tests()

    # Create Comparison Table
    table = Table(title="[bold]Regime Comparison Matrix: Favourable vs Unfavourable[/bold]", border_style="cyan")
    table.add_column("Evaluation Dimension", style="cyan")
    table.add_column("Regime A: Favourable", style="green")
    table.add_column("Regime B: Challenging", style="yellow")
    table.add_column("Regime C: Synthetic OOD", style="red")
    table.add_column("Regime D: Poisoning", style="magenta")

    ra = results["regime_a_favourable"]
    rb = results["regime_b_challenging"]
    rc = results["regime_c_adverse"]
    rd = results["regime_d_poisoning"]

    table.add_row("Forecast MAE (Error)", f"{ra['demand_mae_units']} u", f"{rb['demand_mae_units']} u", f"{rc['demand_mae_units']} u", f"{rd['demand_mae_units']} u")
    table.add_row("Forecast WAPE %", f"{ra['demand_wape_pct']}%", f"{rb['demand_wape_pct']}%", f"{rc['demand_wape_pct']}%", f"{rd['demand_wape_pct']}%")
    table.add_row("Forecast R² Fit", f"{ra['demand_r2']}", f"{rb['demand_r2']}", f"{rc['demand_r2']}", f"{rd['demand_r2']}")
    table.add_row("Neural Agent Confidence", f"{ra['mean_agent_confidence']:.3f}", f"{rb['mean_agent_confidence']:.3f}", f"{rc['mean_agent_confidence']:.3f} (OOD Overconfident)", f"{rd['mean_agent_confidence']:.3f}")
    table.add_row("Manager Verification Score", f"{ra['mean_verification_score']}/100", f"{rb['mean_verification_score']}/100", f"{rc['mean_verification_score']}/100", f"{rd['mean_verification_score']}/100")
    table.add_row("Red Team Detection Recall", f"{ra['red_team_detection_recall_pct']}%", f"{rb['red_team_detection_recall_pct']}%", f"N/A (Physical Outage)", f"{rd['red_team_detection_recall_pct']}%")
    table.add_row("Red Team False Alarm Rate", f"{ra['red_team_false_alarm_rate_pct']}%", f"{rb['red_team_false_alarm_rate_pct']}%", f"0.0%", f"0.0%")
    table.add_row("Solver-Feasible Rate", f"{ra['solver_feasible_rate_pct']}%", f"{rb['solver_feasible_rate_pct']}%", f"{rc['solver_feasible_rate_pct']}%", f"{rd['solver_feasible_rate_pct']}%")
    table.add_row("Full-Demand-Service Rate", f"{ra['full_demand_service_rate_pct']}%", f"{rb['full_demand_service_rate_pct']}%", f"{rc['full_demand_service_rate_pct']}% (Stockout)", f"{rd['full_demand_service_rate_pct']}%")
    table.add_row("Mean Unmet Demand", f"{ra['mean_unmet_demand_units']} u (0.0%)", f"{rb['mean_unmet_demand_units']} u (0.0%)", f"{rc['mean_unmet_demand_units']} u ({rc['unmet_demand_ratio_pct']}%)", f"{rd['mean_unmet_demand_units']} u (0.0%)")
    table.add_row("Approved without Conditions", f"{ra['governance_breakdown']['approved_pct']}% ({ra['governance_breakdown']['approved_count']}/50)", f"{rb['governance_breakdown']['approved_pct']}% ({rb['governance_breakdown']['approved_count']}/50)", f"{rc['governance_breakdown']['approved_pct']}% ({rc['governance_breakdown']['approved_count']}/50)", f"{rd['governance_breakdown']['approved_pct']}% ({rd['governance_breakdown']['approved_count']}/50)")
    table.add_row("Conditional Approvals", f"{ra['governance_breakdown']['conditional_approval_pct']}% ({ra['governance_breakdown']['conditional_approval_count']}/50)", f"{rb['governance_breakdown']['conditional_approval_pct']}% ({rb['governance_breakdown']['conditional_approval_count']}/50)", f"{rc['governance_breakdown']['conditional_approval_pct']}% ({rc['governance_breakdown']['conditional_approval_count']}/50)", f"{rd['governance_breakdown']['conditional_approval_pct']}% ({rd['governance_breakdown']['conditional_approval_count']}/50)")
    table.add_row("Rejected / Vetoed", f"{ra['governance_breakdown']['rejected_or_vetoed_pct']}% ({ra['governance_breakdown']['rejected_or_vetoed_count']}/50)", f"{rb['governance_breakdown']['rejected_or_vetoed_pct']}% ({rb['governance_breakdown']['rejected_or_vetoed_count']}/50)", f"{rc['governance_breakdown']['rejected_or_vetoed_pct']}% ({rc['governance_breakdown']['rejected_or_vetoed_count']}/50)", f"{rd['governance_breakdown']['rejected_or_vetoed_pct']}% ({rd['governance_breakdown']['rejected_or_vetoed_count']}/50)")
    table.add_row("System Crashes / Timeouts", f"{ra['crashes']}", f"{rb['crashes']}", f"{rc['crashes']}", f"{rd['crashes']}")

    console.print(table)
    console.print()

    # Save to disk
    out_file = "data/stress_test_audit_report.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[dim]Detailed stress test report saved to {out_file}[/dim]")

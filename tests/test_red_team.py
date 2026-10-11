"""
Unit tests for Adversarial Red-Team Challenger Protocol
"""

from src.core.contracts import (
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RiskLevel,
)
from src.agents.red_team_challenger import RedTeamChallengerAgent


def test_red_team_falsification_on_stale_data():
    """Validates that the Red-Team Challenger falsifies consensus relying on stale CRM records."""
    challenger = RedTeamChallengerAgent()

    demand = ConsensusDemandContract(
        region="Germany",
        product_line="Automotive Solutions",
        horizon_month="2026-11",
        ml_forecast_units=10000.0,
        pipeline_uplift_units=2500.0,
        backlog_units=4000.0,
        scenario_adjustment_units=0.0,
        consensus_demand_units=12500.0,
        consensus_revenue_usd=15000000.0,
        confidence_score=0.90,
        peer_revisions_count=0,
        consensus_rationale="Uncorrected",
    )
    factory = FactoryAllocationContract(
        total_demand_units=12500.0,
        factory_allocations={"Factory A (Stuttgart)": 12500.0},
        factory_capacities={"Factory A (Stuttgart)": 12000.0},
        factory_utilizations={"Factory A (Stuttgart)": 104.2},
        overloaded_factories=["Factory A (Stuttgart)"],
        transfer_shifts=[],
        confidence_score=0.92,
    )
    supply = SupplyChainAuditContract(
        bill_of_materials={"MAT-M1": 1.0},
        required_materials={"MAT-M1": 12500.0},
        current_inventory={"MAT-M1": 15000.0},
        in_transit_inventory={"MAT-M1": 0.0},
        shortages={"MAT-M1": 0.0},
        material_shortage_detected=False,
        critical_materials_at_risk=[],
        delivery_risk=RiskLevel.LOW,
        total_expedite_cost_usd=0.0,
        confidence_score=0.95,
    )

    # 35% stale pipeline
    row_dict = {
        "unweighted_pipeline_usd": 10_000_000.0,
        "stale_deals_usd": 3_500_000.0,
        "effective_capacity": 10500.0,
        "nominal_capacity": 12000.0,
        "supplier_delay_days": 14,
    }

    challenges = challenger.challenge_consensus(demand, factory, supply, row_dict)
    assert len(challenges) == 3

    # Assert that stale pipeline was caught and falsified
    stale_ch = [c for c in challenges if "Pipeline" in c.assertion_subject][0]
    assert stale_ch.verdict == "FALSIFIED"
    assert "Correlated error detected" in stale_ch.evidence_challenge_details

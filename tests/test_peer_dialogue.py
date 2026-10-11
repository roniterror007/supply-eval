"""
Unit tests for Friendly Peer Dialogue and Mutual Correction Engine
"""

from src.core.contracts import (
    ForecastContract,
    OpportunityContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RiskLevel,
)
from src.agents.peer_dialogue import PeerDialogueEngine


def test_peer_mutual_correction_trigger():
    """Validates that peer agents exchange facts and correct stale pipeline numbers."""
    engine = PeerDialogueEngine()

    forecast = ForecastContract(
        product_line="Automotive Solutions",
        region="Germany",
        horizon_month="2026-11",
        point_forecast_units=10000.0,
        p10_units=8500.0,
        p50_units=10000.0,
        p90_units=11500.0,
        baseline_revenue_usd=12000000.0,
        confidence_score=0.90,
        primary_drivers=["Trend"],
    )
    # Opportunity has high stale pipeline
    opportunity = OpportunityContract(
        region="Germany",
        product_line="Automotive Solutions",
        total_pipeline_usd=5000000.0,
        stale_deals_usd=1200000.0,  # > 500k triggers correction
        high_value_deals_usd=2000000.0,
        at_risk_deals_count=3,
        net_probability_weighted_units=3000.0,
        flagged_deal_ids=["OPP-1", "OPP-2"],
        confidence_score=0.88,
    )
    demand = ConsensusDemandContract(
        region="Germany",
        product_line="Automotive Solutions",
        horizon_month="2026-11",
        ml_forecast_units=10000.0,
        pipeline_uplift_units=1500.0,  # > 1000 triggers critique
        backlog_units=4000.0,
        scenario_adjustment_units=0.0,
        consensus_demand_units=11500.0,
        consensus_revenue_usd=13800000.0,
        confidence_score=0.90,
        peer_revisions_count=0,
        consensus_rationale="Initial unrevised",
    )
    factory = FactoryAllocationContract(
        total_demand_units=11500.0,
        factory_allocations={"Factory A (Stuttgart)": 11500.0},
        factory_capacities={"Factory A (Stuttgart)": 12000.0},
        factory_utilizations={"Factory A (Stuttgart)": 95.8},
        overloaded_factories=[],
        transfer_shifts=[],
        confidence_score=0.92,
    )
    supply = SupplyChainAuditContract(
        bill_of_materials={"MAT-M1": 1.0},
        required_materials={"MAT-M1": 11500.0},
        current_inventory={"MAT-M1": 15000.0},
        in_transit_inventory={"MAT-M1": 0.0},
        shortages={"MAT-M1": 0.0},
        material_shortage_detected=False,
        critical_materials_at_risk=[],
        delivery_risk=RiskLevel.LOW,
        total_expedite_cost_usd=0.0,
        confidence_score=0.95,
    )

    row_dict = {
        "unit_price_usd": 1200.0,
        "supplier_delay_days": 0,
    }

    revised_demand, revised_factory, revised_supply, msgs = engine.conduct_collaborative_review(
        forecast=forecast,
        opportunity=opportunity,
        demand=demand,
        factory=factory,
        supply=supply,
        row_dict=row_dict,
    )

    # Asserts that correction was made
    assert revised_demand.peer_revisions_count >= 1
    assert any(m.is_correction for m in msgs)
    # Asserts that factory allocation was re-aligned to match the new demand
    assert sum(revised_factory.factory_allocations.values()) == revised_demand.consensus_demand_units

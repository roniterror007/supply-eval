"""
Unit tests for Deterministic Verification Engine & Enterprise Grounding Checker
"""

import pytest
from src.core.contracts import (
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RecommendationPlanContract,
    ActionRecommendation,
    RiskLevel,
)
from src.verification.math_verifier import DeterministicMathVerifier
from src.verification.grounding_checker import EnterpriseGroundingChecker


def create_valid_contracts():
    demand = ConsensusDemandContract(
        region="Germany",
        product_line="Automotive Solutions",
        horizon_month="2026-11",
        ml_forecast_units=10000.0,
        pipeline_uplift_units=1000.0,
        backlog_units=4000.0,
        scenario_adjustment_units=1000.0,
        consensus_demand_units=12000.0,
        consensus_revenue_usd=14400000.0,
        confidence_score=0.90,
        peer_revisions_count=0,
        consensus_rationale="Baseline test",
    )
    factory = FactoryAllocationContract(
        total_demand_units=12000.0,
        factory_allocations={"Factory A (Stuttgart)": 12000.0},
        factory_capacities={"Factory A (Stuttgart)": 12000.0},
        factory_utilizations={"Factory A (Stuttgart)": 100.0},
        overloaded_factories=[],
        transfer_shifts=[],
        confidence_score=0.95,
    )
    supply = SupplyChainAuditContract(
        bill_of_materials={"MAT-M1": 1.0},
        required_materials={"MAT-M1": 12000.0},
        current_inventory={"MAT-M1": 12000.0},
        in_transit_inventory={"MAT-M1": 0.0},
        shortages={"MAT-M1": 0.0},
        material_shortage_detected=False,
        critical_materials_at_risk=[],
        delivery_risk=RiskLevel.LOW,
        total_expedite_cost_usd=0.0,
        confidence_score=0.92,
    )
    recommendation = RecommendationPlanContract(
        recommendations=[
            ActionRecommendation(
                priority_rank=1,
                action_type="MAINTAIN_STANDARD_SCHEDULE",
                description="Standard run",
                target_entity="Factory A (Stuttgart)",
                cost_usd=0.0,
                protected_revenue_usd=0.0,
                risk_mitigation_score=1.0,
            )
        ],
        total_cost_usd=0.0,
        total_protected_revenue_usd=0.0,
        net_benefit_usd=0.0,
        post_recommendation_delivery_risk=RiskLevel.LOW,
        expected_capacity_normalized_pct=100.0,
        confidence_score=0.95,
    )
    return demand, factory, supply, recommendation


def test_deterministic_verifier_pass():
    """Validates that correct mathematical equations yield score 100% and validity True."""
    verifier = DeterministicMathVerifier()
    demand, factory, supply, rec = create_valid_contracts()
    res = verifier.verify_plan(demand, factory, supply, rec)
    assert res.is_valid is True
    assert res.score == 100.0
    assert res.conservation_passed is True
    assert res.capacity_math_passed is True
    assert res.inventory_math_passed is True
    assert len(res.violations) == 0


def test_deterministic_verifier_catches_conservation_break():
    """Corrupts factory allocation sum and asserts math verifier detects violation."""
    verifier = DeterministicMathVerifier()
    demand, factory, supply, rec = create_valid_contracts()
    # Artificially alter factory allocation to break conservation
    factory.factory_allocations["Factory A (Stuttgart)"] = 10000.0  # discrepancy of 2,000 units
    res = verifier.verify_plan(demand, factory, supply, rec)
    assert res.is_valid is False
    assert res.conservation_passed is False
    assert res.score < 85.0
    assert any("Conservation of Units VIOLATED" in v for v in res.violations)


def test_grounding_checker():
    """Validates that real entities pass and fake entities fail grounding."""
    checker = EnterpriseGroundingChecker()
    # Real entities
    res_valid = checker.check_entities(
        business_line="Automotive Solutions",
        region="Germany",
        factory_names=["Factory A (Stuttgart)"],
        material_ids=["MAT-M1", "MAT-M2"],
    )
    assert res_valid["grounded"] is True

    # Hallucinated entities
    res_invalid = checker.check_entities(
        business_line="Space Mining Solutions",  # Hallucinated
        region="Atlantis",  # Hallucinated
        factory_names=["Factory X (Moon)"],  # Hallucinated
        material_ids=["MAT-UNOBTANIUM"],  # Hallucinated
    )
    assert res_invalid["grounded"] is False
    assert len(res_invalid["unrecognized_entities"]) == 4

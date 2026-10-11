"""
Unit tests for Adversarial Stress Testing, Two-Part Gate Governance, and Cryptographic Signatures
"""

from src.core.contracts import (
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RecommendationPlanContract,
    ActionRecommendation,
    RiskLevel,
    ApprovalStatus,
    ManagerAuditContract,
    JudgeDecisionContract,
)
from src.verification.math_verifier import DeterministicMathVerifier
from src.agents.judge_manager import JudgeManagerAgent
from src.agents.executive_boss import ExecutiveBossAgent


def test_two_part_gate_hard_failure_non_negotiable():
    """
    Validates that a hard mathematical violation causes status == 'FAIL',
    and soft verification score can NEVER override a hard violation.
    """
    verifier = DeterministicMathVerifier(tolerance_fraction=0.005)

    demand = ConsensusDemandContract(
        region="EMEA",
        product_line="High-Voltage Inverters",
        horizon_month="2026-10",
        ml_forecast_units=10000.0,
        pipeline_uplift_units=0.0,
        backlog_units=0.0,
        scenario_adjustment_units=0.0,
        consensus_demand_units=10000.0,
        consensus_revenue_usd=12000000.0,
        confidence_score=0.95,
        peer_revisions_count=0,
        consensus_rationale="Test",
    )

    # Violate conservation: allocated 15,000 != consensus demand 10,000
    factory_bad = FactoryAllocationContract(
        total_demand_units=15000.0,
        factory_allocations={"Factory A (Stuttgart)": 15000.0},
        factory_capacities={"Factory A (Stuttgart)": 20000.0},
        factory_utilizations={"Factory A (Stuttgart)": 75.0},
        overloaded_factories=[],
        transfer_shifts=[],
        confidence_score=0.95,
    )

    supply = SupplyChainAuditContract(
        bill_of_materials={"MAT-M1": 1.0},
        required_materials={"MAT-M1": 10000.0},
        current_inventory={"MAT-M1": 12000.0},
        in_transit_inventory={"MAT-M1": 0.0},
        shortages={"MAT-M1": 0.0},
        material_shortage_detected=False,
        critical_materials_at_risk=[],
        delivery_risk=RiskLevel.LOW,
        total_expedite_cost_usd=0.0,
        confidence_score=0.95,
    )

    rec = RecommendationPlanContract(
        recommendations=[
            ActionRecommendation(
                priority_rank=1,
                action_type="NOMINAL",
                description="Nominal",
                target_entity="Factory A",
                cost_usd=10000.0,
                protected_revenue_usd=50000.0,
                risk_mitigation_score=0.9,
            )
        ],
        total_cost_usd=10000.0,
        total_protected_revenue_usd=50000.0,
        net_benefit_usd=40000.0,
        post_recommendation_delivery_risk=RiskLevel.LOW,
        expected_capacity_normalized_pct=85.0,
        confidence_score=0.95,
    )

    result = verifier.verify_plan(demand, factory_bad, supply, rec)

    assert result.conservation_passed is False
    assert result.hard_constraints_passed is False
    assert result.judge_decision is not None
    assert result.judge_decision.status == "FAIL"
    assert result.judge_decision.hard_constraints_passed is False
    assert len(result.judge_decision.hard_failures) > 0


def test_executive_boss_ed25519_signature_and_veto():
    """Validates real Ed25519 signing and immediate veto of failed audit plans."""
    boss = ExecutiveBossAgent()

    # Case A: Failed audit -> Veto
    failed_judge_decision = JudgeDecisionContract(
        hard_constraints_passed=False,
        hard_failures=["CONSERVATION_OF_UNITS_VIOLATED"],
        soft_verification_score=60.0,
        evidence_coverage=0.7,
        max_material_uncertainty=0.4,
        status="FAIL",
    )

    failed_audit = ManagerAuditContract(
        audit_id="AUD-FAIL-01",
        all_contracts_passed=False,
        unit_conservation_passed=False,
        capacity_math_passed=True,
        inventory_math_passed=True,
        grounding_check_passed=True,
        peer_corrections_verified_count=1,
        verification_score=60.0,
        identified_discrepancies=["Conservation failed"],
        manager_assessment="Rejected due to conservation failure",
        forward_to_boss=False,
        judge_decision=failed_judge_decision,
        hard_constraints_passed=False,
        hard_failures=["CONSERVATION_OF_UNITS_VIOLATED"],
    )

    rec = RecommendationPlanContract(
        recommendations=[],
        total_cost_usd=50000.0,
        total_protected_revenue_usd=100000.0,
        net_benefit_usd=50000.0,
        post_recommendation_delivery_risk=RiskLevel.LOW,
        expected_capacity_normalized_pct=85.0,
        confidence_score=0.9,
    )

    veto_decision = boss.process({
        "manager_audit_contract": failed_audit,
        "recommendation_contract": rec,
    })

    assert veto_decision.status == ApprovalStatus.REJECTED
    assert veto_decision.approved_budget_usd == 0.0
    assert veto_decision.signature_contract is not None
    # Verify cryptographic signature
    assert ExecutiveBossAgent.verify_signature(veto_decision.signature_contract, boss.public_key) is True

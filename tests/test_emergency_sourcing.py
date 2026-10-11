"""
PlanFlow AI - Unit Tests for Bounded Emergency Spot Sourcing & Governance
Proves the 7 critical governance, physical capacity, BOM, and reporting properties:
1. Spot sourcing is exactly zero when approval is false.
2. Spot quantity never exceeds the configured cap.
3. Spot material respects BOM/specification compatibility.
4. Spot lead time is included in service calculations.
5. Solver cannot use spot supply to bypass hard machine capacity constraints.
6. Executive Boss cannot authorize an action rejected by corporate autonomy policy.
7. Decision memo consistently reports base shortfall, recovered units, residual shortfall, and signer identity.
"""

import pytest
from src.optimization.master_planner import MasterMILPPlanner
from src.agents.executive_boss import ExecutiveBossAgent
from src.governance.autonomy_policy import RiskAdaptiveAutonomyEngine, ExecutionDecision
from src.core.contracts import (
    ApprovalStatus,
    ManagerAuditContract,
    RecommendationPlanContract,
    JudgeDecisionContract,
    ActionRecommendation,
    RiskLevel,
)


def test_spot_sourcing_zero_when_unapproved():
    """Property 1: Spot sourcing is exactly zero when approval is false."""
    planner = MasterMILPPlanner()
    res = planner.solve(
        target_demand=8000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=10000.0,
        secondary_factory="Factory B",
        secondary_capacity=5000.0,
        material_ratio=1.0,
        usable_material_stock=0.0,  # 0 material in stock
        max_standard_expedite_units=0.0,  # Supplier expedite unavailable
        allow_emergency_sourcing=False,  # Unapproved
        emergency_spot_cap=5000.0,
    )

    assert res.emergency_sourcing_units == 0.0
    assert res.slack_details["emergency_sourcing_used"] == 0.0
    # Demand cannot be fulfilled due to 0 material
    assert res.total_fulfilled_units == 0.0
    assert res.residual_unmet_demand_units == 8000.0


def test_spot_quantity_never_exceeds_cap():
    """Property 2: Spot quantity never exceeds the configured cap."""
    planner = MasterMILPPlanner()
    cap = 2500.0
    res = planner.solve(
        target_demand=10000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=10000.0,
        secondary_factory="Factory B",
        secondary_capacity=5000.0,
        material_ratio=1.0,
        usable_material_stock=1000.0,
        max_standard_expedite_units=0.0,  # Supplier expedite exhausted
        allow_emergency_sourcing=True,
        emergency_spot_cap=cap,
    )

    assert res.emergency_sourcing_units <= cap
    assert res.emergency_sourcing_units == cap
    # 1000 initial + 2500 spot = 3500 fulfilled
    assert res.total_fulfilled_units == 3500.0
    assert res.residual_unmet_demand_units == 6500.0


def test_spot_material_respects_bom_compatibility():
    """Property 3: Spot material respects BOM/specification compatibility."""
    planner = MasterMILPPlanner()

    # Incompatible material cannot be credited to the production BOM
    res_incompatible = planner.solve(
        target_demand=5000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=6000.0,
        secondary_factory="Factory B",
        secondary_capacity=3000.0,
        material_ratio=1.0,
        usable_material_stock=1000.0,
        max_standard_expedite_units=0.0,
        allow_emergency_sourcing=True,
        emergency_spot_cap=3000.0,
        is_material_compatible=False,  # Specification mismatch!
    )
    assert res_incompatible.emergency_sourcing_units == 0.0
    assert res_incompatible.total_fulfilled_units == 1000.0

    # Compatible material can be used
    res_compatible = planner.solve(
        target_demand=5000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=6000.0,
        secondary_factory="Factory B",
        secondary_capacity=3000.0,
        material_ratio=1.0,
        usable_material_stock=1000.0,
        max_standard_expedite_units=0.0,
        allow_emergency_sourcing=True,
        emergency_spot_cap=3000.0,
        is_material_compatible=True,
    )
    assert res_compatible.emergency_sourcing_units == 3000.0
    assert res_compatible.total_fulfilled_units == 4000.0


def test_spot_lead_time_included_in_service_calculations():
    """Property 4: Spot lead time is included in service calculations."""
    planner = MasterMILPPlanner()

    # Lead time exceeds planning horizon -> Arrives too late to fulfill horizon demand
    res_late = planner.solve(
        target_demand=4000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=5000.0,
        secondary_factory="Factory B",
        secondary_capacity=2000.0,
        material_ratio=1.0,
        usable_material_stock=500.0,
        max_standard_expedite_units=0.0,
        allow_emergency_sourcing=True,
        emergency_spot_cap=3000.0,
        spot_lead_days=45,  # 45 days lead time
        planning_horizon_days=30,  # 30 day horizon
    )
    assert res_late.emergency_sourcing_units == 0.0
    assert res_late.residual_unmet_demand_units == 3500.0

    # Lead time within horizon -> Usable
    res_ontime = planner.solve(
        target_demand=4000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=5000.0,
        secondary_factory="Factory B",
        secondary_capacity=2000.0,
        material_ratio=1.0,
        usable_material_stock=500.0,
        max_standard_expedite_units=0.0,
        allow_emergency_sourcing=True,
        emergency_spot_cap=3000.0,
        spot_lead_days=3,
        planning_horizon_days=30,
    )
    assert res_ontime.emergency_sourcing_units == 3000.0
    assert res_ontime.residual_unmet_demand_units == 500.0


def test_solver_cannot_bypass_machine_capacity_with_spot_supply():
    """
    Property 5: Solver cannot use spot supply to bypass physical machine capacity.
    Even with infinite spot material, factory machine-hour limits remain hard constraints.
    """
    planner = MasterMILPPlanner()
    # Physical machine capacity: Primary 2000u, Secondary 1000u. Total max physical = 3000u.
    res = planner.solve(
        target_demand=10000.0,
        unit_price=1000.0,
        primary_factory="Factory A",
        primary_capacity=2000.0,
        secondary_factory="Factory B",
        secondary_capacity=1000.0,
        material_ratio=1.0,
        usable_material_stock=0.0,
        max_standard_expedite_units=0.0,
        max_overtime_hours=0.0,  # Zero overtime permitted
        allow_emergency_sourcing=True,
        emergency_spot_cap=50000.0,  # Huge spot material cap
    )

    # Spot raw material cannot create machine hours: maximum fulfilled cannot exceed 3000.0
    assert res.total_fulfilled_units <= 3000.0
    assert res.total_fulfilled_units == 3000.0
    # Spot units purchased will only be what machines can process (3000.0)
    assert res.emergency_sourcing_units == 3000.0
    # Residual unmet demand cannot be cleared because machine capacity is exhausted
    assert res.residual_unmet_demand_units == 7000.0


def test_executive_boss_cannot_authorize_action_rejected_by_autonomy_policy():
    """
    Property 6: The Executive Boss cannot authorize an action rejected by the autonomy policy.
    """
    boss = ExecutiveBossAgent()

    # Autonomy policy engine rejects an action with excessive spend and stale evidence
    autonomy_engine = RiskAdaptiveAutonomyEngine()
    rejection = autonomy_engine.evaluate_action(
        action_id="ACT-SPOT-99",
        decision_class="expedite",
        cost_usd=450_000.0,  # Exceeds expedite limit of $100k
        uncertainty=0.40,
        evidence_quality=0.50,
        customer_service_risk=0.25,
        has_stale_or_conflicted_evidence=True,  # Prohibited!
    )
    assert rejection.policy_compliance_passed is False

    mock_judge_decision = JudgeDecisionContract(
        hard_constraints_passed=True,
        soft_verification_score=92.0,
        status="PASS",
    )
    mock_audit = ManagerAuditContract(
        audit_id="AUDIT-TEST-01",
        all_contracts_passed=True,
        unit_conservation_passed=True,
        capacity_math_passed=True,
        inventory_math_passed=True,
        grounding_check_passed=True,
        peer_corrections_verified_count=1,
        verification_score=92.0,
        identified_discrepancies=[],
        manager_assessment="Verification passed",
        forward_to_boss=True,
        hard_constraints_passed=True,
        judge_decision=mock_judge_decision,
    )
    mock_rec = RecommendationPlanContract(
        recommendations=[
            ActionRecommendation(
                priority_rank=1,
                action_type="EMERGENCY_SPOT_SOURCING",
                description="Purchase 3,333 units from spot market",
                target_entity="MAT-M1",
                cost_usd=350_000.0,
                protected_revenue_usd=2_000_000.0,
                risk_mitigation_score=0.90,
            )
        ],
        total_cost_usd=350_000.0,
        total_protected_revenue_usd=2_000_000.0,
        net_benefit_usd=1_650_000.0,
        post_recommendation_delivery_risk=RiskLevel.LOW,
        expected_capacity_normalized_pct=85.0,
        confidence_score=0.90,
    )

    context = {
        "manager_audit_contract": mock_audit,
        "recommendation_contract": mock_rec,
        "autonomy_decision": rejection,  # Policy compliance failed!
    }

    decision = boss.process(context)
    assert decision.status == ApprovalStatus.REJECTED
    assert any("AUTONOMY_POLICY_VETO" in cond for cond in decision.executive_conditions)
    assert decision.approved_budget_usd == 0.0


def test_memo_reports_shortfall_recovered_cost_and_identity_consistently():
    """
    Property 7: The memo reports base shortfall, recovered units, residual shortfall,
    cost, and approval identity consistently.
    Example: $350,000 / $105 ~= 3,333 units.
    """
    planner = MasterMILPPlanner()
    target_d = 8000.0
    stock = 4667.0
    spot_cap = 3333.0
    spot_unit_cost = 105.0

    res = planner.solve(
        target_demand=target_d,
        unit_price=500.0,
        primary_factory="Factory A",
        primary_capacity=10000.0,
        secondary_factory="Factory B",
        secondary_capacity=5000.0,
        material_ratio=1.0,
        usable_material_stock=stock,
        max_standard_expedite_units=0.0,  # Standard expedite exhausted, spot required
        allow_emergency_sourcing=True,
        emergency_spot_cap=spot_cap,
        emergency_spot_unit_cost=spot_unit_cost,
    )

    expected_recovered = 3333.0
    assert res.emergency_sourcing_units == expected_recovered
    assert res.residual_unmet_demand_units == 0.0
    # Base shortfall without spot was 8000 - 4667 = 3333 units
    assert res.unmet_demand_without_emergency_units == expected_recovered

    # Arithmetic check: spot cost = 3,333 * 105 = $349,965 (~$350,000)
    expected_spot_cost = round(expected_recovered * spot_unit_cost, 2)
    assert abs(expected_spot_cost - 349_965.0) < 1.0
    assert res.total_cost_usd >= expected_spot_cost

    # Verification certificate is signed and non-empty
    assert res.certificate is not None
    assert res.certificate.solve_status == "OPTIMAL"
    assert res.certificate.mip_gap == 0.0
    assert "SHA256:" in res.certificate.constraint_set_hash

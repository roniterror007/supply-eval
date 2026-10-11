"""
Unit tests for Risk-Adaptive Autonomy and Approval Engine
"""

from src.governance.autonomy_policy import RiskAdaptiveAutonomyEngine
from src.governance.approval_simulator import ApprovalSimulator


def test_autonomy_policy_routing():
    """Validates that low-risk actions auto-execute and high-risk/stale actions escalate."""
    engine = RiskAdaptiveAutonomyEngine()

    # 1. Low risk forecast triage -> auto_execute
    d1 = engine.evaluate_action(
        action_id="A1",
        decision_class="forecast",
        cost_usd=50000.0,
        uncertainty=0.10,
        evidence_quality=0.95,
        customer_service_risk=0.05,
    )
    assert d1.can_auto_execute is True
    assert d1.requires_human_escalation is False

    # 2. Large capital expenditure -> requires human escalation
    d2 = engine.evaluate_action(
        action_id="A2",
        decision_class="capacity_expansion",
        cost_usd=500000.0,
        uncertainty=0.08,
        evidence_quality=0.95,
        customer_service_risk=0.02,
    )
    assert d2.can_auto_execute is False
    assert d2.requires_human_escalation is True

    # 3. Action based on stale evidence -> strict prohibition of auto-execute!
    d3 = engine.evaluate_action(
        action_id="A3",
        decision_class="forecast",
        cost_usd=1000.0,
        uncertainty=0.05,
        evidence_quality=0.90,
        customer_service_risk=0.01,
        has_stale_or_conflicted_evidence=True,
    )
    assert d3.can_auto_execute is False
    assert d3.requires_human_escalation is True
    assert "stale" in d3.escalation_reason.lower()

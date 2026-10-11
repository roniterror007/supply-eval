"""
PlanFlow AI - Risk-Adaptive Autonomy Policy Engine
Dynamically determines whether planning actions should be auto-executed, conditionally executed,
or escalated for human review based on financial risk, reversibility, evidence quality, and supply tier.
"""

from typing import Literal, List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from src.core.logger import get_logger

logger = get_logger("AutonomyPolicy")

DecisionClass = Literal[
    "forecast",
    "inventory_rebalance",
    "expedite",
    "production_reschedule",
    "supplier_substitution",
    "capacity_expansion",
]

AllowedMode = Literal["recommend", "conditional_execute", "auto_execute"]


class AutonomyPolicyContract(BaseModel):
    """Governs execution delegation thresholds per enterprise operational category."""
    decision_class: DecisionClass
    allowed_mode: AllowedMode
    max_financial_exposure: float
    max_customer_service_risk: float
    max_uncertainty: float
    min_evidence_quality: float
    required_approvers: List[str]
    reversal_window_hours: Optional[int] = None


@dataclass
class ExecutionDecision:
    action_id: str
    decision_class: str
    assigned_mode: AllowedMode
    can_auto_execute: bool
    requires_human_escalation: bool
    required_approvers: List[str]
    escalation_reason: Optional[str]
    policy_compliance_passed: bool


class RiskAdaptiveAutonomyEngine:
    """Evaluates proposed actions against corporate governance thresholds and evidence quality."""

    DEFAULT_POLICIES: Dict[str, AutonomyPolicyContract] = {
        "forecast": AutonomyPolicyContract(
            decision_class="forecast",
            allowed_mode="auto_execute",
            max_financial_exposure=5_000_000.0,
            max_customer_service_risk=0.30,
            max_uncertainty=0.35,
            min_evidence_quality=0.65,
            required_approvers=[],
            reversal_window_hours=24,
        ),
        "inventory_rebalance": AutonomyPolicyContract(
            decision_class="inventory_rebalance",
            allowed_mode="conditional_execute",
            max_financial_exposure=250_000.0,
            max_customer_service_risk=0.15,
            max_uncertainty=0.20,
            min_evidence_quality=0.80,
            required_approvers=["Plant Logistics Manager"],
            reversal_window_hours=48,
        ),
        "expedite": AutonomyPolicyContract(
            decision_class="expedite",
            allowed_mode="conditional_execute",
            max_financial_exposure=100_000.0,
            max_customer_service_risk=0.10,
            max_uncertainty=0.15,
            min_evidence_quality=0.85,
            required_approvers=["Procurement Director"],
            reversal_window_hours=12,
        ),
        "production_reschedule": AutonomyPolicyContract(
            decision_class="production_reschedule",
            allowed_mode="conditional_execute",
            max_financial_exposure=500_000.0,
            max_customer_service_risk=0.10,
            max_uncertainty=0.15,
            min_evidence_quality=0.85,
            required_approvers=["Operations VP"],
            reversal_window_hours=72,
        ),
        "supplier_substitution": AutonomyPolicyContract(
            decision_class="supplier_substitution",
            allowed_mode="recommend",
            max_financial_exposure=50_000.0,
            max_customer_service_risk=0.05,
            max_uncertainty=0.10,
            min_evidence_quality=0.90,
            required_approvers=["Procurement Director", "Quality VP"],
            reversal_window_hours=None,
        ),
        "capacity_expansion": AutonomyPolicyContract(
            decision_class="capacity_expansion",
            allowed_mode="recommend",
            max_financial_exposure=0.0,
            max_customer_service_risk=0.05,
            max_uncertainty=0.05,
            min_evidence_quality=0.95,
            required_approvers=["Executive Committee", "CFO"],
            reversal_window_hours=None,
        ),
    }

    def __init__(self, custom_policies: Optional[Dict[str, AutonomyPolicyContract]] = None):
        self.policies = custom_policies or self.DEFAULT_POLICIES

    def evaluate_action(
        self,
        action_id: str,
        decision_class: DecisionClass,
        cost_usd: float,
        uncertainty: float,
        evidence_quality: float,
        customer_service_risk: float,
        has_stale_or_conflicted_evidence: bool = False,
    ) -> ExecutionDecision:
        """Determines execution permissions for an action."""
        policy = self.policies.get(decision_class, self.policies["production_reschedule"])

        # Hard Rule: Stale or contradictory evidence strictly prohibits auto-execution
        if has_stale_or_conflicted_evidence:
            return ExecutionDecision(
                action_id=action_id,
                decision_class=decision_class,
                assigned_mode="recommend",
                can_auto_execute=False,
                requires_human_escalation=True,
                required_approvers=policy.required_approvers or ["Supply Chain VP"],
                escalation_reason="Action rests on stale, unconfirmed, or contradictory evidence.",
                policy_compliance_passed=False,
            )

        # Evaluate threshold limits
        violations = []
        if cost_usd > policy.max_financial_exposure and policy.allowed_mode != "auto_execute":
            violations.append(f"Financial spend ${cost_usd:,.0f} exceeds exposure limit (${policy.max_financial_exposure:,.0f})")
        if uncertainty > policy.max_uncertainty:
            violations.append(f"Uncertainty ({uncertainty:.2f}) exceeds policy ceiling ({policy.max_uncertainty:.2f})")
        if evidence_quality < policy.min_evidence_quality:
            violations.append(f"Evidence quality ({evidence_quality:.2f}) is below minimum ({policy.min_evidence_quality:.2f})")
        if customer_service_risk > policy.max_customer_service_risk:
            violations.append(f"Customer service risk ({customer_service_risk:.2f}) exceeds acceptable tolerance")

        if violations or policy.allowed_mode == "recommend":
            return ExecutionDecision(
                action_id=action_id,
                decision_class=decision_class,
                assigned_mode="recommend",
                can_auto_execute=False,
                requires_human_escalation=True,
                required_approvers=policy.required_approvers or ["Operations Director"],
                escalation_reason="; ".join(violations) if violations else "Action type requires formal human approval.",
                policy_compliance_passed=len(violations) == 0,
            )

        # Action is cleared for execution
        return ExecutionDecision(
            action_id=action_id,
            decision_class=decision_class,
            assigned_mode=policy.allowed_mode,
            can_auto_execute=(policy.allowed_mode == "auto_execute"),
            requires_human_escalation=False,
            required_approvers=[],
            escalation_reason=None,
            policy_compliance_passed=True,
        )

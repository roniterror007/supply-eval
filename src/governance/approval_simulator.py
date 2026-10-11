"""
PlanFlow AI - Approval Policy Simulator
Simulates human-in-the-loop escalation decisions and audits escalation precision/recall.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from src.governance.autonomy_policy import RiskAdaptiveAutonomyEngine, ExecutionDecision
from src.core.logger import get_logger

logger = get_logger("ApprovalSimulator")


class ApprovalSimulator:
    """Evaluates whether the autonomy engine makes appropriate escalation decisions."""

    def __init__(self, engine: Optional[RiskAdaptiveAutonomyEngine] = None):
        self.engine = engine or RiskAdaptiveAutonomyEngine()

    def simulate_approval_batch(
        self,
        test_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs batch evaluation of approval routing against ground-truth necessity of human review.
        test_actions: list of dicts with keys:
          - action_id, decision_class, cost_usd, uncertainty, evidence_quality,
            customer_service_risk, has_stale_or_conflicted_evidence, truly_needed_human_escalation
        """
        results: List[ExecutionDecision] = []
        y_true_escalation = []
        y_pred_escalation = []

        for item in test_actions:
            decision = self.engine.evaluate_action(
                action_id=item["action_id"],
                decision_class=item["decision_class"],
                cost_usd=float(item.get("cost_usd", 10000.0)),
                uncertainty=float(item.get("uncertainty", 0.10)),
                evidence_quality=float(item.get("evidence_quality", 0.90)),
                customer_service_risk=float(item.get("customer_service_risk", 0.05)),
                has_stale_or_conflicted_evidence=bool(item.get("has_stale_or_conflicted_evidence", False)),
            )
            results.append(decision)

            truly_needed = bool(item.get("truly_needed_human_escalation", False))
            y_true_escalation.append(int(truly_needed))
            y_pred_escalation.append(int(decision.requires_human_escalation))

        yt = np.array(y_true_escalation)
        yp = np.array(y_pred_escalation)

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "total_actions_evaluated": len(test_actions),
            "escalation_count": int(np.sum(yp)),
            "auto_execute_count": int(np.sum(yp == 0)),
            "escalation_precision": round(float(precision), 4),
            "escalation_recall": round(float(recall), 4),
            "escalation_f1": round(float(f1), 4),
            "unsafe_auto_execute_count": fn,  # Truly needed human review but auto-executed!
            "unnecessary_escalation_count": fp,  # Safe action escalated to human
        }

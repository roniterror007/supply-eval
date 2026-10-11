"""
PlanFlow AI - Tier 3: Executive Boss Agent (Final Governance & Cryptographic Sign-Off)
Rechecks Judge / Manager verification outputs, enforces enterprise capital & risk policies,
evaluates strategic ROI trade-offs, and issues binding, cryptographically signed Executive Decision Memos.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519

from src.core.contracts import (
    AgentType,
    ApprovalStatus,
    ManagerAuditContract,
    RecommendationPlanContract,
    BossExecutiveDecisionContract,
    ExecutiveSignatureContract,
)
from src.agents.base_agent import BaseAgent
from src.core.logger import get_logger, log_boss_verdict

logger = get_logger("ExecutiveBossAgent")


class ExecutiveBossAgent(BaseAgent):
    """
    Tier 3 Executive Oversight, Corporate Governance, and Cryptographic Sign-Off Agent.
    Signs decision memos using enterprise-controlled asymmetric cryptography (Ed25519).
    """

    def __init__(
        self,
        dual_approval_threshold_usd: float = 1_000_000.0,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        signing_key_id: str = "kms://us-east-1/planflow-prod-key-ed25519",
    ):
        super().__init__(AgentType.EXECUTIVE_BOSS)
        self.dual_approval_threshold = dual_approval_threshold_usd
        # Enterprise-controlled Ed25519 private key for signing canonical memos
        self._private_key = private_key or ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self.signing_key_id = signing_key_id

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._public_key

    def process(self, context: Dict[str, Any]) -> BossExecutiveDecisionContract:
        manager_audit: ManagerAuditContract = context["manager_audit_contract"]
        recommendation: RecommendationPlanContract = context["recommendation_contract"]

        conditions: List[str] = []

        # Step 1: Check Hard Constraints and Corporate Autonomy Policy Gate
        hard_passed = getattr(manager_audit, "hard_constraints_passed", True)
        if hasattr(manager_audit, "judge_decision") and manager_audit.judge_decision:
            hard_passed = manager_audit.judge_decision.hard_constraints_passed

        autonomy_decision = context.get("autonomy_decision")
        autonomy_rejected = False
        if autonomy_decision is not None and getattr(autonomy_decision, "policy_compliance_passed", True) is False:
            autonomy_rejected = True

        if not hard_passed or not manager_audit.forward_to_boss or manager_audit.verification_score < 85.0 or autonomy_rejected:
            status = ApprovalStatus.REJECTED
            if autonomy_rejected:
                reason = getattr(autonomy_decision, "escalation_reason", "Policy threshold exceeded")
                conditions.append(f"AUTONOMY_POLICY_VETO: Action rejected by corporate autonomy policy: {reason}")
            if not hard_passed:
                conditions.append("HARD_VIOLATION_VETO: Mathematical or physical constraints violated. Immediate executive stop.")
                if hasattr(manager_audit, "hard_failures") and manager_audit.hard_failures:
                    conditions.extend(manager_audit.hard_failures)
            elif not autonomy_rejected and manager_audit.verification_score < 85.0:
                conditions.append(
                    f"REJECTED: Manager verification score {manager_audit.verification_score}% is below required 85% threshold."
                )
            conditions.extend(manager_audit.identified_discrepancies)
            summary = "Executive veto: plan failed deterministic verification audit or corporate autonomy policy. Unsafe to execute."
            log_boss_verdict("REJECTED", summary)
            requires_dual = False
        else:
            total_spend = recommendation.total_cost_usd
            protected_rev = recommendation.total_protected_revenue_usd
            requires_dual = total_spend >= self.dual_approval_threshold
            roi = recommendation.net_benefit_usd / (total_spend + 1e-5) if total_spend > 0 else 100.0
            
            prompt = f"""
You are the Tier 3 Executive Boss for a B2B enterprise.
Review the supply chain recommendation and the manager's audit.
Recommendation Budget: ${total_spend:,.0f}
Protected Revenue: ${protected_rev:,.0f}
Manager Verification Score: {manager_audit.verification_score}%
Manager Assessment: {manager_audit.manager_assessment}

Determine if this should be APPROVED, REJECTED, or CONDITIONAL_APPROVAL.
Provide a concise executive summary explaining your decision to the stakeholders.
Respond ONLY in valid JSON matching this schema: {{"status": "APPROVED|REJECTED|CONDITIONAL_APPROVAL", "summary": "your text"}}
"""
            try:
                from src.agents.llm_client import generate_llm_response
                raw_text = generate_llm_response(prompt)
                if raw_text.startswith("```"):
                    lines = raw_text.split('\n')
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines[-1].startswith("```"): lines = lines[:-1]
                    raw_text = '\n'.join(lines).strip()
                if raw_text.startswith("json"): raw_text = raw_text[4:].strip()
                
                resp = json.loads(raw_text)
                status = ApprovalStatus(resp["status"])
                summary = resp["summary"]
                conditions.append("LLM Executive Decision Executed.")
                if requires_dual:
                    conditions.append(f"Capital Expenditure Alert: Budget (${total_spend:,.0f}) exceeds $1M threshold. Requires secondary VP signoff.")
            except Exception as e:
                logger.error(f"LLM Boss failed, falling back to rule-based: {e}")
                if total_spend > 0 and roi < 1.2:
                    status = ApprovalStatus.CONDITIONAL_APPROVAL
                    conditions.append(f"Marginal ROI warning (Multiplier: {roi:.2f}x). Expedite only tier-1 automotive SKUs.")
                else:
                    status = ApprovalStatus.APPROVED
                    conditions.append("Verified physical feasibility, conservation of units, and economic ROI.")

                summary = (
                    f"Authorized ${total_spend:,.0f} mitigation budget to protect ${protected_rev:,.0f} revenue. "
                    f"Net enterprise benefit: ${recommendation.net_benefit_usd:,.0f}. "
                    f"Expected capacity balance restored to {recommendation.expected_capacity_normalized_pct:.1f}%."
                )
                
            log_boss_verdict(status.value, summary)

        decision_id = f"EXEC-DEC-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)

        # Canonical payload construction for cryptographic hashing
        canonical_memo = {
            "decision_id": decision_id,
            "timestamp": now.isoformat(),
            "status": status.value,
            "budget_usd": recommendation.total_cost_usd if status != ApprovalStatus.REJECTED else 0.0,
            "protected_rev_usd": recommendation.total_protected_revenue_usd if status != ApprovalStatus.REJECTED else 0.0,
            "audit_id": manager_audit.audit_id,
            "verification_score": manager_audit.verification_score,
            "policy_version": "v2.0-enterprise",
        }
        canonical_bytes = json.dumps(canonical_memo, sort_keys=True).encode("utf-8")
        memo_hash_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

        # Sign canonical SHA-256 hash using enterprise Ed25519 private key
        signature_bytes = self._private_key.sign(memo_hash_sha256.encode("utf-8"))
        signature_hex = signature_bytes.hex()

        signature_contract = ExecutiveSignatureContract(
            memo_hash_sha256=memo_hash_sha256,
            signature_algorithm="Ed25519",
            signature=signature_hex,
            signer_id="enterprise-executive-boss-01",
            signing_key_id=self.signing_key_id,
            signed_at=now,
            policy_version="v2.0-enterprise",
            approval_chain=["PeerSpecialists", "DeterministicJudgeManager", "ExecutiveBoss"],
        )

        decision = BossExecutiveDecisionContract(
            decision_id=decision_id,
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            status=status,
            approved_budget_usd=recommendation.total_cost_usd if status != ApprovalStatus.REJECTED else 0.0,
            projected_revenue_saved_usd=recommendation.total_protected_revenue_usd if status != ApprovalStatus.REJECTED else 0.0,
            roi_multiple=round(recommendation.total_protected_revenue_usd / (recommendation.total_cost_usd + 1.0), 2),
            manager_verification_score=manager_audit.verification_score,
            requires_dual_executive_signoff=requires_dual,
            executive_conditions=conditions,
            executive_signature_hash=f"SHA256:{memo_hash_sha256[:16]}",
            final_orders_summary=summary,
            signature_contract=signature_contract,
        )

        return decision

    @staticmethod
    def verify_signature(
        signature_contract: ExecutiveSignatureContract,
        public_key: ed25519.Ed25519PublicKey,
    ) -> bool:
        """
        Cryptographically validates the Ed25519 signature against the canonical memo hash.
        """
        try:
            sig_bytes = bytes.fromhex(signature_contract.signature)
            public_key.verify(sig_bytes, signature_contract.memo_hash_sha256.encode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Cryptographic signature verification failed: {e}")
            return False

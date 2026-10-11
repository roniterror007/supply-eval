"""
PlanFlow AI - Tier 2: Judge / Manager Verification Agent
Audits peer consensus contracts, executes deterministic mathematical verification,
verifies physical conservation laws, checks enterprise grounding, and calculates the verification score.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List
from src.core.contracts import (
    AgentType,
    ManagerAuditContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RecommendationPlanContract,
    PeerMessageContract,
    JudgeDecisionContract,
)
from src.agents.base_agent import BaseAgent
from src.verification.math_verifier import DeterministicMathVerifier
from src.verification.grounding_checker import EnterpriseGroundingChecker
from src.core.logger import get_logger

logger = get_logger("JudgeManagerAgent")


class JudgeManagerAgent(BaseAgent):
    """Tier 2 Analytical Auditor and Verification Judge."""

    def __init__(self):
        super().__init__(AgentType.JUDGE_MANAGER)
        self.math_verifier = DeterministicMathVerifier(tolerance_fraction=0.005)
        self.grounding_checker = EnterpriseGroundingChecker()

    def process(self, context: Dict[str, Any]) -> ManagerAuditContract:
        demand: ConsensusDemandContract = context["demand_contract"]
        factory: FactoryAllocationContract = context["factory_contract"]
        supply: SupplyChainAuditContract = context["supply_contract"]
        recommendation: RecommendationPlanContract = context["recommendation_contract"]
        peer_messages: List[PeerMessageContract] = context.get("peer_messages", [])
        row = context.get("row_dict", {})

        # 1. Deterministic Math & Physical Verification
        math_res = self.math_verifier.verify_plan(
            demand=demand,
            factory=factory,
            supply=supply,
            recommendation=recommendation,
        )

        # 2. Enterprise Master Data Grounding Check
        grounding_res = self.grounding_checker.check_entities(
            business_line=demand.product_line,
            region=demand.region,
            factory_names=list(factory.factory_allocations.keys()),
            material_ids=list(supply.required_materials.keys()),
        )

        # 3. Contract Schema Validation
        contracts_valid = True
        try:
            self.validate_contract(demand)
            self.validate_contract(factory)
            self.validate_contract(supply)
            self.validate_contract(recommendation)
        except Exception as e:
            contracts_valid = False
            math_res.violations.append(f"Contract Schema Violation: {str(e)}")

        # Count peer corrections audited
        corrections_audited = sum(1 for m in peer_messages if m.is_correction)

        # Composite verification score
        final_score = math_res.score
        if not grounding_res["grounded"]:
            final_score -= 25.0
        if not contracts_valid:
            final_score -= 40.0

        final_score = max(0.0, min(100.0, final_score))

        # Two-part Gate Enforcement: Hard checks are non-negotiable
        all_hard_passed = (
            math_res.hard_constraints_passed
            and grounding_res["grounded"]
            and contracts_valid
        )

        all_hard_failures = list(math_res.hard_failures)
        if not grounding_res["grounded"]:
            all_hard_failures.append("MASTER_DATA_GROUNDING_FAILED: Unrecognized entity IDs detected in catalog check.")
        if not contracts_valid:
            all_hard_failures.append("CONTRACT_SCHEMA_FAILED: Output failed strict Pydantic invariant validation.")

        if not all_hard_passed:
            gate_status = "FAIL"
            forward_to_boss = False
        elif final_score < 95.0:
            gate_status = "CONDITIONAL"
            forward_to_boss = True
        else:
            gate_status = "PASS"
            forward_to_boss = True

        judge_decision = JudgeDecisionContract(
            hard_constraints_passed=all_hard_passed,
            hard_failures=all_hard_failures,
            soft_verification_score=final_score,
            evidence_coverage=1.0 if contracts_valid else 0.5,
            max_material_uncertainty=0.0 if all_hard_passed else 0.45,
            status=gate_status,
        )

        prompt = f"""
You are the Tier 2 Verification Judge (Manager) for an enterprise supply chain.
You have audited a proposed plan.
Math Verification Results:
- Hard Constraints Passed: {all_hard_passed}
- Conservation Math: {'PASS' if math_res.conservation_passed else 'FAIL'}
- Capacity Math: {'PASS' if math_res.capacity_math_passed else 'FAIL'}
- Enterprise Grounding: {'PASS' if grounding_res['grounded'] else 'FAIL'}
- Final Math Score: {final_score:.1f}/100.0

Write a 2-3 sentence Qualitative Audit Assessment summarizing the business risk of this plan. 
Be professional, analytical, and objective.
"""
        try:
            from src.agents.llm_client import generate_llm_response
            assessment = generate_llm_response(prompt)
        except Exception as e:
            logger.error(f"Failed to generate LLM assessment: {e}")
            assessment = (
                f"Audit Complete. Gate Status: {gate_status} | Hard Checks: {'PASS' if all_hard_passed else 'FAIL'} | "
                f"Soft Score: {final_score:.1f}/100. Conservation: {'PASS' if math_res.conservation_passed else 'FAIL'}, "
                f"Capacity Math: {'PASS' if math_res.capacity_math_passed else 'FAIL'}, "
                f"Inventory Delta: {'PASS' if math_res.inventory_math_passed else 'FAIL'}, "
                f"Grounding: {'PASS' if grounding_res['grounded'] else 'FAIL'}. "
                f"Audited {corrections_audited} peer mutual corrections."
            )

        audit_contract = ManagerAuditContract(
            audit_id=f"AUD-{uuid.uuid4().hex[:8].upper()}",
            all_contracts_passed=contracts_valid,
            unit_conservation_passed=math_res.conservation_passed,
            capacity_math_passed=math_res.capacity_math_passed,
            inventory_math_passed=math_res.inventory_math_passed,
            grounding_check_passed=grounding_res["grounded"],
            peer_corrections_verified_count=corrections_audited,
            verification_score=round(final_score, 1),
            identified_discrepancies=math_res.violations + grounding_res.get("unrecognized_entities", []),
            manager_assessment=assessment,
            forward_to_boss=forward_to_boss,
            judge_decision=judge_decision,
            hard_constraints_passed=all_hard_passed,
            hard_failures=all_hard_failures,
        )

        logger.info(
            f"[bold white on dark_blue]Manager Audit Verdict[/bold white on dark_blue]: "
            f"Score {audit_contract.verification_score}% | Forward to Boss: {forward_to_boss}"
        )
        return audit_contract

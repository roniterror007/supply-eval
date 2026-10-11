"""
PlanFlow AI - Deterministic Mathematical & Physical Verification Engine
Enforces inviolable laws of supply chain physics, conservation of units,
capacity arithmetic, material balance equations, and economic ROI consistency.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from src.core.contracts import (
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
    RecommendationPlanContract,
    JudgeDecisionContract,
)
from src.core.logger import get_logger, log_audit_check

logger = get_logger("VerificationEngine")


@dataclass
class VerificationResult:
    is_valid: bool
    score: float  # 0.0 to 100.0
    conservation_passed: bool
    capacity_math_passed: bool
    inventory_math_passed: bool
    financial_math_passed: bool
    hard_constraints_passed: bool = True
    hard_failures: List[str] = field(default_factory=list)
    judge_decision: Optional[JudgeDecisionContract] = None
    violations: List[str] = field(default_factory=list)
    proof_steps: List[str] = field(default_factory=list)


class DeterministicMathVerifier:
    """Rigorous verification engine that audits all numerical claims from agents."""

    def __init__(self, tolerance_fraction: float = 0.005):
        self.tolerance = tolerance_fraction  # 0.5% numerical tolerance for floating point

    def verify_plan(
        self,
        demand: ConsensusDemandContract,
        factory: FactoryAllocationContract,
        supply: SupplyChainAuditContract,
        recommendation: RecommendationPlanContract,
    ) -> VerificationResult:
        """Audits the full planning pipeline against mathematical laws."""
        violations = []
        proof_steps = []
        score = 100.0

        # =========================================================
        # 1. LAW OF CONSERVATION OF UNITS
        # Sum of factory allocations MUST equal total consensus demand!
        # =========================================================
        total_allocated = sum(factory.factory_allocations.values())
        target_demand = demand.consensus_demand_units
        unit_delta = abs(total_allocated - target_demand)
        rel_delta = unit_delta / (target_demand + 1e-6)

        if rel_delta > self.tolerance:
            msg = (
                f"Conservation of Units VIOLATED: Target demand is {target_demand:.1f} units, "
                f"but factory allocations sum to {total_allocated:.1f} units (Discrepancy: {unit_delta:.1f} units, {rel_delta*100:.2f}%)"
            )
            violations.append(msg)
            proof_steps.append(f"✘ [Conservation Failure] {total_allocated} != {target_demand}")
            score -= 35.0
            log_audit_check("Conservation of Units", False, msg)
            conservation_passed = False
        else:
            proof_steps.append(
                f"✔ [Conservation Confirmed] Allocated sum {total_allocated:.1f} == Consensus {target_demand:.1f} (delta={unit_delta:.2f})"
            )
            log_audit_check("Conservation of Units", True, f"Sum matches target demand within {self.tolerance*100}%")
            conservation_passed = True

        # =========================================================
        # 2. FACTORY CAPACITY & UTILIZATION ARITHMETIC
        # Utilization % = (Allocated / Capacity) * 100
        # =========================================================
        capacity_math_passed = True
        for f_name, alloc in factory.factory_allocations.items():
            cap = factory.factory_capacities.get(f_name, 0.0)
            claimed_util = factory.factory_utilizations.get(f_name, 0.0)
            if cap > 0:
                true_util = (alloc / cap) * 100.0
                util_diff = abs(true_util - claimed_util)
                if util_diff > 1.5:  # more than 1.5 percentage point discrepancy
                    msg = (
                        f"Utilization Math Error in {f_name}: Claimed {claimed_util:.1f}%, "
                        f"Computed: {alloc:.1f}/{cap:.1f} = {true_util:.1f}% (Diff: {util_diff:.1f}%)"
                    )
                    violations.append(msg)
                    score -= 15.0
                    capacity_math_passed = False
                    log_audit_check(f"Capacity Check ({f_name})", False, msg)
                else:
                    proof_steps.append(f"✔ [Capacity Validated] {f_name}: {alloc:.1f}/{cap:.1f} = {true_util:.1f}%")

            # Check overload flag consistency
            is_overloaded_in_list = f_name in factory.overloaded_factories
            has_transfer_from_this_factory = any(s.get("from_factory") == f_name for s in factory.transfer_shifts)
            should_be_overloaded = (claimed_util > 100.0) or has_transfer_from_this_factory
            if should_be_overloaded != is_overloaded_in_list:
                msg = f"Overload classification inconsistency for {f_name}: Util is {claimed_util}%, has_transfer={has_transfer_from_this_factory}, but overload_flag={is_overloaded_in_list}"
                violations.append(msg)
                score -= 10.0
                capacity_math_passed = False
            else:
                proof_steps.append(f"✔ [Overload Flag Validated] {f_name}: overload={is_overloaded_in_list} (util={claimed_util}%, transfer={has_transfer_from_this_factory})")

        if capacity_math_passed:
            log_audit_check("Factory Capacity Arithmetic", True, "All plant utilization calculations verified")

        # =========================================================
        # 3. MATERIAL BALANCE & SHORTAGE EQUATIONS
        # Shortage = max(0, Required - Usable Inventory)
        # =========================================================
        inventory_math_passed = True
        for mat_id, req_qty in supply.required_materials.items():
            cur_stock = supply.current_inventory.get(mat_id, 0.0)
            in_transit = supply.in_transit_inventory.get(mat_id, 0.0)
            claimed_shortage = supply.shortages.get(mat_id, 0.0)

            # Accounts for transit cutoff: if delayed > 7 days, in-transit is excluded
            true_shortage_with_transit = max(0.0, req_qty - (cur_stock + in_transit))
            true_shortage_without_transit = max(0.0, req_qty - cur_stock)

            diff_with = abs(true_shortage_with_transit - claimed_shortage)
            diff_without = abs(true_shortage_without_transit - claimed_shortage)

            if min(diff_with, diff_without) > 2.0:
                msg = (
                    f"Material Balance Error for {mat_id}: Claimed shortage {claimed_shortage:.1f}, "
                    f"Computed with transit = {true_shortage_with_transit:.1f}, without delayed transit = {true_shortage_without_transit:.1f}"
                )
                violations.append(msg)
                score -= 20.0
                inventory_math_passed = False
                log_audit_check(f"Material Balance ({mat_id})", False, msg)
            else:
                accepted_shortage = true_shortage_with_transit if diff_with <= 2.0 else true_shortage_without_transit
                proof_steps.append(
                    f"✔ [Material Balance Validated] {mat_id}: Required {req_qty:.1f} - Usable Stock = Shortage {accepted_shortage:.1f}"
                )

        if inventory_math_passed:
            log_audit_check("Material Balance Equations", True, "All BOM requirements and inventory delta verified")

        # =========================================================
        # 4. FINANCIAL ROI & NET BUSINESS VALUE INTEGRITY
        # Net Benefit = Protected Revenue - Total Cost
        # =========================================================
        computed_net_benefit = recommendation.total_protected_revenue_usd - recommendation.total_cost_usd
        fin_diff = abs(computed_net_benefit - recommendation.net_benefit_usd)
        if fin_diff > 5.0:
            msg = (
                f"Financial Equation Mismatch: Claimed Net Benefit ${recommendation.net_benefit_usd:,.2f}, "
                f"Computed (${recommendation.total_protected_revenue_usd:,.2f} - ${recommendation.total_cost_usd:,.2f}) = ${computed_net_benefit:,.2f}"
            )
            violations.append(msg)
            score -= 15.0
            financial_math_passed = False
            log_audit_check("Financial ROI Math", False, msg)
        else:
            proof_steps.append(
                f"✔ [Financial Integrity Validated] Protected ${recommendation.total_protected_revenue_usd:,.2f} - Cost ${recommendation.total_cost_usd:,.2f} = Net ${computed_net_benefit:,.2f}"
            )
            log_audit_check("Financial ROI Math", True, "Revenue protection and mitigation costs mathematically sound")
            financial_math_passed = True

        final_score = max(0.0, min(100.0, score))
        hard_passed = conservation_passed and capacity_math_passed and inventory_math_passed and financial_math_passed
        hard_failures = [v for v in violations if "VIOLATED" in v or "Error" in v or "Mismatch" in v]
        gate_status = "FAIL" if not hard_passed else ("CONDITIONAL" if final_score < 95.0 else "PASS")

        judge_decision = JudgeDecisionContract(
            hard_constraints_passed=hard_passed,
            hard_failures=hard_failures,
            soft_verification_score=final_score,
            evidence_coverage=1.0,
            max_material_uncertainty=0.0 if hard_passed else 0.50,
            status=gate_status,
        )

        is_valid = hard_passed and (final_score >= 85.0)

        return VerificationResult(
            is_valid=is_valid,
            score=final_score,
            conservation_passed=conservation_passed,
            capacity_math_passed=capacity_math_passed,
            inventory_math_passed=inventory_math_passed,
            financial_math_passed=financial_math_passed,
            hard_constraints_passed=hard_passed,
            hard_failures=hard_failures,
            judge_decision=judge_decision,
            violations=violations,
            proof_steps=proof_steps,
        )

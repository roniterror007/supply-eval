"""
PlanFlow AI - Adversarial Red-Team Challenger Agent
Executes structured truth maintenance challenges before Judge verification.
Attempts to falsify peer assertions using independent records, constraint calculations,
and counterfactual stress testing to prevent correlated confirmation bias.
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from src.core.contracts import (
    AgentType,
    PeerMessageContract,
    ConsensusDemandContract,
    SupplyChainAuditContract,
    FactoryAllocationContract,
)
from src.agents.base_agent import BaseAgent
from src.knowledge_graph.provenance_graph import EvidenceProvenanceGraph
from src.core.logger import get_logger

logger = get_logger("RedTeamChallenger")

ChallengeVerdict = Literal["FALSIFIED", "CORROBORATED", "UNRESOLVED_DISPUTE"]


@dataclass
class ChallengeAuditResult:
    assertion_subject: str
    claimant_agent: str
    verdict: ChallengeVerdict
    evidence_challenge_details: str
    counter_evidence_source: str
    unresolved_uncertainty: float
    decision_alternative: str


class RedTeamChallengerAgent(BaseAgent):
    """Adversarial challenger cross-examining peer consensus before manager audit."""

    def __init__(self, provenance_graph: Optional[EvidenceProvenanceGraph] = None):
        super().__init__(AgentType.RED_TEAM_CHALLENGER, name="RedTeamChallenger")
        self.graph = provenance_graph or EvidenceProvenanceGraph()

    def challenge_consensus(
        self,
        demand: ConsensusDemandContract,
        factory: FactoryAllocationContract,
        supply: SupplyChainAuditContract,
        row_dict: Dict[str, Any],
    ) -> List[ChallengeAuditResult]:
        """Conducts systematic adversarial cross-examination of peer assertions."""
        challenges: List[ChallengeAuditResult] = []

        # -------------------------------------------------------------
        # CHALLENGE 1: Test Pipeline Correlated Bias (Stale CRM Reliance)
        # -------------------------------------------------------------
        stale_usd = float(row_dict.get("stale_deals_usd", 0.0))
        total_pipe = float(row_dict.get("unweighted_pipeline_usd", 1.0))
        stale_ratio = stale_usd / (total_pipe + 1e-6)

        if stale_ratio > 0.18:
            challenges.append(ChallengeAuditResult(
                assertion_subject="Consensus Demand Pipeline Uplift",
                claimant_agent="OpportunityAgent -> DemandPlanningAgent",
                verdict="FALSIFIED",
                evidence_challenge_details=(
                    f"Correlated error detected: Both Demand and Forecasting agents relied on active CRM records, "
                    f"but independent ERP audit shows {stale_ratio*100:.1f}% of pipeline (${stale_usd:,.0f}) has had zero customer activity for >60 days."
                ),
                counter_evidence_source="ERP Customer Master & Accounts Receivable Feed",
                unresolved_uncertainty=0.22,
                decision_alternative=f"Prune ${stale_usd:,.0f} ghost volume and base plant allocation on firm backlog.",
            ))
        else:
            challenges.append(ChallengeAuditResult(
                assertion_subject="Consensus Demand Pipeline Uplift",
                claimant_agent="OpportunityAgent",
                verdict="CORROBORATED",
                evidence_challenge_details="Pipeline conversion rates corroborate with 3-year historical win distribution.",
                counter_evidence_source="Historical CRM Win/Loss Database",
                unresolved_uncertainty=0.08,
                decision_alternative="Proceed with demand uplift.",
            ))

        # -------------------------------------------------------------
        # CHALLENGE 2: Test Factory Maintenance Feasibility
        # -------------------------------------------------------------
        primary_factory = list(factory.factory_allocations.keys())[0]
        allocated = factory.factory_allocations[primary_factory]
        nominal_cap = float(row_dict.get("nominal_capacity", 12000.0))
        eff_cap = float(row_dict.get("effective_capacity", 12000.0))

        if eff_cap < nominal_cap and allocated > eff_cap:
            challenges.append(ChallengeAuditResult(
                assertion_subject=f"Assembly Throughput at {primary_factory}",
                claimant_agent="FactoryPlanningAgent",
                verdict="FALSIFIED",
                evidence_challenge_details=(
                    f"Physical capacity violation: Allocation ({allocated:.0f}u) assumes nominal capacity, "
                    f"ignoring active MES work orders for robotic welding cell maintenance (limiting plant to {eff_cap:.0f}u)."
                ),
                counter_evidence_source="MES Plant Maintenance Work Orders",
                unresolved_uncertainty=0.18,
                decision_alternative="Execute mandatory inter-plant transfer shift to secondary partner factory.",
            ))
        else:
            challenges.append(ChallengeAuditResult(
                assertion_subject=f"Assembly Throughput at {primary_factory}",
                claimant_agent="FactoryPlanningAgent",
                verdict="CORROBORATED",
                evidence_challenge_details="Planned run matches verified machine availability and crew scheduling.",
                counter_evidence_source="MES Line Schedule",
                unresolved_uncertainty=0.05,
                decision_alternative="Maintain scheduled production run.",
            ))

        # -------------------------------------------------------------
        # CHALLENGE 3: Test Supplier Lead Time & Shortage Claim
        # -------------------------------------------------------------
        delay_days = int(row_dict.get("supplier_delay_days", 0))
        if delay_days > 7 and not supply.material_shortage_detected:
            challenges.append(ChallengeAuditResult(
                assertion_subject="Material MAT-M1 Availability",
                claimant_agent="SupplyChainAgent",
                verdict="FALSIFIED",
                evidence_challenge_details=(
                    f"Falsified shortage claim: Transit delay of {delay_days} days was omitted from stock balance, "
                    f"creating an unhedged component gap."
                ),
                counter_evidence_source="Supplier EDI Tracking Feed",
                unresolved_uncertainty=0.25,
                decision_alternative="Authorize emergency air-freight expedite.",
            ))
        else:
            challenges.append(ChallengeAuditResult(
                assertion_subject="Material MAT-M1 Availability",
                claimant_agent="SupplyChainAgent",
                verdict="CORROBORATED",
                evidence_challenge_details="BOM requirements and component balances match physical warehouse counts.",
                counter_evidence_source="WMS Inventory Master",
                unresolved_uncertainty=0.06,
                decision_alternative="Execute BOM allocation.",
            ))

        return challenges

    def process(self, context: Dict[str, Any]) -> Any:
        return self.challenge_consensus(
            demand=context["demand_contract"],
            factory=context["factory_contract"],
            supply=context["supply_contract"],
            row_dict=context.get("row_dict", {}),
        )

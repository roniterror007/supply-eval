"""
PlanFlow AI - Verification-Aware Typed Contracts Subsystem
Defines strict Pydantic V2 schemas, invariants, and validation rules for all agent interactions,
deterministic mathematical consistency checks, peer fact exchange, and executive governance.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentType(str, Enum):
    FORECASTING = "ForecastingAgent"
    OPPORTUNITY = "OpportunityAgent"
    DEMAND_PLANNING = "DemandPlanningAgent"
    FACTORY_PLANNING = "FactoryPlanningAgent"
    PRODUCTION = "ProductionAgent"
    SUPPLY_CHAIN = "SupplyChainAgent"
    SCENARIO = "ScenarioAgent"
    RECOMMENDATION = "RecommendationAgent"
    JUDGE_MANAGER = "JudgeManagerAgent"
    EXECUTIVE_BOSS = "ExecutiveBossAgent"
    RED_TEAM_CHALLENGER = "RedTeamChallengerAgent"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    CONDITIONAL_APPROVAL = "CONDITIONAL_APPROVAL"
    REJECTED = "REJECTED"


class PeerMessageContract(BaseModel):
    """Contract for friendly collaborative peer communication and mutual correction."""
    sender: AgentType
    recipient: AgentType
    tone: str = Field(default="friendly_collaborative", description="Warm, collegiate professional tone")
    greeting: str
    facts_asserted: Dict[str, Any]
    critique_or_correction: Optional[str] = None
    is_correction: bool = False
    revised_metric_name: Optional[str] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)


class AgentClaim(BaseModel):
    """A structured, typed factual claim made by an agent on the blackboard."""
    claim_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    topic: str
    statement: str
    value: Optional[float | str | bool] = None
    unit: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    evidence_ids: List[UUID] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    status: Literal["PROPOSED", "SUPPORTED", "CHALLENGED", "REVISED", "ACCEPTED", "REJECTED"] = "PROPOSED"
    supersedes_claim_id: Optional[UUID] = None


class AgentMessage(BaseModel):
    """Structured message protocol for agent interactions on the blackboard."""
    message_id: UUID = Field(default_factory=uuid4)
    sender: str
    recipients: List[str]
    message_type: Literal["ASSERT", "ASK", "CHALLENGE", "CORRECTION", "PROPOSAL", "ACCEPT", "REJECT", "ESCALATE"]
    claim_ids: List[UUID] = Field(default_factory=list)
    evidence_ids: List[UUID] = Field(default_factory=list)
    requested_action: Optional[str] = None
    deadline: Optional[datetime] = None



class ForecastContract(BaseModel):
    """Contract emitted by Forecasting Agent."""
    product_line: str
    region: str
    horizon_month: str
    point_forecast_units: float = Field(ge=0.0)
    p10_units: float = Field(ge=0.0)
    p50_units: float = Field(ge=0.0)
    p90_units: float = Field(ge=0.0)
    baseline_revenue_usd: float = Field(ge=0.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    primary_drivers: List[str]
    model_type: str = "PyTorch_Neural_Surrogate_CUDA"
    # HELICASE Evidence & Uncertainty Extensions
    evidence_ids: List[str] = Field(default_factory=list)
    source_timestamp: str = Field(default="2026-09-11T12:00:00Z")
    lineage: List[str] = Field(default_factory=list)
    confidence_interval: List[float] = Field(default_factory=list)
    uncertainty_contributors: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_quantiles(self) -> "ForecastContract":
        if not (self.p10_units <= self.p50_units <= self.p90_units):
            raise ValueError(
                f"Quantile monotonicity violated: p10={self.p10_units}, "
                f"p50={self.p50_units}, p90={self.p90_units}"
            )
        return self


class OpportunityContract(BaseModel):
    """Contract emitted by Opportunity Intelligence Agent."""
    region: str
    product_line: str
    total_pipeline_usd: float = Field(ge=0.0)
    stale_deals_usd: float = Field(ge=0.0)
    high_value_deals_usd: float = Field(ge=0.0)
    at_risk_deals_count: int = Field(ge=0)
    net_probability_weighted_units: float = Field(ge=0.0)
    flagged_deal_ids: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class ConsensusDemandContract(BaseModel):
    """Contract emitted by Demand Planning Agent after peer consensus."""
    region: str
    product_line: str
    horizon_month: str
    ml_forecast_units: float = Field(ge=0.0)
    pipeline_uplift_units: float
    backlog_units: float = Field(ge=0.0)
    scenario_adjustment_units: float
    consensus_demand_units: float = Field(ge=0.0)
    consensus_revenue_usd: float = Field(ge=0.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    peer_revisions_count: int = Field(ge=0)
    consensus_rationale: str
    # HELICASE Extensions
    evidence_ids: List[str] = Field(default_factory=list)
    source_timestamp: str = Field(default="2026-09-11T12:00:00Z")
    lineage: List[str] = Field(default_factory=list)
    confidence_interval: List[float] = Field(default_factory=list)
    uncertainty_contributors: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_math_consensus(self) -> "ConsensusDemandContract":
        computed = max(
            self.backlog_units,
            self.ml_forecast_units + self.pipeline_uplift_units + self.scenario_adjustment_units,
        )
        # Check conservation within small numerical tolerance
        diff = abs(computed - self.consensus_demand_units)
        if diff > 2.0 and abs(self.consensus_demand_units) > 0:
            pct_diff = diff / self.consensus_demand_units
            if pct_diff > 0.05:  # more than 5% discrepancy
                raise ValueError(
                    f"Consensus demand math invariant mismatch: {self.consensus_demand_units} vs computed {computed}"
                )
        return self


class FactoryAllocationContract(BaseModel):
    """Contract emitted by Factory Planning Agent."""
    total_demand_units: float = Field(ge=0.0)
    factory_allocations: Dict[str, float]
    factory_capacities: Dict[str, float]
    factory_utilizations: Dict[str, float]
    overloaded_factories: List[str]
    transfer_shifts: List[Dict[str, Any]]
    confidence_score: float = Field(ge=0.0, le=1.0)
    # HELICASE Extensions
    evidence_ids: List[str] = Field(default_factory=list)
    source_timestamp: str = Field(default="2026-09-11T12:00:00Z")
    lineage: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_allocation_conservation(self) -> "FactoryAllocationContract":
        total_allocated = sum(self.factory_allocations.values())
        diff = abs(total_allocated - self.total_demand_units)
        if diff > 1.0 and self.total_demand_units > 0:
            if diff / self.total_demand_units > 0.01:
                raise ValueError(
                    f"Factory allocation violates conservation of units! "
                    f"Allocated: {total_allocated}, Target: {self.total_demand_units}"
                )
        return self


class ProductionPlanContract(BaseModel):
    """Contract emitted by Production Agent."""
    factory_name: str
    scheduled_units: float = Field(ge=0.0)
    effective_utilization_pct: float = Field(ge=0.0)
    bottleneck_detected: bool
    bottleneck_reason: Optional[str] = None
    overtime_hours_required: float = Field(ge=0.0)
    prebuild_units_recommended: float = Field(ge=0.0)
    production_cost_usd: float = Field(ge=0.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    # HELICASE Extensions
    evidence_ids: List[str] = Field(default_factory=list)
    source_timestamp: str = Field(default="2026-09-11T12:00:00Z")


class SupplyChainAuditContract(BaseModel):
    """Contract emitted by Supply Chain Agent."""
    bill_of_materials: Dict[str, float]  # Material ID -> required qty per finished good
    required_materials: Dict[str, float]
    current_inventory: Dict[str, float]
    in_transit_inventory: Dict[str, float]
    shortages: Dict[str, float]
    material_shortage_detected: bool
    critical_materials_at_risk: List[str]
    delivery_risk: RiskLevel
    total_expedite_cost_usd: float = Field(ge=0.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    # HELICASE Extensions
    evidence_ids: List[str] = Field(default_factory=list)
    source_timestamp: str = Field(default="2026-09-11T12:00:00Z")
    lineage: List[str] = Field(default_factory=list)
    confidence_interval: List[float] = Field(default_factory=list)
    uncertainty_contributors: List[str] = Field(default_factory=list)


class ScenarioSimulationContract(BaseModel):
    """Contract emitted by Scenario Twin Agent."""
    scenario_id: str
    scenario_description: str
    demand_multiplier: float = Field(gt=0.0)
    supplier_delay_days: int = Field(ge=0)
    capacity_multiplier: float = Field(gt=0.0)
    baseline_demand_units: float
    simulated_demand_units: float
    simulated_max_factory_utilization: float
    simulated_shortage_units: float
    simulated_delivery_risk: RiskLevel


class ActionRecommendation(BaseModel):
    """A prioritized, actionable business recommendation."""
    priority_rank: int = Field(ge=1)
    action_type: str  # e.g., "SHIFT_FACTORY", "EXPEDITE_SUPPLIER", "PULL_PRODUCTION", "SALES_CONFIRMATION"
    description: str
    target_entity: str  # e.g., "Factory B (Detroit)", "MAT-M1"
    cost_usd: float = Field(ge=0.0)
    protected_revenue_usd: float = Field(ge=0.0)
    risk_mitigation_score: float = Field(ge=0.0, le=1.0)

    @property
    def net_roi_multiplier(self) -> float:
        if self.cost_usd == 0:
            return round(self.protected_revenue_usd / 1.0, 2)
        return round(self.protected_revenue_usd / self.cost_usd, 2)


class RecommendationPlanContract(BaseModel):
    """Contract emitted by Recommendation Agent."""
    recommendations: List[ActionRecommendation]
    total_cost_usd: float = Field(ge=0.0)
    total_protected_revenue_usd: float = Field(ge=0.0)
    net_benefit_usd: float
    post_recommendation_delivery_risk: RiskLevel
    expected_capacity_normalized_pct: float
    confidence_score: float = Field(ge=0.0, le=1.0)


class EvidenceReference(BaseModel):
    """Traceable, auditable reference to ground evidence in the enterprise data lake."""
    evidence_id: UUID = Field(default_factory=uuid4)
    source_system: Literal["ERP", "CRM", "MES", "WMS", "supplier_portal", "manual"]
    source_record_id: str
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    freshness_hours: float = 0.0
    source_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    conflict_status: Literal["NONE", "OPEN", "RESOLVED", "ACCEPTED_BY_APPROVER"] = "NONE"


class JudgeDecisionContract(BaseModel):
    """Two-part verification gate separating hard physical laws from soft quality scoring."""
    hard_constraints_passed: bool
    hard_failures: List[str] = Field(default_factory=list)
    soft_verification_score: float = Field(ge=0.0, le=100.0)  # 0-100
    evidence_coverage: float = Field(ge=0.0, le=1.0, default=1.0)        # 0-1
    max_material_uncertainty: float = Field(ge=0.0, le=1.0, default=0.0)
    status: Literal["PASS", "CONDITIONAL", "FAIL"]


class OptimizationCertificateContract(BaseModel):
    """Solver-certified audit certificate for HiGHS MILP master planner solutions."""
    solve_status: Literal["OPTIMAL", "FEASIBLE_GAP", "INFEASIBLE", "TIME_LIMIT", "ERROR"]
    objective_value: Optional[float] = None
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_seconds: float
    candidate_space_version: str = "v2.0"
    scenario_set_version: str = "v2.0-stochastic"
    constraint_set_hash: str


class ExecutiveSignatureContract(BaseModel):
    """Cryptographically signed authorization payload using enterprise private key."""
    memo_hash_sha256: str
    signature_algorithm: Literal["Ed25519", "RSA-PSS"] = "Ed25519"
    signature: str
    signer_id: str
    signing_key_id: str
    signed_at: datetime = Field(default_factory=datetime.utcnow)
    policy_version: str = "v2.0-enterprise"
    approval_chain: List[str] = Field(default_factory=list)


class AutonomyPolicyContract(BaseModel):
    """Governs dynamic execution autonomy per supply chain operational category."""
    decision_type: str = Field(default="inventory_rebalance")
    execution_mode: Literal["RECOMMEND", "CONDITIONAL_EXECUTE", "AUTO_EXECUTE"] = "RECOMMEND"
    max_financial_exposure_usd: float = Field(ge=0.0, default=100000.0)
    max_customer_sla_risk: float = Field(ge=0.0, le=1.0, default=0.05)
    max_uncertainty: float = Field(ge=0.0, le=1.0, default=0.20)
    min_evidence_freshness_score: float = Field(ge=0.0, le=1.0, default=0.80)
    requires_dual_approval: bool = False
    reversible_within_hours: Optional[int] = None
    required_approver_roles: List[str] = Field(default_factory=lambda: ["VP_Supply_Chain"])


class ManagerAuditContract(BaseModel):
    """Contract emitted by Judge / Manager Agent after auditing peer consensus & math."""
    audit_id: str
    all_contracts_passed: bool
    unit_conservation_passed: bool
    capacity_math_passed: bool
    inventory_math_passed: bool
    grounding_check_passed: bool
    peer_corrections_verified_count: int = Field(ge=0)
    verification_score: float = Field(ge=0.0, le=100.0)
    identified_discrepancies: List[str]
    manager_assessment: str
    forward_to_boss: bool
    judge_decision: Optional[JudgeDecisionContract] = None
    hard_constraints_passed: bool = True
    hard_failures: List[str] = Field(default_factory=list)


class BossExecutiveDecisionContract(BaseModel):
    """Contract emitted by Executive Boss Agent for final governance sign-off."""
    decision_id: str
    timestamp: str
    status: ApprovalStatus
    approved_budget_usd: float = Field(ge=0.0)
    projected_revenue_saved_usd: float = Field(ge=0.0)
    roi_multiple: float
    manager_verification_score: float
    requires_dual_executive_signoff: bool
    executive_conditions: List[str]
    executive_signature_hash: str
    final_orders_summary: str
    signature_contract: Optional[ExecutiveSignatureContract] = None


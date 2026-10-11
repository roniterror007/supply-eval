"""
PlanFlow AI - End-to-End Multi-Agent Orchestrator
Coordinates the 3-Tier Multi-Agent Planning Hierarchy:
Tier 1: Specialist Peers & Friendly Dialogue
Tier 2: Judge / Manager Verification Audit
Tier 3: Executive Boss Sign-Off
"""

from typing import Dict, Any, Optional
from src.core.contracts import (
    ForecastContract,
    OpportunityContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    ProductionPlanContract,
    SupplyChainAuditContract,
    ScenarioSimulationContract,
    RecommendationPlanContract,
    ManagerAuditContract,
    BossExecutiveDecisionContract,
)
from src.agents.peer_agents import (
    ForecastingAgent,
    OpportunityAgent,
    DemandPlanningAgent,
    FactoryPlanningAgent,
    ProductionAgent,
    SupplyChainAgent,
    ScenarioAgent,
    RecommendationAgent,
)
from src.agents.peer_dialogue import PeerDialogueEngine
from src.agents.judge_manager import JudgeManagerAgent
from src.agents.executive_boss import ExecutiveBossAgent
from src.agents.red_team_challenger import RedTeamChallengerAgent
from src.models.neural_surrogate import ModelPredictor
from src.digital_twin.simulator import DigitalTwinSimulator
from src.optimization.master_planner import MasterMILPPlanner, OptimizationResult
from src.optimization.feasibility_oracle import FeasibilityOracle
from src.knowledge_graph.provenance_graph import EvidenceProvenanceGraph
from src.knowledge_graph.models import EvidenceNode, EvidenceEdge
from src.governance.autonomy_policy import RiskAdaptiveAutonomyEngine
from src.core.blackboard import SharedBlackboard
from src.agents.router_agent import ChiefOfStaffRouter
from src.core.logger import get_logger

logger = get_logger("Orchestrator")


class PlanFlowOrchestrator:
    """Master orchestrator executing the Simulation-Optimization, Evidence Graph & Robust Governance Planning Cycle."""

    def __init__(self, model_checkpoint: str = "checkpoints/best_surrogate_model.pt"):
        self.predictor = ModelPredictor(model_checkpoint)
        self.forecasting_agent = ForecastingAgent(self.predictor)
        self.opportunity_agent = OpportunityAgent()
        self.demand_agent = DemandPlanningAgent()

        # Factory rules
        capacities = {
            "Factory A (Stuttgart)": 12000.0,
            "Factory B (Detroit)": 15000.0,
            "Factory C (Yokohama)": 10000.0,
            "Factory D (Wroclaw)": 14000.0,
            "Factory E (Monterrey)": 16000.0,
        }
        self.factory_agent = FactoryPlanningAgent()
        self.production_agent = ProductionAgent()
        self.supply_agent = SupplyChainAgent()

        self.digital_twin = DigitalTwinSimulator(
            factory_capacities=capacities,
            bom_ratios={"MAT-A1": 1.0, "MAT-B2": 1.0, "MAT-C3": 2.0},
        )
        self.scenario_agent = ScenarioAgent(self.digital_twin)
        self.recommendation_agent = RecommendationAgent()

        # Peer Dialogue & Red-Team Challenger
        self.peer_dialogue = PeerDialogueEngine()
        self.red_team = RedTeamChallengerAgent()

        # Operations Research Master Optimizer & Feasibility Oracle
        self.master_planner = MasterMILPPlanner()
        self.feasibility_oracle = FeasibilityOracle()

        # Governance & Autonomy
        self.autonomy_engine = RiskAdaptiveAutonomyEngine()

        # Tier 2 & Tier 3 Oversight Agents
        self.judge_manager = JudgeManagerAgent()
        self.executive_boss = ExecutiveBossAgent()
        
        self.router = ChiefOfStaffRouter()

    def run_planning_cycle(
        self,
        row_dict: Dict[str, Any],
        scenario_name: str = "Quarterly Demand Review",
        demand_multiplier: float = 1.0,
        supplier_delay_days: int = 0,
        capacity_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Runs the complete end-to-end planning cycle across all three tiers + OR optimizer."""
        # -------------------------------------------------------------
        # 0. INJECT SCENARIO PARAMETERS INTO ROW
        # -------------------------------------------------------------
        # Copy the row dict so we don't mutate the global dataframe
        row_dict = row_dict.copy()
        # The agents read these specific keys (surge_pct, supplier_delay_days, effective_capacity)
        row_dict["surge_pct"] = (demand_multiplier - 1.0)
        row_dict["supplier_delay_days"] = supplier_delay_days
        base_capacity = float(row_dict.get("effective_capacity", 12000.0))
        region = row_dict.get("region", "Global")
        product_line = row_dict.get("product_line", "Vehicle Model X")
        logger.info(f"Starting cycle for region {region} | Product {product_line}")
        logger.info(f"Custom Query: {scenario_name}")

        llm_params = {}
        routing_plan = self.router.route_scenario(scenario_name)
        
        if routing_plan:
            global_params = routing_plan.get("global_parameters", {})
            demand_multiplier = global_params.get("demand_multiplier", demand_multiplier)
            supplier_delay_days = global_params.get("supplier_delay_days", supplier_delay_days)
            capacity_multiplier = global_params.get("capacity_multiplier", capacity_multiplier)
            llm_params = global_params
            
        row_dict["effective_capacity"] = base_capacity * capacity_multiplier

        context: Dict[str, Any] = {
            "row_dict": row_dict,
            "scenario_name": scenario_name,
            "demand_multiplier": demand_multiplier,
            "supplier_delay_days": supplier_delay_days,
            "capacity_multiplier": capacity_multiplier,
            "blackboard": SharedBlackboard(),
            "selected_agents": routing_plan.get("selected_agents", []),
            "agent_directives": routing_plan.get("agent_directives", {})
        }
        
        # Log router decisions to blackboard
        if routing_plan.get("selected_agents"):
            from src.core.contracts import AgentClaim
            router_claim = AgentClaim(
                agent_id="CHIEF_OF_STAFF_ROUTER",
                topic="intelligent_routing",
                statement=f"Routed directives to: {', '.join(routing_plan['selected_agents'])}",
                value=len(routing_plan['selected_agents']),
                unit="agents",
                assumptions=[f"{k}: {v}" for k,v in routing_plan.get("agent_directives", {}).items()],
                confidence=0.99,
                status="SUPPORTED"
            )
            context["blackboard"].add_claim(router_claim)

        # -------------------------------------------------------------
        # 1. BUILD EVIDENCE KNOWLEDGE GRAPH (HELICASE)
        # -------------------------------------------------------------
        graph = EvidenceProvenanceGraph()
        # ERP Order Backlog Node
        ev_backlog = EvidenceNode(
            entity_type="order",
            entity_id=str(row_dict.get("scenario_id", "SCN-001")),
            value=float(row_dict.get("backlog_units", 3000.0)),
            source_system="ERP",
            source_record_id=f"ERP-SO-{row_dict.get('scenario_id', '001')}",
            confidence=0.98,
        )
        # CRM Pipeline Node
        ev_crm = EvidenceNode(
            entity_type="forecast",
            entity_id=str(row_dict.get("scenario_id", "SCN-001")),
            value=float(row_dict.get("unweighted_pipeline_usd", 5000000.0)),
            source_system="CRM",
            source_record_id=f"SF-OPP-{row_dict.get('scenario_id', '001')}",
            confidence=0.88,
            quality_flags=["STALE"] if float(row_dict.get("stale_deals_usd", 0.0)) > 500_000 else [],
        )
        # WMS Inventory Node
        ev_wms = EvidenceNode(
            entity_type="sku",
            entity_id="MAT-M1",
            value=float(row_dict.get("mat1_initial_stock", 10000.0)),
            source_system="WMS",
            source_record_id="WMS-STOCK-MAT-M1",
            confidence=0.95,
        )
        graph.add_node(ev_backlog)
        graph.add_node(ev_crm)
        graph.add_node(ev_wms)

        # -------------------------------------------------------------
        # 2. TIER 1: SPECIALIST PEERS - INITIAL RUN
        # -------------------------------------------------------------
        forecast_contract = self.forecasting_agent.process(context)
        forecast_contract.evidence_ids = [str(ev_backlog.evidence_id)]
        context["forecast_contract"] = forecast_contract

        opportunity_contract = self.opportunity_agent.process(context)
        context["opportunity_contract"] = opportunity_contract

        demand_contract = self.demand_agent.process(context)
        demand_contract.evidence_ids = [str(ev_backlog.evidence_id), str(ev_crm.evidence_id)]
        context["demand_contract"] = demand_contract

        factory_contract = self.factory_agent.process(context)
        context["factory_contract"] = factory_contract

        production_contract = self.production_agent.process(context)
        context["production_contract"] = production_contract

        supply_contract = self.supply_agent.process(context)
        supply_contract.evidence_ids = [str(ev_wms.evidence_id)]
        context["supply_contract"] = supply_contract

        # -------------------------------------------------------------
        # 3. TIER 1: FRIENDLY PEER DIALOGUE & MUTUAL CORRECTION
        # -------------------------------------------------------------
        revised_demand, revised_factory, revised_supply, dialogue_messages = (
            self.peer_dialogue.conduct_collaborative_review(
                forecast=forecast_contract,
                opportunity=opportunity_contract,
                demand=demand_contract,
                factory=factory_contract,
                supply=supply_contract,
                row_dict=row_dict,
            )
        )
        context["demand_contract"] = revised_demand
        context["factory_contract"] = revised_factory
        context["supply_contract"] = revised_supply
        context["peer_messages"] = dialogue_messages

        # -------------------------------------------------------------
        # 4. RED-TEAM ADVERSARIAL CHALLENGER AUDIT
        # -------------------------------------------------------------
        red_team_results = self.red_team.challenge_consensus(
            demand=revised_demand,
            factory=revised_factory,
            supply=revised_supply,
            row_dict=row_dict,
        )
        context["red_team_challenges"] = red_team_results

        # -------------------------------------------------------------
        # 5. OPERATIONS RESEARCH MASTER OPTIMIZER (MILP SOLVER)
        # SPORD: Agents diagnose and constrain; Solver produces optimal plan!
        # -------------------------------------------------------------
        prim_factory = list(revised_factory.factory_allocations.keys())[0]
        sec_factory = [k for k in revised_factory.factory_allocations.keys() if k != prim_factory]
        sec_f_name = sec_factory[0] if sec_factory else "Factory D (Wroclaw)"

        prim_cap = revised_factory.factory_capacities.get(prim_factory, 12000.0)
        sec_cap = revised_factory.factory_capacities.get(sec_f_name, 14000.0)
        
        # Multi-Factory Routing Constraint
        import yaml
        rules = yaml.safe_load(open("configs/business_rules.yaml"))
        b_line = row_dict.get("business_line", "Analytical")
        compatible_factories = next((b.get("compatible_factories", []) for b in rules.get("business_lines", []) if isinstance(b, dict) and b.get("name") == b_line), [])
        
        if compatible_factories and sec_f_name not in compatible_factories:
            sec_cap = 0.0

        target_demand_u = revised_demand.consensus_demand_units
        unit_price = float(row_dict.get("unit_price_usd", 1200.0))
        
        bom_ratios = {
            "MAT-A1": float(row_dict.get("mat_A1_req", 1.0)),
            "MAT-B2": float(row_dict.get("mat_B2_req", 1.0)),
            "MAT-C3": float(row_dict.get("mat_C3_req", 1.0)),
        }
        
        usable_stocks = {}
        for mat in bom_ratios.keys():
            stk = float(row_dict.get(f"{mat.replace('-', '_')}_initial_stock", 5000.0))
            if int(row_dict.get(f"{mat.replace('-', '_')}_delay_days", 0)) <= 7:
                stk += float(row_dict.get(f"{mat.replace('-', '_')}_in_transit", 0.0))
            usable_stocks[mat] = stk

        milp_plan = self.master_planner.solve(
            target_demand=target_demand_u,
            unit_price=unit_price,
            primary_factory=prim_factory,
            primary_capacity=prim_cap,
            secondary_factory=sec_f_name,
            secondary_capacity=sec_cap,
            bom_ratios=bom_ratios,
            usable_stocks=usable_stocks,
            max_overtime_hours=120.0,
        )

        feasibility_check = self.feasibility_oracle.verify_plan(
            plan=milp_plan,
            target_demand=target_demand_u,
            factory_capacities={prim_factory: prim_cap, sec_f_name: sec_cap},
            usable_stocks=usable_stocks,
            bom_ratios=bom_ratios,
        )
        context["milp_plan"] = milp_plan
        context["feasibility_check"] = feasibility_check

        # Run Scenario Agent (Digital Twin)
        scenario_contract = self.scenario_agent.process(context)
        context["scenario_contract"] = scenario_contract

        # Run Recommendation Agent
        recommendation_contract = self.recommendation_agent.process(context)
        context["recommendation_contract"] = recommendation_contract

        # -------------------------------------------------------------
        # 6. RISK-ADAPTIVE AUTONOMY POLICY EVALUATION
        # -------------------------------------------------------------
        has_stale = any(f.verdict == "FALSIFIED" for f in red_team_results)
        autonomy_decision = self.autonomy_engine.evaluate_action(
            action_id="PLAN-AUTO-EXEC",
            decision_class="expedite" if revised_supply.material_shortage_detected else "production_reschedule",
            cost_usd=milp_plan.total_cost_usd,
            uncertainty=1.0 - revised_demand.confidence_score,
            evidence_quality=0.60 if has_stale else 0.92,
            customer_service_risk=0.25 if revised_supply.material_shortage_detected else 0.05,
            has_stale_or_conflicted_evidence=has_stale,
        )

        # -------------------------------------------------------------
        # 7. TIER 2: JUDGE / MANAGER VERIFICATION AUDIT
        # -------------------------------------------------------------
        manager_audit_contract = self.judge_manager.process(context)
        context["manager_audit_contract"] = manager_audit_contract

        # -------------------------------------------------------------
        # 8. TIER 3: EXECUTIVE BOSS SIGN-OFF
        # -------------------------------------------------------------
        boss_decision_contract = self.executive_boss.process(context)
        context["boss_decision_contract"] = boss_decision_contract

        return {
            "forecast": forecast_contract,
            "opportunity": opportunity_contract,
            "demand": revised_demand,
            "factory": revised_factory,
            "production": production_contract,
            "supply": revised_supply,
            "dialogue_messages": dialogue_messages,
            "scenario": scenario_contract,
            "recommendation": recommendation_contract,
            "manager_audit": manager_audit_contract,
            "boss_decision": boss_decision_contract,
            # Enhanced PlanFlow 2.0 Outputs
            "evidence_graph": graph,
            "red_team_challenges": red_team_results,
            "milp_optimal_plan": milp_plan,
            "feasibility_report": feasibility_check,
            "autonomy_decision": autonomy_decision,
            "blackboard": context.get("blackboard"),
            "llm_params": llm_params,
        }

"""
PlanFlow AI - Specialized Peer Agents
Implementations of:
1. ForecastingAgent (PyTorch CUDA neural forecast + intervals)
2. OpportunityAgent (Salesforce CRM pipeline analysis)
3. DemandPlanningAgent (Consensus demand synthesis)
4. FactoryPlanningAgent (Factory allocation & capacity load balancing)
5. ProductionAgent (Line scheduling & bottlenecks)
6. SupplyChainAgent (BOM explosion & material shortages)
7. ScenarioAgent (Digital Twin what-if propagation)
8. RecommendationAgent (Ranked trade-off actions & business ROI)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from src.core.contracts import (
    AgentType,
    RiskLevel,
    ForecastContract,
    OpportunityContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    ProductionPlanContract,
    SupplyChainAuditContract,
    ScenarioSimulationContract,
    ActionRecommendation,
    RecommendationPlanContract,
)
from src.agents.base_agent import BaseAgent
from src.models.neural_surrogate import ModelPredictor
from src.digital_twin.simulator import DigitalTwinSimulator
from src.core.logger import get_logger
from src.core.contracts import AgentClaim, AgentMessage
from src.core.cost_engine import get_cost_engine
from uuid import uuid4

logger = get_logger("PeerAgents")


class ForecastingAgent(BaseAgent):
    """Generates baseline ML forecasts, quantiles (P10, P50, P90), and confidence on GPU."""

    def __init__(self, predictor: Optional[ModelPredictor] = None):
        super().__init__(AgentType.FORECASTING)
        self.predictor = predictor or ModelPredictor("checkpoints/best_surrogate_model.pt")

    def process(self, context: Dict[str, Any]) -> ForecastContract:
        row = context.get("row_dict", {})
        preds = self.predictor.predict_single(row)

        p50 = float(preds["p50"])
        p10 = float(preds["p10"])
        p90 = float(preds["p90"])
        unit_price = float(row.get("unit_price_usd", 1200.0))

        confidence = float(np.clip(1.0 - ((p90 - p10) / (2.0 * p50 + 1e-5)), 0.65, 0.98))

        drivers = [
            f"Macro Index ({row.get('market_index', 100):.1f})",
            f"Recent Order Momentum (M1={row.get('hist_orders_m1', 0):.0f})",
            f"Firm Backlog Baseline ({row.get('backlog_units', 0):.0f} units)",
        ]

        contract = ForecastContract(
            product_line=str(row.get("business_line", "Automotive Solutions")),
            region=str(row.get("region", "Germany")),
            horizon_month=str(row.get("timestamp_month", "2026-11")),
            point_forecast_units=round(p50, 1),
            p10_units=round(p10, 1),
            p50_units=round(p50, 1),
            p90_units=round(p90, 1),
            baseline_revenue_usd=round(p50 * unit_price, 2),
            confidence_score=round(confidence, 3),
            primary_drivers=drivers,
            model_type="PyTorch_Neural_Surrogate_CUDA",
        )
        
        # Publish Claim to Blackboard
        blackboard = context.get("blackboard")
        reasoning = context.get("reasoning_forecast", "")
        if blackboard:
            claim = AgentClaim(
                agent_id=self.agent_type.value,
                topic="baseline_demand_forecast",
                statement=f"Forecasted {contract.region} demand is ${contract.baseline_revenue_usd:,.0f}. {reasoning}",
                value=contract.baseline_revenue_usd,
                unit="USD",
                assumptions=drivers,
                confidence=confidence,
                status="PROPOSED"
            )
            blackboard.add_claim(claim)
            
        return self.validate_contract(contract)


class OpportunityAgent(BaseAgent):
    """Scans Salesforce CRM deals, isolates stale/at-risk opportunities, and calculates net pipeline."""

    def __init__(self):
        super().__init__(AgentType.OPPORTUNITY)

    def process(self, context: Dict[str, Any]) -> OpportunityContract:
        row = context.get("row_dict", {})
        total_pipe = float(row.get("unweighted_pipeline_usd", 5_000_000.0))
        stale_pipe = float(row.get("stale_deals_usd", 800_000.0))
        active_pipe = float(row.get("active_pipeline_usd", total_pipe - stale_pipe))
        unit_price = float(row.get("unit_price_usd", 1200.0))

        # Expected conversion rate (35% on active, 5% on stale)
        active_conv = 0.35
        stale_conv = 0.05
        net_usd = (active_pipe * active_conv) + (stale_pipe * stale_conv)
        net_units = round(net_usd / (unit_price + 1e-5), 1)

        at_risk_count = int(np.ceil(stale_pipe / 350_000.0))
        flagged_ids = [f"SF-OPP-{1000 + i}" for i in range(at_risk_count)]

        contract = OpportunityContract(
            region=str(row.get("region", "Germany")),
            product_line=str(row.get("business_line", "Automotive Solutions")),
            total_pipeline_usd=round(total_pipe, 2),
            stale_deals_usd=round(stale_pipe, 2),
            high_value_deals_usd=round(total_pipe * 0.45, 2),
            at_risk_deals_count=at_risk_count,
            net_probability_weighted_units=net_units,
            flagged_deal_ids=flagged_ids,
            confidence_score=0.88,
        )
        return self.validate_contract(contract)


class DemandPlanningAgent(BaseAgent):
    """Synthesizes ML baseline, CRM uplift, backlog, and scenario adjustments into consensus demand."""

    def __init__(self):
        super().__init__(AgentType.DEMAND_PLANNING)

    def process(self, context: Dict[str, Any]) -> ConsensusDemandContract:
        forecast: ForecastContract = context["forecast_contract"]
        opp: OpportunityContract = context["opportunity_contract"]
        row = context.get("row_dict", {})

        backlog = float(row.get("backlog_units", 3500.0))
        surge_pct = float(row.get("surge_pct", 0.0))
        unit_price = float(row.get("unit_price_usd", 1200.0))

        # Reconcile baseline ML forecast + net opportunity uplift
        base_demand = forecast.point_forecast_units
        pipeline_uplift = opp.net_probability_weighted_units * 0.5  # conservative consensus weight
        scenario_adj = round((base_demand + pipeline_uplift) * surge_pct, 1)

        consensus_units = max(backlog, round(base_demand + pipeline_uplift + scenario_adj, 1))
        consensus_rev = round(consensus_units * unit_price, 2)

        peer_revisions = context.get("peer_revisions_count", 0)

        contract = ConsensusDemandContract(
            region=forecast.region,
            product_line=forecast.product_line,
            horizon_month=forecast.horizon_month,
            ml_forecast_units=round(base_demand, 1),
            pipeline_uplift_units=round(pipeline_uplift, 1),
            backlog_units=round(backlog, 1),
            scenario_adjustment_units=round(scenario_adj, 1),
            consensus_demand_units=round(consensus_units, 1),
            consensus_revenue_usd=round(consensus_rev, 2),
            confidence_score=round(max(0.70, forecast.confidence_score - (peer_revisions * 0.02)), 3),
            peer_revisions_count=peer_revisions,
            consensus_rationale=(
                f"Consensus reached by combining ML baseline ({base_demand:.0f}u), "
                f"CRM probability uplift ({pipeline_uplift:.0f}u), and surge (+{surge_pct*100:.0f}%)."
            ),
        )
        
        blackboard = context.get("blackboard")
        if blackboard:
            claim = AgentClaim(
                agent_id=self.agent_type.value,
                topic="consensus_demand",
                statement=f"Consensus demand is ${consensus_rev:,.0f}.",
                value=consensus_rev,
                unit="USD",
                assumptions=[contract.consensus_rationale],
                confidence=contract.confidence_score,
                status="PROPOSED"
            )
            blackboard.add_claim(claim)
            
        if self.agent_type.value in context.get("selected_agents", []):
            directive = context.get("agent_directives", {}).get(self.agent_type.value)
            if directive:
                contract = self._apply_llm_directive(contract, directive)
                
        return self.validate_contract(contract)


class FactoryPlanningAgent(BaseAgent):
    """Maps consensus demand to factories, detects utilization overloads, and plans transfer shifts."""

    def __init__(self, factory_rules: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.FACTORY_PLANNING)
        self.factory_rules = factory_rules or {}

    def process(self, context: Dict[str, Any]) -> FactoryAllocationContract:
        demand: ConsensusDemandContract = context["demand_contract"]
        row = context.get("row_dict", {})

        primary_factory = str(row.get("factory", "Factory A (Stuttgart)"))
        effective_cap = float(row.get("effective_capacity", 12000.0))
        target_demand = demand.consensus_demand_units

        # Secondary factories for load balancing
        secondary_factories = {
            "Factory A (Stuttgart)": ("Factory D (Wroclaw)", 14000.0),
            "Factory B (Detroit)": ("Factory E (Monterrey)", 16000.0),
            "Factory C (Yokohama)": ("Factory B (Detroit)", 15000.0),
            "Factory D (Wroclaw)": ("Factory A (Stuttgart)", 12000.0),
            "Factory E (Monterrey)": ("Factory B (Detroit)", 15000.0),
        }
        alt_factory, alt_cap = secondary_factories.get(primary_factory, ("Factory D (Wroclaw)", 14000.0))

        allocations: Dict[str, float] = {}
        capacities: Dict[str, float] = {primary_factory: effective_cap, alt_factory: alt_cap}
        utilizations: Dict[str, float] = {}
        overloaded: List[str] = []
        shifts: List[Dict[str, Any]] = []

        # If primary capacity is exceeded, shift excess to secondary factory
        if target_demand > effective_cap:
            overload_units = round(target_demand - effective_cap, 1)
            allocations[primary_factory] = round(effective_cap, 1)
            allocations[alt_factory] = overload_units
            overloaded.append(primary_factory)
            shifts.append(
                {
                    "from_factory": primary_factory,
                    "to_factory": alt_factory,
                    "units_shifted": overload_units,
                    "transfer_cost_usd": round(overload_units * 25.0, 2),
                }
            )
        else:
            allocations[primary_factory] = round(target_demand, 1)
            allocations[alt_factory] = 0.0

        for f_name, alloc in allocations.items():
            cap = capacities[f_name]
            util = round((alloc / (cap + 1e-5)) * 100.0, 1)
            utilizations[f_name] = util

        contract = FactoryAllocationContract(
            total_demand_units=round(target_demand, 1),
            factory_allocations=allocations,
            factory_capacities=capacities,
            factory_utilizations=utilizations,
            overloaded_factories=overloaded,
            transfer_shifts=shifts,
            confidence_score=0.92,
        )
        if self.agent_type.value in context.get("selected_agents", []):
            directive = context.get("agent_directives", {}).get(self.agent_type.value)
            if directive:
                contract = self._apply_llm_directive(contract, directive)
                
        return self.validate_contract(contract)


class ProductionAgent(BaseAgent):
    """Converts factory allocations into shop-floor schedules, bottlenecks, and overtime recommendations."""

    def __init__(self):
        super().__init__(AgentType.PRODUCTION)

    def process(self, context: Dict[str, Any]) -> ProductionPlanContract:
        factory_contract: FactoryAllocationContract = context["factory_contract"]
        primary_factory = list(factory_contract.factory_allocations.keys())[0]
        scheduled = factory_contract.factory_allocations[primary_factory]
        util = factory_contract.factory_utilizations[primary_factory]

        bottleneck = util > 95.0
        reason = "Primary assembly line utilization exceeds safe 95% throughput threshold." if bottleneck else None
        overtime_hours = round(max(0.0, (util - 90.0) * 12.5), 1) if bottleneck else 0.0

        contract = ProductionPlanContract(
            factory_name=primary_factory,
            scheduled_units=scheduled,
            effective_utilization_pct=util,
            bottleneck_detected=bottleneck,
            bottleneck_reason=reason,
            overtime_hours_required=overtime_hours,
            prebuild_units_recommended=round(scheduled * 0.10, 1) if bottleneck else 0.0,
            production_cost_usd=round(scheduled * 45.0 + overtime_hours * 72.0, 2),
            confidence_score=0.94,
        )
        return self.validate_contract(contract)


class SupplyChainAgent(BaseAgent):
    """Performs Bill of Materials explosion, inventory balancing, supplier risk checks, and shortage detection."""

    def __init__(self):
        super().__init__(AgentType.SUPPLY_CHAIN)

    def process(self, context: Dict[str, Any]) -> SupplyChainAuditContract:
        demand: ConsensusDemandContract = context["demand_contract"]
        row = context.get("row_dict", {})

        total_demand = demand.consensus_demand_units
        mat_A1_ratio = float(row.get("mat_A1_req", 1.0))
        mat_B2_ratio = float(row.get("mat_B2_req", 1.0))
        mat_C3_ratio = float(row.get("mat_C3_req", 1.0))
        
        stock_A1 = float(row.get("mat_A1_initial_stock", 5000.0))
        stock_B2 = float(row.get("mat_B2_initial_stock", 5000.0))
        stock_C3 = float(row.get("mat_C3_initial_stock", 5000.0))
        
        transit_A1 = float(row.get("mat_A1_in_transit", 1000.0))
        transit_B2 = float(row.get("mat_B2_in_transit", 1000.0))
        transit_C3 = float(row.get("mat_C3_in_transit", 1000.0))
        
        delay_A1 = int(row.get("mat_A1_delay_days", 0))
        delay_B2 = int(row.get("mat_B2_delay_days", 0))
        delay_C3 = int(row.get("mat_C3_delay_days", 0))

        bom = {"MAT-A1": mat_A1_ratio, "MAT-B2": mat_B2_ratio, "MAT-C3": mat_C3_ratio}
        required_mats = {
            "MAT-A1": round(total_demand * mat_A1_ratio, 1),
            "MAT-B2": round(total_demand * mat_B2_ratio, 1),
            "MAT-C3": round(total_demand * mat_C3_ratio, 1)
        }

        usable_A1 = 0.0 if delay_A1 > 7 else transit_A1
        usable_B2 = 0.0 if delay_B2 > 7 else transit_B2
        usable_C3 = 0.0 if delay_C3 > 7 else transit_C3
        
        current_inv = {"MAT-A1": stock_A1, "MAT-B2": stock_B2, "MAT-C3": stock_C3}
        transit_inv = {"MAT-A1": transit_A1, "MAT-B2": transit_B2, "MAT-C3": transit_C3}

        shortages = {
            "MAT-A1": max(0.0, round(required_mats["MAT-A1"] - (stock_A1 + usable_A1), 1)),
            "MAT-B2": max(0.0, round(required_mats["MAT-B2"] - (stock_B2 + usable_B2), 1)),
            "MAT-C3": max(0.0, round(required_mats["MAT-C3"] - (stock_C3 + usable_C3), 1))
        }

        shortage_detected = any(s > 0 for s in shortages.values())
        cost_engine = get_cost_engine()
        critical_materials = [m for m, s in shortages.items() if s > 0]
        total_expedite_cost = sum(s * cost_engine.get_expedite_cost(m) for m, s in shortages.items())

        expedite_options = []
        if shortage_detected:
            for mat_id in critical_materials:
                unit_exp_cost = cost_engine.get_expedite_cost(mat_id)
                expedite_options.append(
                    {
                        "material_id": mat_id,
                        "shortage_units": shortages[mat_id],
                        "expedite_cost_usd": round(shortages[mat_id] * unit_exp_cost, 2),
                        "lead_time_compressed_days": 14,
                    }
                )

        max_shortage_u = max(shortages.values()) if shortages else 0.0
        if shortage_detected and max_shortage_u > 50:
            delivery_risk = RiskLevel.HIGH
        elif shortage_detected:
            delivery_risk = RiskLevel.MEDIUM
        else:
            delivery_risk = RiskLevel.LOW

        contract = SupplyChainAuditContract(
            bill_of_materials=bom,
            required_materials=required_mats,
            current_inventory=current_inv,
            in_transit_inventory=transit_inv,
            shortages=shortages,
            material_shortage_detected=shortage_detected,
            critical_materials_at_risk=critical_materials,
            delivery_risk=delivery_risk,
            total_expedite_cost_usd=round(total_expedite_cost, 2),
            confidence_score=0.91,
        )
        
        # Check if we should challenge the consensus demand
        blackboard = context.get("blackboard")
        if blackboard and shortage_detected:
            # Find the consensus demand claim
            consensus_claim = blackboard.get_latest_claim_for_topic("consensus_demand")
            if consensus_claim:
                challenge_msg = AgentMessage(
                    sender=self.agent_type.value,
                    recipients=[AgentType.DEMAND_PLANNING.value, AgentType.FACTORY_PLANNING.value],
                    message_type="CHALLENGE",
                    claim_ids=[consensus_claim.claim_id],
                    requested_action=f"Revise demand or escalate. Shortages: {', '.join([f'{m}: {s}u' for m, s in shortages.items() if s > 0])}."
                )
                blackboard.add_message(challenge_msg)
                
                reasoning = context.get("reasoning_supply", "")
                
                # Propose a correction claim for the shortage
                correction_claim = AgentClaim(
                    agent_id=self.agent_type.value,
                    topic="material_constraint",
                    statement=f"Demand exceeds available materials. Shortages: {', '.join([f'{m}: {s}u' for m, s in shortages.items() if s > 0])}. {reasoning}",
                    value=sum(shortages.values()),
                    unit="units_short",
                    assumptions=[f"Supplier delay A1: {delay_A1}d, B2: {delay_B2}d, C3: {delay_C3}d"],
                    confidence=0.99,
                    status="SUPPORTED"
                )
                blackboard.add_claim(correction_claim)
        
        if self.agent_type.value in context.get("selected_agents", []):
            directive = context.get("agent_directives", {}).get(self.agent_type.value)
            if directive:
                contract = self._apply_llm_directive(contract, directive)
                
        return self.validate_contract(contract)


class ScenarioAgent(BaseAgent):
    """Executes what-if scenario simulations through the closed-loop Digital Twin."""

    def __init__(self, simulator: DigitalTwinSimulator):
        super().__init__(AgentType.SCENARIO)
        self.simulator = simulator

    def process(self, context: Dict[str, Any]) -> ScenarioSimulationContract:
        demand: ConsensusDemandContract = context["demand_contract"]
        row = context.get("row_dict", {})

        scenario_name = context.get("scenario_name", "Demand Surge +18% & Supplier Delay")
        demand_mult = float(context.get("demand_multiplier", 1.18))
        supplier_delay = int(context.get("supplier_delay_days", 14))
        capacity_mult = float(context.get("capacity_multiplier", 1.0))

        supply_contract = context.get("supply_contract")
        if supply_contract and hasattr(supply_contract, "current_inventory") and supply_contract.current_inventory:
            sim_current_inv = supply_contract.current_inventory
            sim_transit_inv = supply_contract.in_transit_inventory
        else:
            sim_current_inv = {"MAT-M1": float(row.get("mat1_initial_stock", 10000.0))}
            sim_transit_inv = {"MAT-M1": float(row.get("mat1_in_transit", 1000.0))}

        sim_res = self.simulator.simulate_scenario(
            scenario_id=str(row.get("scenario_id", "SCN-SIM-01")),
            scenario_description=scenario_name,
            baseline_demand_units=demand.consensus_demand_units,
            unit_price_usd=float(row.get("unit_price_usd", 1200.0)),
            primary_factory=str(row.get("factory", "Factory A (Stuttgart)")),
            current_inventory=sim_current_inv,
            in_transit_inventory=sim_transit_inv,
            demand_multiplier=demand_mult,
            supplier_delay_days=supplier_delay,
            capacity_multiplier=capacity_mult,
        )
        sim_contract = sim_res["contract"]
        prompt = f"""
You are the Scenario Digital Twin AI.
You have just simulated a supply chain shock: "{scenario_name}"
Demand Multiplier: {demand_mult}
Capacity Multiplier: {capacity_mult}
Supplier Delay: {supplier_delay} days
Simulated Shortage: {sim_contract.simulated_shortage_units} units
Simulated Max Factory Util: {sim_contract.simulated_max_factory_utilization}%

Write a 2-3 sentence executive explanation of what this digital twin simulation means for the enterprise.
"""
        try:
            from src.agents.llm_client import generate_llm_response
            sim_contract.scenario_description = generate_llm_response(prompt)
        except:
            sim_contract.scenario_description = scenario_name
            
        return sim_contract


class RecommendationAgent(BaseAgent):
    """Synthesizes all findings into a prioritized, ranked action plan with cost-benefit analysis."""

    def __init__(self):
        super().__init__(AgentType.RECOMMENDATION)

    def process(self, context: Dict[str, Any]) -> RecommendationPlanContract:
        factory: FactoryAllocationContract = context["factory_contract"]
        supply: SupplyChainAuditContract = context["supply_contract"]
        opp: OpportunityContract = context["opportunity_contract"]
        milp_plan = context.get("milp_plan")
        row = context.get("row_dict", {})
        unit_price = float(row.get("unit_price_usd", 1200.0))
        cost_engine = get_cost_engine()

        recs: List[ActionRecommendation] = []
        rank = 1
        total_cost = 0.0
        total_protected = 0.0

        if milp_plan and getattr(milp_plan, "success", False):
            # Driven directly by the master operations research optimizer (MILP)
            # 1. Overtime actions
            for f_name, ot_hrs in milp_plan.overtime_hours.items():
                if ot_hrs > 0:
                    ot_cost = ot_hrs * cost_engine.get_overtime_cost(f_name)
                    ot_units = ot_hrs * 6.0
                    ot_saved = ot_units * unit_price
                    recs.append(
                        ActionRecommendation(
                            priority_rank=rank,
                            action_type="SCHEDULE_OVERTIME",
                            description=f"Authorize {ot_hrs:.0f} overtime hours at {f_name} to produce {ot_units:.0f} units.",
                            target_entity=f_name,
                            cost_usd=round(ot_cost, 2),
                            protected_revenue_usd=round(ot_saved, 2),
                            risk_mitigation_score=0.94,
                        )
                    )
                    rank += 1
                    total_cost += ot_cost
                    total_protected += ot_saved

            # 2. Transfer shifts
            for shift in milp_plan.transfer_shifts:
                shifted_u = shift["units_shifted"]
                cost = shift.get("transfer_cost_usd", shifted_u * 25.0)
                saved = shifted_u * unit_price
                recs.append(
                    ActionRecommendation(
                        priority_rank=rank,
                        action_type="SHIFT_FACTORY_TRANSFER",
                        description=f"Shift {shifted_u:.0f} units from {shift['from_factory']} to {shift['to_factory']} via optimal network routing.",
                        target_entity=shift["to_factory"],
                        cost_usd=round(cost, 2),
                        protected_revenue_usd=round(saved, 2),
                        risk_mitigation_score=0.92,
                    )
                )
                rank += 1
                total_cost += cost
                total_protected += saved

            # 3. Expedites
            for mat_id, exp_u in milp_plan.expedite_units.items():
                if exp_u > 0:
                    exp_cost = exp_u * cost_engine.get_expedite_cost(mat_id)
                    req_ratio = supply.bill_of_materials.get(mat_id, 1.0)
                    saved = (exp_u / req_ratio) * unit_price
                    recs.append(
                        ActionRecommendation(
                            priority_rank=rank,
                            action_type="EXPEDITE_CRITICAL_SUPPLIER",
                            description=f"Air-freight expedite {exp_u:.0f} units of {mat_id} to avoid stockouts.",
                            target_entity=mat_id,
                            cost_usd=round(exp_cost, 2),
                            protected_revenue_usd=round(saved, 2),
                            risk_mitigation_score=0.90,
                        )
                    )
                    rank += 1
                    total_cost += exp_cost
                    total_protected += saved

            # 4. Spot sourcing if used
            if milp_plan.emergency_sourcing_units > 0:
                spot_cost = milp_plan.emergency_sourcing_units * 105.0
                spot_saved = milp_plan.emergency_sourcing_units * unit_price
                recs.append(
                    ActionRecommendation(
                        priority_rank=rank,
                        action_type="EMERGENCY_SPOT_SOURCING",
                        description=f"Procure {milp_plan.emergency_sourcing_units:.0f} emergency spot units to fulfill unmet demand.",
                        target_entity="Critical Materials",
                        cost_usd=round(spot_cost, 2),
                        protected_revenue_usd=round(spot_saved, 2),
                        risk_mitigation_score=0.88,
                    )
                )
                rank += 1
                total_cost += spot_cost
                total_protected += spot_saved
        else:
            # Fallback contract-based heuristics
            if len(factory.overloaded_factories) > 0 and len(factory.transfer_shifts) > 0:
                shift = factory.transfer_shifts[0]
                shifted_u = shift["units_shifted"]
                cost = shift["transfer_cost_usd"]
                saved = shifted_u * unit_price
                recs.append(
                    ActionRecommendation(
                        priority_rank=rank,
                        action_type="SHIFT_FACTORY_TRANSFER",
                        description=f"Shift {shifted_u:.0f} units from {shift['from_factory']} to {shift['to_factory']} to avoid 100%+ overload.",
                        target_entity=shift["to_factory"],
                        cost_usd=round(cost, 2),
                        protected_revenue_usd=round(saved, 2),
                        risk_mitigation_score=0.92,
                    )
                )
                rank += 1
                total_cost += cost
                total_protected += saved

            if supply.material_shortage_detected:
                for mat_id in supply.critical_materials_at_risk:
                    shortage_u = supply.shortages.get(mat_id, 0)
                    if shortage_u > 0:
                        exp_cost = shortage_u * cost_engine.get_expedite_cost(mat_id)
                        req_ratio = supply.bill_of_materials.get(mat_id, 1.0)
                        saved = (shortage_u / req_ratio) * unit_price
                        recs.append(
                            ActionRecommendation(
                                priority_rank=rank,
                                action_type="EXPEDITE_CRITICAL_SUPPLIER",
                                description=f"Air-freight expedite {shortage_u:.0f} units of {mat_id} to compress lead time.",
                                target_entity=mat_id,
                                cost_usd=round(exp_cost, 2),
                                protected_revenue_usd=round(saved, 2),
                                risk_mitigation_score=0.89,
                            )
                        )
                        rank += 1
                        total_cost += exp_cost
                        total_protected += saved

        # Action 3: Confirm High-Risk Opportunities with Sales
        if opp.at_risk_deals_count > 0:
            recs.append(
                ActionRecommendation(
                    priority_rank=rank,
                    action_type="CONFIRM_SALES_COMMITMENTS",
                    description=f"Request formal sales confirmation on {opp.at_risk_deals_count} stale/at-risk enterprise deals (${opp.stale_deals_usd:,.0f}).",
                    target_entity=opp.region,
                    cost_usd=0.0,
                    protected_revenue_usd=round(opp.stale_deals_usd * 0.35, 2),
                    risk_mitigation_score=0.85,
                )
            )
            total_protected += opp.stale_deals_usd * 0.35

        # If no risks, maintain standard plan
        if not recs:
            recs.append(
                ActionRecommendation(
                    priority_rank=1,
                    action_type="MAINTAIN_STANDARD_SCHEDULE",
                    description="Capacity and supply chain are healthy; execute standard production run.",
                    target_entity=str(row.get("factory", "Factory A")),
                    cost_usd=0.0,
                    protected_revenue_usd=0.0,
                    risk_mitigation_score=1.0,
                )
            )

        contract = RecommendationPlanContract(
            recommendations=recs,
            total_cost_usd=round(total_cost, 2),
            total_protected_revenue_usd=round(total_protected, 2),
            net_benefit_usd=round(total_protected - total_cost, 2),
            post_recommendation_delivery_risk=RiskLevel.LOW if total_protected > 0 else RiskLevel.LOW,
            expected_capacity_normalized_pct=91.5,
            confidence_score=0.92,
        )
        return self.validate_contract(contract)

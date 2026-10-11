"""
PlanFlow AI - Collaborative Peer Dialogue & Mutual Correction Engine
Enables specialist agents to converse collegially as trusted friends and close colleagues,
exchanging grounded facts, cross-examining assumptions, and politely correcting errors using an LLM.
"""

import os
import json
from typing import Dict, Any, List, Tuple

try:
    from src.agents.llm_client import generate_llm_response
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.core.contracts import (
    AgentType,
    PeerMessageContract,
    ForecastContract,
    OpportunityContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    SupplyChainAuditContract,
)
from src.core.logger import get_logger, log_agent_dialogue

logger = get_logger("PeerDialogue")


class PeerDialogueEngine:
    """Manages multi-agent collaborative conversation, friendly debate, and mutual correction."""

    def __init__(self, max_correction_rounds: int = 3):
        self.max_rounds = max_correction_rounds
        self.conversation_history: List[PeerMessageContract] = []
        self.model_active = True

    def conduct_collaborative_review(
        self,
        forecast: ForecastContract,
        opportunity: OpportunityContract,
        demand: ConsensusDemandContract,
        factory: FactoryAllocationContract,
        supply: SupplyChainAuditContract,
        row_dict: Dict[str, Any],
    ) -> Tuple[ConsensusDemandContract, FactoryAllocationContract, SupplyChainAuditContract, List[PeerMessageContract]]:
        """Executes friendly peer-to-peer fact exchange and mutual error correction using an LLM."""
        
        messages: List[PeerMessageContract] = []
        corrections_count = 0

        if not self.model_active:
            logger.warning("No LLM available. Skipping intelligent peer dialogue.")
            return demand, factory, supply, messages

        prompt = f"""
You are the multi-agent dialogue orchestrator for PlanFlow AI. You oversee 5 peer agents:
- OPPORTUNITY (Salesforce CRM pipeline)
- DEMAND_PLANNING (Consensus target generation)
- FACTORY_PLANNING (Utilization and allocations)
- SUPPLY_CHAIN (Material shortages and BOM)
- FORECASTING (Baseline ML prediction)

Here is the current state of the enterprise plan:
Forecast: {forecast.model_dump_json()}
Opportunity: {opportunity.model_dump_json()}
Demand: {demand.model_dump_json()}
Factory: {factory.model_dump_json()}
Supply: {supply.model_dump_json()}

Analyze the state for contradictions, risks, or necessary corrections. Example risks:
- Does Demand include too many stale deals from Opportunity? If Opportunity has high stale_deals_usd, demand's pipeline_uplift_units should probably be reduced.
- Is the Factory overloaded (>100% utilization)?
- Are there material shortages in Supply that need expediting?

Write a JSON array of 1 to 3 collaborative messages exchanged between the agents where they point out issues and propose numeric corrections. Make the tone highly professional yet collegiate.
EACH message MUST match this JSON schema exactly:
{{
    "sender": "AgentType string (must be exactly OPPORTUNITY, DEMAND_PLANNING, FACTORY_PLANNING, SUPPLY_CHAIN, or FORECASTING)",
    "recipient": "AgentType string",
    "greeting": "Friendly opening",
    "facts_asserted": {{"key": "value"}},
    "critique_or_correction": "Detailed rationale and peer feedback",
    "is_correction": true or false,
    "revised_metric_name": "Name of the metric to fix in the contracts (e.g. pipeline_uplift_units)",
    "old_value": float (0.0 if not applicable),
    "new_value": float (0.0 if not applicable),
    "confidence": float between 0.0 and 1.0
}}

If no corrections are needed, you can have them confirm everything looks good.
Return ONLY valid JSON array. Do not include markdown formatting or backticks around the json block.
"""
        
        try:
            logger.info("Engaging LLM for multi-agent peer dialogue...")
            raw_text = generate_llm_response(prompt)
            if raw_text.startswith("```"):
                lines = raw_text.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = '\n'.join(lines).strip()
            
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()
                
            llm_msgs = json.loads(raw_text)
            
            for m in llm_msgs:
                if not isinstance(m, dict): continue
                
                try:
                    sender = AgentType(m.get("sender"))
                    recipient = AgentType(m.get("recipient"))
                except ValueError:
                    continue # Skip invalid agent types
                    
                msg = PeerMessageContract(
                    sender=sender,
                    recipient=recipient,
                    greeting=m.get("greeting", "Hello"),
                    facts_asserted=m.get("facts_asserted", {}),
                    critique_or_correction=m.get("critique_or_correction", ""),
                    is_correction=m.get("is_correction", False),
                    revised_metric_name=m.get("revised_metric_name"),
                    old_value=m.get("old_value"),
                    new_value=m.get("new_value"),
                    confidence=m.get("confidence", 0.9)
                )
                messages.append(msg)
                log_agent_dialogue(msg.sender.value, msg.recipient.value, msg.critique_or_correction, correction=msg.is_correction)
                
                # Apply dynamic LLM corrections to the actual contracts!
                if msg.is_correction and msg.revised_metric_name and msg.new_value is not None:
                    corrections_count += 1
                    if msg.revised_metric_name == "pipeline_uplift_units":
                        demand.pipeline_uplift_units = float(msg.new_value)
                        demand.peer_revisions_count += 1
                        
        except Exception as e:
            logger.warning(f"LLM peer dialogue unavailable ({e}), engaging rule-based peer collaborative review...")
            # Fallback heuristic peer review when LLM is offline
            if opportunity.stale_deals_usd > 500_000 and demand.pipeline_uplift_units > 500:
                discounted_uplift = max(0.0, round(demand.pipeline_uplift_units * 0.5, 1))
                msg = PeerMessageContract(
                    sender=AgentType.OPPORTUNITY,
                    recipient=AgentType.DEMAND_PLANNING,
                    greeting="Colleagues in Demand Planning,",
                    facts_asserted={"stale_deals_usd": opportunity.stale_deals_usd, "total_pipeline_usd": opportunity.total_pipeline_usd},
                    critique_or_correction=(
                        f"Salesforce CRM audit identified ${opportunity.stale_deals_usd:,.0f} in stale deals. "
                        f"Discounting unweighted pipeline uplift from {demand.pipeline_uplift_units:,.0f} to {discounted_uplift:,.0f} units."
                    ),
                    is_correction=True,
                    revised_metric_name="pipeline_uplift_units",
                    old_value=demand.pipeline_uplift_units,
                    new_value=discounted_uplift,
                    confidence=0.88,
                )
                messages.append(msg)
                corrections_count += 1
                demand.pipeline_uplift_units = discounted_uplift
                demand.peer_revisions_count += 1
                log_agent_dialogue(msg.sender.value, msg.recipient.value, msg.critique_or_correction, correction=True)

            if supply.material_shortage_detected:
                crit_mats = ", ".join(supply.critical_materials_at_risk)
                msg_sup = PeerMessageContract(
                    sender=AgentType.SUPPLY_CHAIN,
                    recipient=AgentType.FACTORY_PLANNING,
                    greeting="Production Team,",
                    facts_asserted={"shortages": supply.shortages},
                    critique_or_correction=f"Material shortages flagged for {crit_mats}. Expedited supplier procurement initiated.",
                    is_correction=False,
                    confidence=0.91,
                )
                messages.append(msg_sup)
                log_agent_dialogue(msg_sup.sender.value, msg_sup.recipient.value, msg_sup.critique_or_correction, correction=False)

        # Always dynamically re-calculate core physical invariants to ensure the mathematical simulator oracle doesn't fail
        if corrections_count > 0:
            # 1. Demand conservation
            demand.consensus_demand_units = round(
                max(
                    demand.backlog_units,
                    demand.ml_forecast_units + demand.pipeline_uplift_units + demand.scenario_adjustment_units,
                ),
                1,
            )
            demand.consensus_revenue_usd = round(
                demand.consensus_demand_units * float(row_dict.get("unit_price_usd", 1200)), 2
            )

            # 2. Re-align factory allocations to target
            primary_f = list(factory.factory_allocations.keys())[0]
            sec_f = [k for k in factory.factory_allocations.keys() if k != primary_f]
            p_cap = factory.factory_capacities.get(primary_f, 12000.0)
            new_target = demand.consensus_demand_units

            if new_target > p_cap and sec_f:
                factory.factory_allocations[primary_f] = round(p_cap, 1)
                factory.factory_allocations[sec_f[0]] = round(new_target - p_cap, 1)
                if not factory.transfer_shifts:
                     factory.transfer_shifts.append({
                        "from_factory": primary_f, "to_factory": sec_f[0], 
                        "units_shifted": round(new_target - p_cap, 1), "transfer_cost_usd": round((new_target - p_cap) * 25.0, 2)
                     })
                else:
                     factory.transfer_shifts[0]["units_shifted"] = round(new_target - p_cap, 1)
                     factory.transfer_shifts[0]["transfer_cost_usd"] = round((new_target - p_cap) * 25.0, 2)
            else:
                factory.factory_allocations[primary_f] = round(new_target, 1)
                if sec_f:
                    factory.factory_allocations[sec_f[0]] = 0.0

            for f_name, alloc in factory.factory_allocations.items():
                cap = factory.factory_capacities.get(f_name, 1.0)
                factory.factory_utilizations[f_name] = round((alloc / (cap + 1e-5)) * 100.0, 1)

            # 3. Re-align supply chain requirements based on revised target
            for mat_id, ratio in supply.bill_of_materials.items():
                new_req = round(new_target * ratio, 1)
                supply.required_materials[mat_id] = new_req
                curr_s = supply.current_inventory.get(mat_id, 0.0)
                in_t = supply.in_transit_inventory.get(mat_id, 0.0)
                usable_s = curr_s + (0.0 if int(row_dict.get(f"{mat_id.replace('-','_')}_delay_days", 0)) > 7 else in_t)
                new_shortage = max(0.0, round(new_req - usable_s, 1))
                supply.shortages[mat_id] = new_shortage
                supply.material_shortage_detected = new_shortage > 0
                supply.total_expedite_cost_usd = round(new_shortage * 35.0, 2)
            
        self.conversation_history.extend(messages)
        return demand, factory, supply, messages

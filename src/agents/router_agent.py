"""
PlanFlow AI - Chief of Staff Router Agent
Analyzes unstructured scenarios and dynamically routes specific natural language instructions 
to the relevant specialist agents (Demand, Supply, Factory, etc).
"""

import os
import json
from typing import Dict, Any, List

try:
    from src.agents.llm_client import generate_llm_response
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.core.contracts import AgentType
from src.core.logger import get_logger

logger = get_logger("RouterAgent")

class ChiefOfStaffRouter:
    def __init__(self):
        self.model_active = True

    def route_scenario(self, scenario_text: str) -> Dict[str, Any]:
        """
        Parses a scenario into specific Agent directives.
        """
        # Default fallback
        default_routing = {
            "selected_agents": [],
            "agent_directives": {},
            "global_parameters": {
                "demand_multiplier": 1.0,
                "supplier_delay_days": 0,
                "capacity_multiplier": 1.0
            }
        }

        if not scenario_text or len(scenario_text.strip()) < 5 or scenario_text.strip() == "Baseline":
            return default_routing

        if not self.model_active:
            return default_routing

        prompt = f"""
You are the Chief of Staff Router for the PlanFlow AI multi-agent system.
Your job is to read a user's natural language scenario query and figure out WHICH specialist agents need to be queried, and WHAT specific directive to give them.

The available agents are:
- DEMAND_PLANNING (Handles demand multipliers, surges, pipeline)
- FACTORY_PLANNING (Handles capacity cuts, plant shutdowns, utilization)
- SUPPLY_CHAIN (Handles material shortages, supplier delays)

Query: "{scenario_text}"

Determine which agents need to act to satisfy this query.
Return a JSON object EXACTLY matching this schema:
{{
    "selected_agents": ["LIST_OF_AGENT_TYPES"],
    "agent_directives": {{
        "AGENT_TYPE": "A specific 1-sentence instruction for this agent to mathematically adjust its contract based on the query."
    }},
    "global_parameters": {{
        "demand_multiplier": float (e.g., 1.2 for +20% demand, 1.0 for normal),
        "capacity_multiplier": float (e.g., 0.5 for 50% capacity cut, 1.0 for normal),
        "supplier_delay_days": int (e.g., 10 for 10 days delay, 0 for normal)
    }}
}}

Return ONLY valid JSON array or object. Do not include markdown formatting or backticks around the json block.
"""
        
        try:
            logger.info(f"ChiefOfStaff Router analyzing query: {scenario_text}")
            if not self.model_active:
                return default_routing

            raw_text = generate_llm_response(prompt)
            if raw_text.startswith("```"):
                lines = raw_text.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = '\n'.join(lines).strip()
            
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()
                
            routing_plan = json.loads(raw_text)
            
            # Ensure global parameters exist
            if "global_parameters" not in routing_plan:
                routing_plan["global_parameters"] = default_routing["global_parameters"]
                
            logger.info(f"Router Decision: {routing_plan.get('selected_agents', [])}")
            return routing_plan
            
        except Exception as e:
            logger.error(f"Router agent failed to parse LLM response: {e}")
            return default_routing

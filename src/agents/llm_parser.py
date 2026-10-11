import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("LLMParser")

try:
    from src.agents.llm_client import generate_llm_response
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def parse_scenario_query(query: str) -> Dict[str, Any]:
    """
    Parses a natural language scenario query into deterministic simulation parameters.
    Returns a dictionary with multipliers and agent reasoning contexts.
    """
    # Default fallback values
    default_response = {
        "demand_multiplier": 1.0,
        "supplier_delay_days": 0,
        "capacity_multiplier": 1.0,
        "reasoning_forecast": "Executing baseline forecast strategy.",
        "reasoning_supply": "Supplier lead times are expected to be normal.",
        "reasoning_capacity": "Factory operating at normal capacity."
    }

    if not query or len(query.strip()) < 5:
        return default_response

    if HAS_GENAI:
        try:
            prompt = f"""
            You are a planning AI parsing a supply chain scenario query.
            Extract the following parameters from the text query.
            - demand_multiplier: Float (1.0 is normal, 1.2 is +20% surge, 0.8 is -20% drop)
            - supplier_delay_days: Integer (days of delay for materials)
            - capacity_multiplier: Float (1.0 is 100% capacity, 0.5 is 50% capacity, e.g. from hurricanes)
            - reasoning_forecast: A short 1-sentence reasoning for the Forecasting Agent based on the query.
            - reasoning_supply: A short 1-sentence reasoning for the Supply Chain Agent based on the query.
            - reasoning_capacity: A short 1-sentence reasoning for the Factory Agent based on the query.

            Query: "{query}"

            Return ONLY a valid JSON object matching the keys exactly.
            """
            
            raw_text = generate_llm_response(prompt)
            if raw_text.startswith("```"):
                lines = raw_text.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = '\n'.join(lines)
            
            result = json.loads(raw_text)
            
            # Ensure defaults for missing keys
            for k in default_response:
                if k not in result:
                    result[k] = default_response[k]
                    
            logger.info(f"LLM parsed query into: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to parse query with LLM: {e}")
            # Fall back to heuristic below if API fails

    # ---------------------------------------------------------
    # Fallback heuristic if no API key or API fails (Mock LLM)
    # ---------------------------------------------------------
    logger.info("Using heuristic NLP fallback for parsing query.")
    query_lower = query.lower()
    
    res = default_response.copy()
    
    # Demand heuristics
    if "surge" in query_lower or "spike" in query_lower:
        res["demand_multiplier"] = 1.18
        res["reasoning_forecast"] = "Detected an unexpected demand surge from the scenario context."
    elif "drop" in query_lower or "crash" in query_lower:
        res["demand_multiplier"] = 0.85
        res["reasoning_forecast"] = "Detected a demand crash from the scenario context."
        
    # Supply heuristics
    if "delay" in query_lower:
        res["supplier_delay_days"] = 15
        res["reasoning_supply"] = "Supplier delays identified in the scenario; lead times extended."
        
    # Capacity heuristics
    if "hurricane" in query_lower or "storm" in query_lower or "strike" in query_lower:
        res["capacity_multiplier"] = 0.60
        res["reasoning_capacity"] = "Severe capacity disruption detected due to extreme event."
        
    return res

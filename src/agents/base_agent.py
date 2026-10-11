"""
PlanFlow AI - Base Agent Definition
Provides base contract validation, memory of exchanged facts, and logging.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from src.core.contracts import AgentType, PeerMessageContract
from src.core.logger import get_logger
import os
import json

try:
    from src.agents.llm_client import generate_llm_response
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = get_logger("BaseAgent")


class BaseAgent(ABC):
    """Abstract base class for all PlanFlow AI autonomous agents."""

    def __init__(self, agent_type: AgentType, name: Optional[str] = None):
        self.agent_type = agent_type
        self.name = name or agent_type.value
        self.memory: List[Dict[str, Any]] = []
        self.peer_messages: List[PeerMessageContract] = []

    def record_fact(self, key: str, value: Any, source: str):
        """Stores verified factual knowledge in working memory."""
        self.memory.append({"key": key, "value": value, "source": source})

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieves most recent fact from working memory."""
        for item in reversed(self.memory):
            if item["key"] == key:
                return item["value"]
        return default

    def validate_contract(self, contract_instance: BaseModel) -> BaseModel:
        """Enforces runtime validation of agent output schemas."""
        # Pydantic v2 automatically validates on instantiate
        return contract_instance

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> BaseModel:
        """Executes the agent's core reasoning and returns a validated contract."""
        pass

    def _apply_llm_directive(self, contract: BaseModel, directive: str) -> BaseModel:
        """Uses LLM to dynamically apply a Router directive to the agent's deterministic contract."""
        if not HAS_GENAI:
            return contract
            
        prompt = f"""
You are the {self.agent_type.value} specialist agent.
You have generated a baseline plan (in JSON).
Your manager (the Chief of Staff Router) has given you a specific directive to override this plan.

Baseline Plan: {contract.model_dump_json()}
Manager Directive: "{directive}"

Update your baseline plan to reflect the directive.
Return the updated plan EXACTLY matching the original JSON schema.
Return ONLY valid JSON.
"""
        try:
            raw_text = generate_llm_response(prompt)
            if raw_text.startswith("```"):
                lines = raw_text.split('\n')
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_text = '\n'.join(lines).strip()
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()
                
            updated_dict = json.loads(raw_text)
            return type(contract)(**updated_dict)
        except Exception as e:
            logger.error(f"Failed to apply LLM directive in {self.agent_type.value}: {e}")
            return contract

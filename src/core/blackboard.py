"""
PlanFlow AI - Shared Evidence Blackboard
Provides a versioned state store containing accepted facts, proposed facts,
challenges, corrections, assumptions, and unresolved conflicts.
"""

from typing import Dict, List, Optional
from uuid import UUID
from src.core.contracts import AgentClaim, AgentMessage
from src.core.logger import get_logger

logger = get_logger("Blackboard")

class SharedBlackboard:
    def __init__(self):
        self.claims: Dict[UUID, AgentClaim] = {}
        self.messages: List[AgentMessage] = []
        # Index by topic for quick access
        self.topic_index: Dict[str, List[UUID]] = {}
    
    def add_claim(self, claim: AgentClaim) -> None:
        self.claims[claim.claim_id] = claim
        if claim.topic not in self.topic_index:
            self.topic_index[claim.topic] = []
        self.topic_index[claim.topic].append(claim.claim_id)
        
        # If this claim supersedes another, update the superseded claim's status
        if claim.supersedes_claim_id and claim.supersedes_claim_id in self.claims:
            old_claim = self.claims[claim.supersedes_claim_id]
            if old_claim.status != "REJECTED":
                old_claim.status = "REVISED"
                
        logger.info(f"Blackboard: Added claim '{claim.topic}' by {claim.agent_id} (Status: {claim.status})")

    def update_claim_status(self, claim_id: UUID, new_status: str) -> None:
        if claim_id in self.claims:
            self.claims[claim_id].status = new_status
            logger.info(f"Blackboard: Claim {claim_id} status updated to {new_status}")

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)
        logger.info(f"Blackboard: [{message.message_type}] from {message.sender}")
        
        # Automatically update claim status if message is a CHALLENGE
        if message.message_type == "CHALLENGE":
            for cid in message.claim_ids:
                if cid in self.claims:
                    self.update_claim_status(cid, "CHALLENGED")

    def get_claims_by_topic(self, topic: str, active_only: bool = True) -> List[AgentClaim]:
        c_ids = self.topic_index.get(topic, [])
        found_claims = [self.claims[cid] for cid in c_ids]
        if active_only:
            found_claims = [c for c in found_claims if c.status in ["PROPOSED", "SUPPORTED", "ACCEPTED", "CHALLENGED"]]
        return found_claims
    
    def get_latest_claim_for_topic(self, topic: str) -> Optional[AgentClaim]:
        claims = self.get_claims_by_topic(topic, active_only=True)
        if claims:
            # Return the one added last
            return claims[-1]
        return None

    def get_all_active_claims(self) -> List[AgentClaim]:
        return [c for c in self.claims.values() if c.status in ["PROPOSED", "SUPPORTED", "ACCEPTED", "CHALLENGED"]]

    def get_messages(self) -> List[AgentMessage]:
        return self.messages

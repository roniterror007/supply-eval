"""
PlanFlow AI - Evidence & Provenance Data Models (HELICASE Framework)
Defines structured nodes and edges for data provenance, temporal validity windows,
confidence annotations, and uncertainty contribution tracking.
"""

from uuid import UUID, uuid4
from datetime import datetime
from typing import Literal, Any, List, Optional
from pydantic import BaseModel, Field

EntityType = Literal["sku", "supplier", "factory", "order", "forecast", "constraint", "plan", "decision", "recommendation"]
SourceSystem = Literal["ERP", "MES", "CRM", "WMS", "supplier_portal", "manual"]
RelationType = Literal["supports", "derives", "contradicts", "supersedes", "assumes"]


class EvidenceNode(BaseModel):
    """Represents a discrete verified fact or measurement in the enterprise data lake."""
    evidence_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    entity_id: str
    value: Any
    source_system: SourceSystem
    source_record_id: str
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: float = Field(ge=0.0, le=1.0)
    quality_flags: List[str] = Field(default_factory=list)


class EvidenceEdge(BaseModel):
    """Represents an epistemic derivation, dependency, or contradiction between evidence nodes."""
    parent_evidence_id: UUID
    child_evidence_id: UUID
    relation: RelationType
    transformation: str
    agent_id: Optional[str] = None
    uncertainty_contribution: float = Field(default=0.05, ge=0.0, le=1.0)

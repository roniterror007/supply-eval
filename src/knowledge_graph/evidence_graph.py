"""
PlanFlow AI - Temporal Provenance DAG Engine (HELICASE Architecture)
Maintains an immutable, time-indexed directed acyclic graph of enterprise evidence,
temporal validity windows [valid_from, valid_to], and supports decision replay.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
import networkx as nx
from src.core.contracts import EvidenceReference
from src.knowledge_graph.models import EvidenceNode, EvidenceEdge
from src.knowledge_graph.source_quality import SourceQualityEngine
from src.core.logger import get_logger

logger = get_logger("EvidenceGraph")


class TemporalEvidenceGraph:
    """
    Temporal Provenance Directed Acyclic Graph (DAG) for enterprise planning.
    Ensures every fact, forecast, and decision memo is traceable to root observations.
    """

    def __init__(self):
        self.dag = nx.DiGraph()
        self.nodes: Dict[UUID, EvidenceNode] = {}
        self.edges: List[EvidenceEdge] = []
        self.quality_engine = SourceQualityEngine()

    def add_fact(
        self,
        entity_type: str,
        entity_id: str,
        value: Any,
        source_system: str,
        source_record_id: str,
        confidence: float = 1.0,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        observed_at: Optional[datetime] = None,
        quality_flags: Optional[List[str]] = None,
    ) -> EvidenceNode:
        """Inserts an atomic enterprise observation into the temporal graph."""
        obs_time = observed_at or datetime.now(timezone.utc)
        node = EvidenceNode(
            evidence_id=uuid4(),
            entity_type=entity_type,  # type: ignore
            entity_id=entity_id,
            value=value,
            source_system=source_system,  # type: ignore
            source_record_id=source_record_id,
            observed_at=obs_time,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            quality_flags=quality_flags or [],
        )
        self.nodes[node.evidence_id] = node
        self.dag.add_node(
            node.evidence_id,
            entity_type=entity_type,
            entity_id=entity_id,
            value=value,
            source_system=source_system,
            observed_at=obs_time,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
        )
        return node

    def add_derivation(
        self,
        parent_id: UUID,
        child_id: UUID,
        relation: str = "derives",
        transformation: str = "agent_inference",
        agent_id: Optional[str] = None,
        uncertainty_contribution: float = 0.05,
    ) -> EvidenceEdge:
        """Adds a directed derivation edge, enforcing DAG acyclicity."""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent evidence {parent_id} does not exist in graph.")
        if child_id not in self.nodes:
            raise KeyError(f"Child evidence {child_id} does not exist in graph.")

        # Check for cycles
        if nx.has_path(self.dag, child_id, parent_id):
            raise ValueError(
                f"Adding edge {parent_id} -> {child_id} would introduce a cycle in the provenance DAG!"
            )

        edge = EvidenceEdge(
            parent_evidence_id=parent_id,
            child_evidence_id=child_id,
            relation=relation,  # type: ignore
            transformation=transformation,
            agent_id=agent_id,
            uncertainty_contribution=uncertainty_contribution,
        )
        self.edges.append(edge)
        self.dag.add_edge(
            parent_id,
            child_id,
            relation=relation,
            transformation=transformation,
            agent_id=agent_id,
            uncertainty_contribution=uncertainty_contribution,
        )
        return edge

    def get_temporal_view(self, as_of: datetime) -> "TemporalEvidenceGraph":
        """
        Extracts a point-in-time subgraph replay as of timestamp `as_of`.
        Filters out nodes observed after `as_of` or expired before `as_of`.
        """
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        subgraph = TemporalEvidenceGraph()
        valid_node_ids: Set[UUID] = set()

        for node_id, node in self.nodes.items():
            obs = node.observed_at.replace(tzinfo=timezone.utc) if node.observed_at.tzinfo is None else node.observed_at
            if obs > as_of:
                continue  # Future fact relative to as_of
            if node.valid_to:
                v_to = node.valid_to.replace(tzinfo=timezone.utc) if node.valid_to.tzinfo is None else node.valid_to
                if v_to < as_of:
                    continue  # Expired fact

            subgraph.nodes[node_id] = node
            subgraph.dag.add_node(node_id, **self.dag.nodes[node_id])
            valid_node_ids.add(node_id)

        for edge in self.edges:
            if edge.parent_evidence_id in valid_node_ids and edge.child_evidence_id in valid_node_ids:
                subgraph.edges.append(edge)
                subgraph.dag.add_edge(
                    edge.parent_evidence_id,
                    edge.child_evidence_id,
                    relation=edge.relation,
                    transformation=edge.transformation,
                    agent_id=edge.agent_id,
                    uncertainty_contribution=edge.uncertainty_contribution,
                )

        return subgraph

    def get_ancestors(self, target_id: UUID) -> List[EvidenceNode]:
        """Returns all upstream ancestor nodes that contributed to target_id."""
        if target_id not in self.nodes:
            return []
        ancestor_ids = nx.ancestors(self.dag, target_id)
        return [self.nodes[aid] for aid in ancestor_ids]

    def get_root_causes(self, target_id: UUID) -> List[EvidenceNode]:
        """Returns the terminal root sources (in-degree == 0) for target_id."""
        ancestors = self.get_ancestors(target_id)
        return [node for node in ancestors if self.dag.in_degree(node.evidence_id) == 0]

    def to_evidence_reference(self, node: EvidenceNode) -> EvidenceReference:
        """Converts an EvidenceNode to an EvidenceReference with evaluated quality."""
        ref = EvidenceReference(
            evidence_id=node.evidence_id,
            source_system=node.source_system,
            source_record_id=node.source_record_id,
            observed_at=node.observed_at,
            valid_from=node.valid_from,
            valid_to=node.valid_to,
            confidence=node.confidence,
        )
        self.quality_engine.evaluate_evidence(ref)
        return ref

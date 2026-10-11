"""
PlanFlow AI - Evidence Provenance Graph Subsystem (NetworkX DAG)
Maintains the directed acyclic graph of enterprise evidence, tracks data lineage,
and propagates uncertainty across agent transformations and handoffs.
"""

from uuid import UUID
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
import networkx as nx
import numpy as np
from src.knowledge_graph.models import EvidenceNode, EvidenceEdge
from src.core.logger import get_logger

logger = get_logger("ProvenanceGraph")


class EvidenceProvenanceGraph:
    """Maintains a directed graph of evidence nodes, derivations, and uncertainty chains."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes_by_id: Dict[UUID, EvidenceNode] = {}
        self.edges_list: List[EvidenceEdge] = []

    def add_node(self, node: EvidenceNode):
        """Inserts an evidence node into the provenance graph."""
        self.nodes_by_id[node.evidence_id] = node
        self.graph.add_node(
            node.evidence_id,
            entity_type=node.entity_type,
            entity_id=node.entity_id,
            value=node.value,
            source_system=node.source_system,
            observed_at=node.observed_at,
            confidence=node.confidence,
            quality_flags=node.quality_flags,
        )

    def add_edge(self, edge: EvidenceEdge):
        """Inserts a directed relation between two evidence nodes."""
        if edge.parent_evidence_id not in self.nodes_by_id:
            raise KeyError(f"Parent evidence {edge.parent_evidence_id} not found in graph.")
        if edge.child_evidence_id not in self.nodes_by_id:
            raise KeyError(f"Child evidence {edge.child_evidence_id} not found in graph.")

        self.edges_list.append(edge)
        self.graph.add_edge(
            edge.parent_evidence_id,
            edge.child_evidence_id,
            relation=edge.relation,
            transformation=edge.transformation,
            agent_id=edge.agent_id,
            uncertainty_contribution=edge.uncertainty_contribution,
        )

    def get_lineage(self, target_evidence_id: UUID) -> List[EvidenceNode]:
        """Returns the complete upstream lineage of evidence leading to target_evidence_id."""
        if target_evidence_id not in self.nodes_by_id:
            return []
        ancestors = nx.ancestors(self.graph, target_evidence_id)
        # Sort in topological order
        ordered_ancestors = [n for n in nx.topological_sort(self.graph) if n in ancestors]
        ordered_ancestors.append(target_evidence_id)
        return [self.nodes_by_id[uid] for uid in ordered_ancestors]

    def propagate_uncertainty(self, target_evidence_id: UUID) -> Tuple[float, List[str]]:
        """
        Propagates uncertainty along the derivation chain to the target evidence.
        Returns (propagated_confidence, list_of_uncertainty_contributors).
        """
        lineage = self.get_lineage(target_evidence_id)
        if not lineage:
            return 1.0, []

        # Start with base node confidence
        base_conf = self.nodes_by_id[target_evidence_id].confidence
        uncertainty_variance = (1.0 - base_conf) ** 2
        contributors = []

        subgraph = self.graph.subgraph([node.evidence_id for node in lineage])
        for u, v, data in subgraph.edges(data=True):
            edge_uncert = data.get("uncertainty_contribution", 0.05)
            transformation = data.get("transformation", "derivation")
            agent = data.get("agent_id", "UnknownAgent")
            uncertainty_variance += (edge_uncert ** 2)
            if edge_uncert >= 0.08:
                contributors.append(f"{agent} ({transformation}, delta_u={edge_uncert:.2f})")

        total_uncertainty = float(np.sqrt(uncertainty_variance))
        propagated_conf = float(np.clip(1.0 - total_uncertainty, 0.40, 0.99))
        return round(propagated_conf, 3), contributors

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Scans the graph for contradictory relations."""
        contradictions = []
        for edge in self.edges_list:
            if edge.relation == "contradicts":
                p = self.nodes_by_id[edge.parent_evidence_id]
                c = self.nodes_by_id[edge.child_evidence_id]
                contradictions.append({
                    "parent_node": p,
                    "child_node": c,
                    "transformation": edge.transformation,
                    "agent_id": edge.agent_id,
                })
        return contradictions

    def is_stale(self, evidence_id: UUID, max_age_days: int = 45) -> bool:
        """Checks if an evidence node's timestamp exceeds staleness threshold."""
        if evidence_id not in self.nodes_by_id:
            return False
        node = self.nodes_by_id[evidence_id]
        now = datetime.now(timezone.utc)
        obs = node.observed_at.replace(tzinfo=timezone.utc) if node.observed_at.tzinfo is None else node.observed_at
        age_days = (now - obs).total_seconds() / 86400.0
        return age_days > max_age_days or ("STALE" in node.quality_flags)

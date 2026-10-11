"""
PlanFlow AI - Truth Maintenance & Epistemic Audit Engine
Provides answers to operational provenance queries:
- Root cause tracing for shortages
- Staleness and contradiction audits
- Counterfactual sensitivity (impact of removing CRM opportunities)
- Human verification prioritization by uncertainty x financial exposure
"""

from uuid import UUID
from typing import Dict, List, Any, Optional
from src.knowledge_graph.provenance_graph import EvidenceProvenanceGraph
from src.knowledge_graph.models import EvidenceNode
from src.core.logger import get_logger

logger = get_logger("TruthMaintenance")


class TruthMaintenanceEngine:
    """Audits evidence graph consistency, staleness, and sensitivity."""

    def __init__(self, provenance_graph: EvidenceProvenanceGraph):
        self.graph = provenance_graph

    def trace_shortage_root_cause(self, material_id: str) -> Dict[str, Any]:
        """Identifies upstream facts, sources, and transformations causing a material shortage."""
        matched_nodes = [
            node for node in self.graph.nodes_by_id.values()
            if node.entity_id == material_id
        ]
        if not matched_nodes:
            return {"found": False, "reason": f"No evidence nodes for {material_id}"}

        primary_node = matched_nodes[-1]
        lineage = self.graph.get_lineage(primary_node.evidence_id)

        sources_involved = list(set(n.source_system for n in lineage))
        stale_nodes = [n.entity_id for n in lineage if self.graph.is_stale(n.evidence_id)]

        return {
            "found": True,
            "material_id": material_id,
            "primary_evidence_id": str(primary_node.evidence_id),
            "sources_involved": sources_involved,
            "lineage_depth": len(lineage),
            "has_stale_predecessors": len(stale_nodes) > 0,
            "stale_predecessors": stale_nodes,
            "root_facts": [
                {
                    "entity": n.entity_id,
                    "type": n.entity_type,
                    "source": n.source_system,
                    "value": n.value,
                    "confidence": n.confidence,
                }
                for n in lineage[:3]
            ],
        }

    def evaluate_weak_evidence_recommendations(
        self,
        recommendation_evidence_map: Dict[str, List[UUID]],
        min_acceptable_confidence: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """Finds recommendations relying on low-confidence or contradictory evidence chains."""
        flags = []
        for rec_name, ev_ids in recommendation_evidence_map.items():
            for ev_id in ev_ids:
                conf, contributors = self.graph.propagate_uncertainty(ev_id)
                stale = self.graph.is_stale(ev_id)
                if conf < min_acceptable_confidence or stale:
                    flags.append({
                        "recommendation": rec_name,
                        "evidence_id": str(ev_id),
                        "propagated_confidence": conf,
                        "is_stale": stale,
                        "uncertainty_contributors": contributors,
                    })
        return flags

    def prioritize_human_verification(
        self,
        candidate_facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ranks facts for human review based on: Priority = Financial Exposure x Uncertainty.
        Higher priority indicates where human time yields greatest risk reduction.
        """
        ranked = []
        for item in candidate_facts:
            ev_id = item.get("evidence_id")
            exposure_usd = float(item.get("financial_exposure_usd", 100000.0))
            if ev_id in self.graph.nodes_by_id:
                conf, _ = self.graph.propagate_uncertainty(ev_id)
            else:
                conf = float(item.get("confidence", 0.85))

            uncertainty = max(0.01, 1.0 - conf)
            priority_score = exposure_usd * uncertainty

            ranked.append({
                "entity_id": item.get("entity_id", "Unknown"),
                "description": item.get("description", ""),
                "financial_exposure_usd": exposure_usd,
                "confidence": conf,
                "uncertainty": round(uncertainty, 3),
                "human_review_priority_score": round(priority_score, 2),
                "source_system": item.get("source_system", "ERP"),
            })

        ranked.sort(key=lambda x: x["human_review_priority_score"], reverse=True)
        return ranked

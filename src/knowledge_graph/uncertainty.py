"""
PlanFlow AI - Uncertainty Propagation Engine (HELICASE Framework)
Models epistemic and aleatoric uncertainty over individual facts, multi-step agent
investigation trajectories, and accumulated memory/evidence paths.
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID
import math
import numpy as np
import networkx as nx
from src.knowledge_graph.evidence_graph import TemporalEvidenceGraph
from src.core.logger import get_logger

logger = get_logger("UncertaintyPropagation")


class UncertaintyPropagationEngine:
    """
    Propagates uncertainty through multi-step reasoning trajectories and evidence DAGs.
    Implements HELICASE multi-hop uncertainty aggregation.
    """

    def __init__(self, evidence_graph: Optional[TemporalEvidenceGraph] = None):
        self.evidence_graph = evidence_graph or TemporalEvidenceGraph()

    def propagate_path_uncertainty(
        self,
        evidence_ids: List[UUID],
    ) -> float:
        """
        Computes compound uncertainty along an ordered multi-step reasoning trajectory.
        U_total = 1 - prod_{i=1}^k (1 - u_i), where u_i = 1 - confidence_i + delta_step.
        """
        if not evidence_ids:
            return 0.0

        retention = 1.0
        for eid in evidence_ids:
            node = self.evidence_graph.nodes.get(eid)
            confidence = node.confidence if node else 0.90
            step_uncertainty = max(0.01, min(0.99, 1.0 - confidence + 0.05))
            retention *= (1.0 - step_uncertainty)

        compound_uncertainty = 1.0 - retention
        return round(float(compound_uncertainty), 4)

    def compute_trajectory_risk(
        self,
        target_evidence_id: UUID,
    ) -> Dict[str, float]:
        """
        Traces all directed paths from root observations to target_evidence_id
        and computes maximum, mean, and variance of trajectory uncertainties.
        """
        if target_evidence_id not in self.evidence_graph.nodes:
            return {"max_uncertainty": 0.0, "mean_uncertainty": 0.0, "variance": 0.0}

        roots = self.evidence_graph.get_root_causes(target_evidence_id)
        if not roots:
            node = self.evidence_graph.nodes[target_evidence_id]
            u = 1.0 - node.confidence
            return {"max_uncertainty": round(u, 4), "mean_uncertainty": round(u, 4), "variance": 0.0}

        path_uncertainties = []
        for root in roots:
            try:
                paths = list(nx.all_simple_paths(
                    self.evidence_graph.dag,
                    source=root.evidence_id,
                    target=target_evidence_id,
                ))
                for path in paths:
                    u_p = self.propagate_path_uncertainty(path)
                    path_uncertainties.append(u_p)
            except Exception:
                continue

        if not path_uncertainties:
            path_uncertainties = [0.10]

        max_u = max(path_uncertainties)
        mean_u = sum(path_uncertainties) / len(path_uncertainties)
        var_u = float(np.var(path_uncertainties)) if len(path_uncertainties) > 1 else 0.0

        return {
            "max_uncertainty": round(float(max_u), 4),
            "mean_uncertainty": round(float(mean_u), 4),
            "variance": round(float(var_u), 6),
            "paths_evaluated": len(path_uncertainties),
        }

    def evaluate_material_uncertainty_threshold(
        self,
        uncertainty: float,
        threshold: float = 0.25,
    ) -> Tuple[bool, str]:
        """
        Checks whether material uncertainty exceeds safety limits for the Judge gate.
        """
        if uncertainty > threshold:
            msg = (
                f"HIGH_UNCERTAINTY_EXCEEDED: Uncertainty {uncertainty:.3f} exceeds "
                f"deterministic threshold {threshold:.3f}. Requires Executive sign-off."
            )
            return False, msg
        return True, "Material uncertainty within acceptable tolerance."

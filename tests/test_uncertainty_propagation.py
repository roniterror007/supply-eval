"""
Unit tests for HELICASE Uncertainty Propagation Engine
"""

from uuid import uuid4
from src.knowledge_graph.evidence_graph import TemporalEvidenceGraph
from src.knowledge_graph.uncertainty import UncertaintyPropagationEngine


def test_trajectory_uncertainty_compounding():
    """Validates that multi-hop trajectories compound uncertainty conservatively."""
    graph = TemporalEvidenceGraph()
    uncertainty_engine = UncertaintyPropagationEngine(graph)

    # Chain: root1 -> step2 -> step3
    n1 = graph.add_fact("order", "O1", 100, "ERP", "R1", confidence=0.95)
    n2 = graph.add_fact("forecast", "F1", 120, "manual", "R2", confidence=0.90)
    n3 = graph.add_fact("plan", "P1", 120, "manual", "R3", confidence=0.85)

    graph.add_derivation(n1.evidence_id, n2.evidence_id)
    graph.add_derivation(n2.evidence_id, n3.evidence_id)

    # Propagate along chain
    compound_u = uncertainty_engine.propagate_path_uncertainty([n1.evidence_id, n2.evidence_id, n3.evidence_id])
    assert compound_u > 0.15
    assert compound_u < 1.0

    # Risk metrics for target n3
    risk = uncertainty_engine.compute_trajectory_risk(n3.evidence_id)
    assert risk["max_uncertainty"] > 0.0
    assert risk["paths_evaluated"] == 1


def test_material_uncertainty_threshold():
    """Validates threshold gating on material uncertainty."""
    uncertainty_engine = UncertaintyPropagationEngine()

    passed_low, msg_low = uncertainty_engine.evaluate_material_uncertainty_threshold(0.12, threshold=0.25)
    assert passed_low is True

    passed_high, msg_high = uncertainty_engine.evaluate_material_uncertainty_threshold(0.38, threshold=0.25)
    assert passed_high is False
    assert "HIGH_UNCERTAINTY_EXCEEDED" in msg_high

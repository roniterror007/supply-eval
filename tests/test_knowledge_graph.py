"""
Unit tests for Evidence Provenance Graph and Uncertainty Propagation (HELICASE)
"""

from uuid import uuid4
from datetime import datetime, timedelta, timezone
from src.knowledge_graph.models import EvidenceNode, EvidenceEdge
from src.knowledge_graph.provenance_graph import EvidenceProvenanceGraph
from src.knowledge_graph.truth_maintenance import TruthMaintenanceEngine


def test_evidence_graph_lineage_and_uncertainty():
    """Validates evidence insertion, topological lineage, and propagated uncertainty."""
    graph = EvidenceProvenanceGraph()

    # Raw ERP node
    n_erp = EvidenceNode(
        entity_type="order",
        entity_id="ORDER-101",
        value=5000.0,
        source_system="ERP",
        source_record_id="ERP-001",
        confidence=0.95,
    )
    # Derived forecast node
    n_fc = EvidenceNode(
        entity_type="forecast",
        entity_id="FC-2026-11",
        value=5500.0,
        source_system="manual",
        source_record_id="MOD-001",
        confidence=0.90,
    )
    # Plan node
    n_plan = EvidenceNode(
        entity_type="constraint",
        entity_id="PLAN-001",
        value=5500.0,
        source_system="manual",
        source_record_id="OPT-001",
        confidence=0.88,
    )

    graph.add_node(n_erp)
    graph.add_node(n_fc)
    graph.add_node(n_plan)

    # Add edges
    graph.add_edge(EvidenceEdge(
        parent_evidence_id=n_erp.evidence_id,
        child_evidence_id=n_fc.evidence_id,
        relation="derives",
        transformation="neural_forecast",
        agent_id="ForecastingAgent",
        uncertainty_contribution=0.08,
    ))
    graph.add_edge(EvidenceEdge(
        parent_evidence_id=n_fc.evidence_id,
        child_evidence_id=n_plan.evidence_id,
        relation="supports",
        transformation="consensus_synthesis",
        agent_id="DemandPlanningAgent",
        uncertainty_contribution=0.05,
    ))

    # Lineage check
    lineage = graph.get_lineage(n_plan.evidence_id)
    assert len(lineage) == 3
    assert lineage[0].evidence_id == n_erp.evidence_id
    assert lineage[-1].evidence_id == n_plan.evidence_id

    # Uncertainty propagation
    prop_conf, contributors = graph.propagate_uncertainty(n_plan.evidence_id)
    assert prop_conf <= 0.88
    assert len(contributors) >= 1


def test_truth_maintenance_staleness():
    """Validates that stale nodes are flagged properly."""
    graph = EvidenceProvenanceGraph()
    old_time = datetime.now(timezone.utc) - timedelta(days=60)
    stale_node = EvidenceNode(
        entity_type="forecast",
        entity_id="CRM-STALE-DEAL",
        value=2000000.0,
        source_system="CRM",
        source_record_id="SF-001",
        observed_at=old_time,
        confidence=0.80,
    )
    graph.add_node(stale_node)

    assert graph.is_stale(stale_node.evidence_id, max_age_days=45) is True

"""
Unit tests for HELICASE Temporal Evidence Graph and Provenance DAG
"""

from datetime import datetime, timezone, timedelta
from uuid import uuid4
import pytest
from src.knowledge_graph.evidence_graph import TemporalEvidenceGraph
from src.knowledge_graph.source_quality import SourceQualityEngine
from src.knowledge_graph.lineage import LineageTracer


def test_evidence_graph_temporal_dag_and_cycle_prevention():
    """Validates that DAG allows valid derivations and prevents directed cycles."""
    graph = TemporalEvidenceGraph()
    now = datetime.now(timezone.utc)

    # 1. Add root observations
    root_erp = graph.add_fact(
        entity_type="order",
        entity_id="ORD-9901",
        value={"units": 5000},
        source_system="ERP",
        source_record_id="SAP-DOC-1234",
        observed_at=now - timedelta(hours=10),
    )

    root_mes = graph.add_fact(
        entity_type="factory",
        entity_id="Factory A (Stuttgart)",
        value={"available_capacity": 12000},
        source_system="MES",
        source_record_id="MES-LINE-01",
        observed_at=now - timedelta(hours=5),
    )

    # 2. Add derived forecast
    derived_fc = graph.add_fact(
        entity_type="forecast",
        entity_id="FC-2026-Q4",
        value={"point_forecast": 5200},
        source_system="manual",
        source_record_id="SURROGATE-NET",
        observed_at=now - timedelta(hours=2),
    )

    graph.add_derivation(root_erp.evidence_id, derived_fc.evidence_id, relation="derives")
    graph.add_derivation(root_mes.evidence_id, derived_fc.evidence_id, relation="supports")

    # 3. Verify lineage
    ancestors = graph.get_ancestors(derived_fc.evidence_id)
    assert len(ancestors) == 2
    roots = graph.get_root_causes(derived_fc.evidence_id)
    assert len(roots) == 2

    # 4. Cycle prevention: deriving root_erp from derived_fc must fail
    with pytest.raises(ValueError, match="cycle"):
        graph.add_derivation(derived_fc.evidence_id, root_erp.evidence_id, relation="derives")


def test_temporal_view_replay():
    """Validates point-in-time replay filtering based on observed_at and valid_to."""
    graph = TemporalEvidenceGraph()
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)

    # Fact observed at t0, valid indefinitely
    f1 = graph.add_fact("order", "O1", 100, "ERP", "REC1", observed_at=t0)
    # Fact observed at t2 (in future of t1)
    f2 = graph.add_fact("order", "O2", 200, "ERP", "REC2", observed_at=t2)
    # Fact observed at t0, but expired at t0 + 1 hour
    f3 = graph.add_fact("order", "O3", 300, "ERP", "REC3", observed_at=t0, valid_to=t0 + timedelta(hours=1))

    replay_graph = graph.get_temporal_view(as_of=t1)

    assert f1.evidence_id in replay_graph.nodes
    assert f2.evidence_id not in replay_graph.nodes  # future
    assert f3.evidence_id not in replay_graph.nodes  # expired


def test_source_quality_and_staleness_check():
    """Validates exponential freshness decay and CRM staleness policy blocking."""
    quality_engine = SourceQualityEngine()
    now = datetime.now(timezone.utc)

    # Fresh ERP fact (1 hour old)
    fresh_ref = graph_ref = TemporalEvidenceGraph().add_fact(
        "order", "O1", 100, "ERP", "R1", observed_at=now - timedelta(hours=1)
    )
    ref_fresh = TemporalEvidenceGraph().to_evidence_reference(fresh_ref)
    score_fresh = quality_engine.evaluate_evidence(ref_fresh)
    assert score_fresh > 0.85

    # Stale CRM fact (100 hours old)
    stale_crm = TemporalEvidenceGraph().add_fact(
        "order", "DEAL-99", 5000, "CRM", "OPP-99", observed_at=now - timedelta(hours=100)
    )
    ref_stale = TemporalEvidenceGraph().to_evidence_reference(stale_crm)
    is_blocked, reason = quality_engine.check_staleness_block(ref_stale, max_allowed_hours=72.0)
    assert is_blocked is True
    assert "STALENESS_VIOLATION" in reason


def test_independent_corroboration_detection():
    """Ensures multiple claims from the same source are flagged as non-independent."""
    quality_engine = SourceQualityEngine()
    g = TemporalEvidenceGraph()
    now = datetime.now(timezone.utc)

    n1 = g.add_fact("forecast", "F1", 1000, "CRM", "REC-001", observed_at=now)
    n2 = g.add_fact("forecast", "F1", 1000, "CRM", "REC-001", observed_at=now)
    n3 = g.add_fact("forecast", "F1", 1000, "ERP", "SAP-888", observed_at=now)

    r1 = g.to_evidence_reference(n1)
    r2 = g.to_evidence_reference(n2)
    r3 = g.to_evidence_reference(n3)

    # Case A: Duplicate source records (r1 and r2)
    is_indep_dup, count_dup, duplicates = quality_engine.verify_independent_corroboration([r1, r2])
    assert is_indep_dup is False
    assert len(duplicates) > 0

    # Case B: Distinct sources (r1 and r3)
    is_indep_clean, count_clean, duplicates_clean = quality_engine.verify_independent_corroboration([r1, r3])
    assert is_indep_clean is True
    assert count_clean == 2

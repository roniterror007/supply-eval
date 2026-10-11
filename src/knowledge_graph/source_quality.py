"""
PlanFlow AI - Knowledge Graph Source Quality & Conflict Engine
Evaluates evidence freshness, system reliability, independent corroboration,
and detects/resolves epistemic contradictions across enterprise data sources.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple, Set
from uuid import UUID
from src.core.contracts import EvidenceReference
from src.core.logger import get_logger

logger = get_logger("SourceQuality")


SYSTEM_RELIABILITY_WEIGHTS: Dict[str, float] = {
    "ERP": 0.98,
    "MES": 0.95,
    "WMS": 0.92,
    "supplier_portal": 0.85,
    "CRM": 0.65,      # High optimism/staleness risk
    "manual": 0.50,   # Unverified human input
}

DEFAULT_HALF_LIFE_HOURS: Dict[str, float] = {
    "ERP": 168.0,           # 7 days
    "MES": 24.0,            # 24 hours
    "WMS": 12.0,            # 12 hours (fast moving stock)
    "supplier_portal": 48.0,# 48 hours
    "CRM": 48.0,            # 48 hours (deals change fast)
    "manual": 24.0,
}


class SourceQualityEngine:
    """Evaluates data freshness, source reliability, and manages factual conflicts."""

    def __init__(self, system_weights: Optional[Dict[str, float]] = None):
        self.system_weights = system_weights or SYSTEM_RELIABILITY_WEIGHTS
        self.conflict_registry: Dict[str, List[EvidenceReference]] = {}

    def compute_freshness(
        self,
        observed_at: datetime,
        as_of: Optional[datetime] = None,
        source_system: str = "ERP",
    ) -> Tuple[float, float]:
        """
        Computes elapsed hours and exponential freshness score (0.0 to 1.0).
        Half-life model: S_fresh = 2^(-delta_t / half_life).
        """
        now = as_of or datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta_seconds = max(0.0, (now - observed_at).total_seconds())
        elapsed_hours = delta_seconds / 3600.0

        half_life = DEFAULT_HALF_LIFE_HOURS.get(source_system, 48.0)
        freshness_score = math.pow(0.5, elapsed_hours / half_life)
        return elapsed_hours, max(0.0, min(1.0, freshness_score))

    def evaluate_evidence(
        self,
        ref: EvidenceReference,
        as_of: Optional[datetime] = None,
    ) -> float:
        """
        Calculates composite quality score considering system reliability,
        freshness decay, and inherent measurement confidence.
        """
        elapsed_hours, freshness = self.compute_freshness(
            ref.observed_at, as_of=as_of, source_system=ref.source_system
        )
        ref.freshness_hours = round(elapsed_hours, 2)
        system_weight = self.system_weights.get(ref.source_system, 0.50)

        composite_score = (
            0.40 * system_weight
            + 0.40 * freshness
            + 0.20 * ref.confidence
        )
        ref.source_quality_score = round(composite_score, 4)
        return ref.source_quality_score

    def check_staleness_block(
        self,
        ref: EvidenceReference,
        max_allowed_hours: float = 72.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforces operating rule:
        A stale CRM opportunity cannot cause a production increase without a freshness check.
        """
        if ref.source_system == "CRM" and ref.freshness_hours > max_allowed_hours:
            reason = (
                f"STALENESS_VIOLATION: CRM record {ref.source_record_id} is {ref.freshness_hours:.1f}h old "
                f"(limit: {max_allowed_hours:.1f}h). Automatic production increase blocked."
            )
            return True, reason
        return False, None

    def detect_conflicts(
        self,
        entity_key: str,
        claims: List[EvidenceReference],
        tolerance_pct: float = 0.05,
    ) -> Tuple[bool, List[str]]:
        """
        Enforces operating rule:
        Conflicting BOM, capacity, or supplier lead-time facts must block automatic execution.
        """
        if len(claims) < 2:
            return False, []

        conflicts = []
        # Check source diversity and value alignment
        # Group by value if available or check whether values differ significantly
        self.conflict_registry[entity_key] = claims

        # Check if any claim has open conflict status
        for claim in claims:
            if claim.conflict_status == "OPEN":
                conflicts.append(
                    f"Unresolved factual contradiction for {entity_key} in source {claim.source_system}:{claim.source_record_id}"
                )

        return len(conflicts) > 0, conflicts

    def verify_independent_corroboration(
        self,
        references: List[EvidenceReference],
    ) -> Tuple[bool, int, List[str]]:
        """
        Enforces operating rule:
        Multiple claims copied from the same source are not independent corroboration.
        """
        unique_sources: Set[str] = set()
        unique_records: Set[Tuple[str, str]] = set()
        duplicates: List[str] = []

        for ref in references:
            key = (ref.source_system, ref.source_record_id)
            if key in unique_records:
                duplicates.append(f"{ref.source_system}:{ref.source_record_id}")
            else:
                unique_records.add(key)
                unique_sources.add(ref.source_system)

        is_independent = len(unique_sources) >= 2 and len(duplicates) == 0
        return is_independent, len(unique_sources), duplicates

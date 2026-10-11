"""
Unit tests for PlanFlow AI Pydantic Contracts and Invariant Enforcement
"""

import pytest
from pydantic import ValidationError
from src.core.contracts import (
    ForecastContract,
    ConsensusDemandContract,
    FactoryAllocationContract,
    RiskLevel,
    ActionRecommendation,
    RecommendationPlanContract,
)


def test_forecast_contract_quantile_monotonicity():
    """Validates that ForecastContract enforces p10 <= p50 <= p90."""
    # Valid contract
    fc = ForecastContract(
        product_line="Automotive Solutions",
        region="Germany",
        horizon_month="2026-11",
        point_forecast_units=10000.0,
        p10_units=8500.0,
        p50_units=10000.0,
        p90_units=11500.0,
        baseline_revenue_usd=12000000.0,
        confidence_score=0.92,
        primary_drivers=["Macro Trend", "Backlog"],
    )
    assert fc.p10_units <= fc.p50_units <= fc.p90_units

    # Violating monotonicity (p10 > p50) should raise ValidationError
    with pytest.raises(ValidationError):
        ForecastContract(
            product_line="Automotive Solutions",
            region="Germany",
            horizon_month="2026-11",
            point_forecast_units=10000.0,
            p10_units=10500.0,  # Invalid: p10 > p50
            p50_units=10000.0,
            p90_units=11500.0,
            baseline_revenue_usd=12000000.0,
            confidence_score=0.92,
            primary_drivers=["Macro Trend"],
        )


def test_factory_allocation_conservation():
    """Validates that FactoryAllocationContract enforces unit conservation."""
    # Valid conservation
    fac = FactoryAllocationContract(
        total_demand_units=15000.0,
        factory_allocations={"Factory A": 12000.0, "Factory D": 3000.0},
        factory_capacities={"Factory A": 12000.0, "Factory D": 14000.0},
        factory_utilizations={"Factory A": 100.0, "Factory D": 21.4},
        overloaded_factories=["Factory A"],
        transfer_shifts=[],
        confidence_score=0.95,
    )
    assert sum(fac.factory_allocations.values()) == fac.total_demand_units

    # Discrepancy > 1% should fail
    with pytest.raises(ValidationError):
        FactoryAllocationContract(
            total_demand_units=15000.0,
            factory_allocations={"Factory A": 10000.0, "Factory D": 2000.0},  # Sum = 12000 != 15000
            factory_capacities={"Factory A": 12000.0, "Factory D": 14000.0},
            factory_utilizations={"Factory A": 83.3, "Factory D": 14.3},
            overloaded_factories=[],
            transfer_shifts=[],
            confidence_score=0.95,
        )


def test_action_recommendation_roi():
    """Validates ActionRecommendation ROI calculation."""
    rec = ActionRecommendation(
        priority_rank=1,
        action_type="EXPEDITE_SUPPLIER",
        description="Air freight expedite",
        target_entity="MAT-M1",
        cost_usd=50000.0,
        protected_revenue_usd=500000.0,
        risk_mitigation_score=0.90,
    )
    assert rec.net_roi_multiplier == 10.0

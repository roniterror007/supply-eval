"""
PlanFlow AI - Discrete-Event Supply Chain Flow Simulator
Models day-by-day factory operations, component arrivals, WIP assembly queues,
daily capacity constraints, and customer ship commitments over a 30-day cycle.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DailySimulationTrace:
    day: int
    demand_received: float
    production_completed: float
    material_stock_end: float
    backlog_queue: float
    orders_shipped: float
    stockout_occurred: bool


@dataclass
class SimulationOutcome:
    cycle_days: int
    total_demand: float
    total_shipped: float
    fill_rate_pct: float
    stockout_days: int
    recovery_time_days: int
    final_backlog: float
    daily_traces: List[DailySimulationTrace] = field(default_factory=list)


class DiscreteEventSupplyChainSimulator:
    """Simulates daily execution dynamics of manufacturing and supply delivery."""

    def __init__(self, cycle_days: int = 30):
        self.cycle_days = cycle_days

    def simulate(
        self,
        monthly_demand: float,
        daily_capacity: float,
        initial_material_stock: float,
        material_ratio: float,
        material_in_transit: float,
        supplier_lead_time_days: int,
        daily_demand_volatility: float = 0.15,
    ) -> SimulationOutcome:
        """Runs the 30-day step-by-step discrete-event simulation."""
        # Distribute monthly demand across days with daily volatility
        np.random.seed(42)
        base_daily_demand = monthly_demand / self.cycle_days
        daily_demands = np.maximum(
            0.0,
            np.random.normal(base_daily_demand, base_daily_demand * daily_demand_volatility, size=self.cycle_days)
        )
        # Normalize sum to match target
        daily_demands = daily_demands * (monthly_demand / np.sum(daily_demands))

        mat_stock = initial_material_stock
        backlog = 0.0
        total_shipped = 0.0
        stockout_days = 0
        recovery_day = 0
        traces = []
        had_stockout = False

        for day in range(1, self.cycle_days + 1):
            # 1. Supplier Delivery Event
            if day == supplier_lead_time_days:
                mat_stock += material_in_transit

            # 2. Demand Arrival Event
            new_demand = float(daily_demands[day - 1])
            backlog += new_demand

            # 3. Production Event
            # Maximum units buildable given available materials and line capacity
            mat_buildable = mat_stock / material_ratio if material_ratio > 0 else backlog
            possible_prod = min(backlog, daily_capacity, mat_buildable)
            prod_completed = max(0.0, float(possible_prod))

            # Deduct materials and backlog
            mat_used = prod_completed * material_ratio
            mat_stock -= mat_used
            backlog -= prod_completed
            total_shipped += prod_completed

            is_stockout = (mat_stock <= 0.1) and (backlog > 0)
            if is_stockout:
                stockout_days += 1
                had_stockout = True
            elif had_stockout and recovery_day == 0 and backlog <= daily_capacity:
                recovery_day = day

            traces.append(DailySimulationTrace(
                day=day,
                demand_received=round(new_demand, 1),
                production_completed=round(prod_completed, 1),
                material_stock_end=round(max(0.0, mat_stock), 1),
                backlog_queue=round(backlog, 1),
                orders_shipped=round(prod_completed, 1),
                stockout_occurred=is_stockout,
            ))

        fill_rate = (total_shipped / (monthly_demand + 1e-6)) * 100.0
        recovery_time = recovery_day if recovery_day > 0 else (stockout_days if had_stockout else 0)

        return SimulationOutcome(
            cycle_days=self.cycle_days,
            total_demand=round(monthly_demand, 1),
            total_shipped=round(total_shipped, 1),
            fill_rate_pct=round(min(100.0, fill_rate), 2),
            stockout_days=stockout_days,
            recovery_time_days=recovery_time,
            final_backlog=round(backlog, 1),
            daily_traces=traces,
        )

"""
PlanFlow AI - Master Operations Research Planner (MILP)
Uses Mixed-Integer Linear Programming (HiGHS solver via scipy.optimize.milp)
to select globally optimal, feasible production allocations, overtime schedules,
inter-plant transfers, and supplier expedites under explicit physical constraints.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
import scipy.optimize as opt
from scipy.optimize import LinearConstraint, Bounds
from src.core.logger import get_logger
import time
import hashlib
from src.core.contracts import OptimizationCertificateContract
from src.optimization.objectives import PlanOptimizationObjectives, ObjectiveWeights
from src.core.cost_engine import get_cost_engine

logger = get_logger("MasterMILPPlanner")


@dataclass
class OptimizationResult:
    status: str
    success: bool
    objective_value: float
    production_allocations: Dict[str, float]
    overtime_hours: Dict[str, float]
    transfer_shifts: List[Dict[str, Any]]
    expedite_units: Dict[str, float]
    unmet_demand_units: float
    total_fulfilled_units: float
    total_cost_usd: float
    total_protected_revenue_usd: float
    carbon_emissions_kg: float
    emergency_sourcing_units: float = 0.0
    unmet_demand_without_emergency_units: float = 0.0
    residual_unmet_demand_units: float = 0.0
    full_service_achieved: bool = True
    certificate: Optional[OptimizationCertificateContract] = None
    shadow_prices: Dict[str, float] = field(default_factory=dict)
    slack_details: Dict[str, float] = field(default_factory=dict)


class MasterMILPPlanner:
    """Solves the multi-plant, multi-material enterprise master planning problem."""

    def __init__(self, weights: ObjectiveWeights = ObjectiveWeights()):
        self.weights = weights
        self.objectives = PlanOptimizationObjectives(weights)

    def solve(
        self,
        target_demand: float,
        unit_price: float,
        primary_factory: str,
        primary_capacity: float,
        secondary_factory: str,
        secondary_capacity: float,
        bom_ratios: Any = None,
        usable_stocks: Any = None,
        max_overtime_hours: float = 120.0,
        units_per_overtime_hour: float = 6.0,
        allow_emergency_sourcing: bool = False,
        emergency_spot_cap: float = 0.0,
        emergency_spot_unit_cost: float = 105.0,
        spot_lead_days: int = 3,
        planning_horizon_days: int = 30,
        is_material_compatible: bool = True,
        max_standard_expedite_units: Optional[float] = None,
        material_ratio: Optional[float] = None,
        usable_material_stock: Optional[float] = None,
        **kwargs,
    ) -> OptimizationResult:
        # Backward compatibility for single material callers
        if bom_ratios is None:
            if material_ratio is not None:
                bom_ratios = {"MAT-A1": float(material_ratio)}
            else:
                bom_ratios = {"MAT-A1": 1.0}
        elif isinstance(bom_ratios, (int, float)):
            bom_ratios = {"MAT-A1": float(bom_ratios)}

        if usable_stocks is None:
            if usable_material_stock is not None:
                usable_stocks = {"MAT-A1": float(usable_material_stock)}
            else:
                usable_stocks = {"MAT-A1": target_demand}
        elif isinstance(usable_stocks, (int, float)):
            usable_stocks = {"MAT-A1": float(usable_stocks)}

        cost_engine = get_cost_engine()
        # Variables for N materials
        mat_keys = list(bom_ratios.keys())
        N = len(mat_keys)
        
        # x[0]: Primary production (units)
        # x[1]: Secondary production (units)
        # x[2]: Overtime hours at Primary Factory (hours)
        # x[3]: Inter-factory transfer volume (units)
        # x[4 .. 3+N]: Standard expedite volume for each material (units, integer)
        # x[4+N]: Unmet customer demand slack (units)
        # x[5+N .. 4+2N]: Emergency spot procurement for each material (units)
        
        num_vars = 5 + 2 * N
        
        c = np.zeros(num_vars)
        c[0] = cost_engine.get_production_cost(primary_factory)
        c[1] = cost_engine.get_production_cost(secondary_factory)
        c[2] = cost_engine.get_overtime_cost(primary_factory)
        c[3] = cost_engine.get_transfer_cost(primary_factory, secondary_factory) + (0.15 * cost_engine.get_carbon_cost_per_kg())
        
        for i, mat in enumerate(mat_keys):
            c[4 + i] = cost_engine.get_expedite_cost(mat) + (2.5 * cost_engine.get_carbon_cost_per_kg())
            
        c[4 + N] = unit_price * cost_engine.get_sla_penalty_multiplier()
        
        for i in range(N):
            c[5 + N + i] = emergency_spot_unit_cost

        # Bounds: l <= x <= u
        can_use_spot = allow_emergency_sourcing and is_material_compatible and (spot_lead_days <= planning_horizon_days)
        spot_cap = emergency_spot_cap if can_use_spot else 0.0
        std_expedite_cap = max_standard_expedite_units if max_standard_expedite_units is not None else target_demand * 10.0

        lb = np.zeros(num_vars)
        ub = np.zeros(num_vars)
        
        ub[0] = primary_capacity + (max_overtime_hours * units_per_overtime_hour)
        ub[1] = secondary_capacity
        ub[2] = max_overtime_hours
        ub[3] = target_demand
        for i in range(N):
            ub[4 + i] = std_expedite_cap
        ub[4 + N] = target_demand
        for i in range(N):
            ub[5 + N + i] = spot_cap
            
        bounds = Bounds(lb, ub)

        # Integrality
        integrality = np.zeros(num_vars)
        for i in range(N):
            integrality[4 + i] = 1

        # Linear Constraints
        # 4 base rows + N material rows
        num_constraints = 4 + N
        A = np.zeros((num_constraints, num_vars))
        
        # Row 0: Demand Balance
        A[0, 0] = 1.0; A[0, 1] = 1.0; A[0, 4 + N] = 1.0
        # Row 1: Primary Capacity
        A[1, 0] = 1.0; A[1, 2] = -units_per_overtime_hour
        # Row 2: Secondary Capacity
        A[2, 1] = 1.0
        # Row 3: Transfer Balance
        A[3, 1] = 1.0; A[3, 3] = -1.0
        
        # Rows 4 to 3+N: Material Balances
        for i, mat in enumerate(mat_keys):
            ratio = bom_ratios[mat]
            A[4 + i, 0] = ratio
            A[4 + i, 1] = ratio
            A[4 + i, 4 + i] = -1.0
            A[4 + i, 5 + N + i] = -1.0

        constraint_lb = np.zeros(num_constraints)
        constraint_lb[0] = target_demand
        constraint_lb[1] = -np.inf
        constraint_lb[2] = -np.inf
        constraint_lb[3] = 0.0
        for i in range(N):
            constraint_lb[4 + i] = -np.inf

        constraint_ub = np.zeros(num_constraints)
        constraint_ub[0] = target_demand
        constraint_ub[1] = primary_capacity
        constraint_ub[2] = secondary_capacity
        constraint_ub[3] = 0.0
        for i, mat in enumerate(mat_keys):
            constraint_ub[4 + i] = usable_stocks[mat]

        # Generate cryptographic constraint fingerprint
        constraint_bytes = A.tobytes() + constraint_lb.tobytes() + constraint_ub.tobytes() + c.tobytes()
        constraint_set_hash = f"SHA256:{hashlib.sha256(constraint_bytes).hexdigest()[:16]}"

        constraints = LinearConstraint(A, constraint_lb, constraint_ub)

        # Solve with SciPy's HiGHS solver
        start_time = time.perf_counter()
        res = opt.milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        runtime_seconds = round(time.perf_counter() - start_time, 4)

        if not res.success:
            logger.warning(f"[yellow]MILP solver status: {res.status} - {res.message}[/yellow]")
            status_code = "INFEASIBLE" if res.status == 2 else ("TIME_LIMIT" if res.status == 1 else "ERROR")
            cert = OptimizationCertificateContract(
                solve_status=status_code,
                objective_value=None,
                best_bound=None,
                mip_gap=None,
                runtime_seconds=runtime_seconds,
                candidate_space_version="v2.0-spord",
                scenario_set_version="v2.0-enterprise",
                constraint_set_hash=constraint_set_hash,
            )
            return OptimizationResult(
                status=status_code,
                success=False,
                objective_value=float("inf"),
                production_allocations={},
                overtime_hours={},
                transfer_shifts=[],
                expedite_units={},
                unmet_demand_units=target_demand,
                total_fulfilled_units=0.0,
                total_cost_usd=0.0,
                total_protected_revenue_usd=0.0,
                carbon_emissions_kg=0.0,
                certificate=cert,
            )

        x = res.x
        prim_prod = round(float(x[0]), 1)
        sec_prod = round(float(x[1]), 1)
        ot_hours = round(float(x[2]), 1)
        trans_u = round(float(x[3]), 1)
        
        exp_dict = {}
        spot_dict = {}
        for i, mat in enumerate(mat_keys):
            e = round(float(x[4 + i]), 1)
            if e > 0: exp_dict[mat] = e
            s = round(float(x[5 + N + i]), 1)
            if s > 0: spot_dict[mat] = s
            
        unmet_u = round(float(x[4 + N]), 1)
        total_spot_u = sum(spot_dict.values())

        fulfilled_u = prim_prod + sec_prod
        protected_rev = fulfilled_u * unit_price

        # Unmet demand without emergency actions vs residual unmet demand
        # (Simplified estimation of spot impact on unmet demand)
        unmet_without_spot = round(unmet_u + total_spot_u, 1)
        full_service = (unmet_u <= 1.0)

        # Cost breakdown
        prod_allocs = {primary_factory: prim_prod, secondary_factory: sec_prod}
        ot_dict = {primary_factory: ot_hours}
        trans_dict = {f"{primary_factory} -> {secondary_factory}": trans_u} if trans_u > 0 else {}

        cost_breakdown = self.objectives.compute_direct_cost(
            production_allocations=prod_allocs,
            overtime_hours=ot_dict,
            transfer_units=trans_dict,
            expedite_units=exp_dict,
            unmet_demand_units=unmet_u,
            unit_price=unit_price,
        )
        # Add emergency spot procurement cost if authorized
        if total_spot_u > 0:
            cost_breakdown["emergency_spot_cost"] = round(total_spot_u * emergency_spot_unit_cost, 2)
            cost_breakdown["total_direct_cost"] += cost_breakdown["emergency_spot_cost"]

        carbon_kg = self.objectives.compute_carbon_emissions_kg(trans_dict, exp_dict)

        shifts_list = []
        if trans_u > 0:
            shifts_list.append({
                "from_factory": primary_factory,
                "to_factory": secondary_factory,
                "units_shifted": trans_u,
                "transfer_cost_usd": round(trans_u * 25.0, 2),
            })

        logger.info(
            f"[bold green]MILP Optimal Solution Found[/bold green]: Fulfilled: {fulfilled_u:,.0f}u / {target_demand:,.0f}u | "
            f"Cost: ${cost_breakdown['total_direct_cost']:,.0f} | Expedite: {sum(exp_dict.values()):,.0f}u | Spot: {total_spot_u:,.0f}u"
        )

        gap = getattr(res, "mip_gap", None)
        solve_status = "OPTIMAL" if (gap is None or gap < 1e-4) else "FEASIBLE_GAP"

        cert = OptimizationCertificateContract(
            solve_status=solve_status,
            objective_value=round(float(res.fun), 2),
            best_bound=round(float(getattr(res, "best_bound", res.fun)), 2),
            mip_gap=round(float(gap), 4) if gap is not None else 0.0,
            runtime_seconds=runtime_seconds,
            candidate_space_version="v2.0-spord",
            scenario_set_version="v2.0-enterprise",
            constraint_set_hash=constraint_set_hash,
        )

        return OptimizationResult(
            status=solve_status,
            success=True,
            objective_value=round(float(res.fun), 2),
            production_allocations=prod_allocs,
            overtime_hours=ot_dict,
            transfer_shifts=shifts_list,
            expedite_units=exp_dict,
            unmet_demand_units=unmet_u,
            total_fulfilled_units=fulfilled_u,
            total_cost_usd=cost_breakdown["total_direct_cost"],
            total_protected_revenue_usd=round(protected_rev, 2),
            carbon_emissions_kg=carbon_kg,
            emergency_sourcing_units=total_spot_u,
            unmet_demand_without_emergency_units=unmet_without_spot,
            residual_unmet_demand_units=unmet_u,
            full_service_achieved=full_service,
            certificate=cert,
            slack_details={
                "demand_unmet_slack": unmet_u,
                "primary_capacity_headroom": round(primary_capacity - prim_prod, 1),
                "emergency_sourcing_used": total_spot_u,
            },
        )

    def solve_network(
        self,
        target_demand: float,
        unit_price: float,
        factory_capacities: Dict[str, float],
        bom_ratios: Any = None,
        usable_stocks: Any = None,
        max_overtime_hours: float = 120.0,
        units_per_overtime_hour: float = 6.0,
        allow_emergency_sourcing: bool = False,
        emergency_spot_cap: float = 0.0,
        emergency_spot_unit_cost: float = 105.0,
        spot_lead_days: int = 3,
        planning_horizon_days: int = 30,
        is_material_compatible: bool = True,
        max_standard_expedite_units: Optional[float] = None,
        material_ratio: Optional[float] = None,
        usable_material_stock: Optional[float] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Solves multi-site network allocation across N factories simultaneously.
        """
        factories = list(factory_capacities.keys())
        if len(factories) <= 2:
            p_f = factories[0] if len(factories) > 0 else "Factory A"
            s_f = factories[1] if len(factories) > 1 else "Factory B"
            return self.solve(
                target_demand=target_demand,
                unit_price=unit_price,
                primary_factory=p_f,
                primary_capacity=factory_capacities.get(p_f, 12000.0),
                secondary_factory=s_f,
                secondary_capacity=factory_capacities.get(s_f, 10000.0),
                bom_ratios=bom_ratios,
                usable_stocks=usable_stocks,
                max_overtime_hours=max_overtime_hours,
                units_per_overtime_hour=units_per_overtime_hour,
                allow_emergency_sourcing=allow_emergency_sourcing,
                emergency_spot_cap=emergency_spot_cap,
                emergency_spot_unit_cost=emergency_spot_unit_cost,
                spot_lead_days=spot_lead_days,
                planning_horizon_days=planning_horizon_days,
                is_material_compatible=is_material_compatible,
                max_standard_expedite_units=max_standard_expedite_units,
                material_ratio=material_ratio,
                usable_material_stock=usable_material_stock,
            )

        K = len(factories)
        if bom_ratios is None:
            if material_ratio is not None:
                bom_ratios = {"MAT-A1": float(material_ratio)}
            else:
                bom_ratios = {"MAT-A1": 1.0}
        elif isinstance(bom_ratios, (int, float)):
            bom_ratios = {"MAT-A1": float(bom_ratios)}

        if usable_stocks is None:
            if usable_material_stock is not None:
                usable_stocks = {"MAT-A1": float(usable_material_stock)}
            else:
                usable_stocks = {"MAT-A1": target_demand}
        elif isinstance(usable_stocks, (int, float)):
            usable_stocks = {"MAT-A1": float(usable_stocks)}

        mat_keys = list(bom_ratios.keys())
        N = len(mat_keys)
        cost_engine = get_cost_engine()

        num_vars = 2 * K + 2 * N + 1
        c = np.zeros(num_vars)
        for k, f_name in enumerate(factories):
            c[k] = cost_engine.get_production_cost(f_name)
            c[K + k] = cost_engine.get_overtime_cost(f_name)

        for m_idx, mat in enumerate(mat_keys):
            c[2 * K + m_idx] = cost_engine.get_expedite_cost(mat) + (2.5 * cost_engine.get_carbon_cost_per_kg())

        c[2 * K + N] = unit_price * cost_engine.get_sla_penalty_multiplier()

        for m_idx in range(N):
            c[2 * K + N + 1 + m_idx] = emergency_spot_unit_cost

        can_use_spot = allow_emergency_sourcing and is_material_compatible and (spot_lead_days <= planning_horizon_days)
        spot_cap = emergency_spot_cap if can_use_spot else 0.0
        std_expedite_cap = max_standard_expedite_units if max_standard_expedite_units is not None else target_demand * 10.0

        lb = np.zeros(num_vars)
        ub = np.zeros(num_vars)
        for k, f_name in enumerate(factories):
            ub[k] = factory_capacities[f_name] + (max_overtime_hours * units_per_overtime_hour)
            ub[K + k] = max_overtime_hours

        for m_idx in range(N):
            ub[2 * K + m_idx] = std_expedite_cap

        ub[2 * K + N] = target_demand

        for m_idx in range(N):
            ub[2 * K + N + 1 + m_idx] = spot_cap

        bounds = Bounds(lb, ub)
        integrality = np.zeros(num_vars)
        for m_idx in range(N):
            integrality[2 * K + m_idx] = 1

        num_constraints = 1 + K + N
        A = np.zeros((num_constraints, num_vars))
        constraint_lb = np.zeros(num_constraints)
        constraint_ub = np.zeros(num_constraints)

        # Row 0: Demand balance
        for k in range(K):
            A[0, k] = 1.0
        A[0, 2 * K + N] = 1.0
        constraint_lb[0] = target_demand
        constraint_ub[0] = target_demand

        # Factory capacity constraints
        for k, f_name in enumerate(factories):
            row_idx = 1 + k
            A[row_idx, k] = 1.0
            A[row_idx, K + k] = -units_per_overtime_hour
            constraint_lb[row_idx] = -np.inf
            constraint_ub[row_idx] = factory_capacities[f_name]

        # Material balance constraints
        for m_idx, mat in enumerate(mat_keys):
            row_idx = 1 + K + m_idx
            ratio = bom_ratios[mat]
            for k in range(K):
                A[row_idx, k] = ratio
            A[row_idx, 2 * K + m_idx] = -1.0
            A[row_idx, 2 * K + N + 1 + m_idx] = -1.0
            constraint_lb[row_idx] = -np.inf
            constraint_ub[row_idx] = usable_stocks[mat]

        constraint_bytes = A.tobytes() + constraint_lb.tobytes() + constraint_ub.tobytes() + c.tobytes()
        constraint_set_hash = f"SHA256:{hashlib.sha256(constraint_bytes).hexdigest()[:16]}"
        constraints = LinearConstraint(A, constraint_lb, constraint_ub)

        start_time = time.perf_counter()
        res = opt.milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        runtime_seconds = round(time.perf_counter() - start_time, 4)

        if not res.success:
            cert = OptimizationCertificateContract(
                solve_status="INFEASIBLE", objective_value=None, best_bound=None, mip_gap=None,
                runtime_seconds=runtime_seconds, candidate_space_version="v2.0-spord",
                scenario_set_version="v2.0-enterprise", constraint_set_hash=constraint_set_hash,
            )
            return OptimizationResult(
                status="INFEASIBLE", success=False, objective_value=float("inf"),
                production_allocations={}, overtime_hours={}, transfer_shifts=[], expedite_units={},
                unmet_demand_units=target_demand, total_fulfilled_units=0.0, total_cost_usd=0.0,
                total_protected_revenue_usd=0.0, carbon_emissions_kg=0.0, certificate=cert,
            )

        x = res.x
        prod_allocs = {factories[k]: round(float(x[k]), 1) for k in range(K)}
        ot_dict = {factories[k]: round(float(x[K + k]), 1) for k in range(K)}
        exp_dict = {mat_keys[m]: round(float(x[2 * K + m]), 1) for m in range(N) if x[2 * K + m] > 0.05}
        unmet_u = round(float(x[2 * K + N]), 1)
        spot_dict = {mat_keys[m]: round(float(x[2 * K + N + 1 + m]), 1) for m in range(N) if x[2 * K + N + 1 + m] > 0.05}
        total_spot_u = sum(spot_dict.values())
        fulfilled_u = sum(prod_allocs.values())

        trans_dict = {}
        shifts_list = []
        primary_factory = factories[0]
        for f_name, alloc in prod_allocs.items():
            if f_name != primary_factory and alloc > 0:
                cost_trans = cost_engine.get_transfer_cost(primary_factory, f_name)
                trans_dict[f"{primary_factory} -> {f_name}"] = alloc
                shifts_list.append({
                    "from_factory": primary_factory,
                    "to_factory": f_name,
                    "units_shifted": alloc,
                    "transfer_cost_usd": round(alloc * cost_trans, 2),
                })

        cost_breakdown = self.objectives.compute_direct_cost(
            production_allocations=prod_allocs,
            overtime_hours=ot_dict,
            transfer_units=trans_dict,
            expedite_units=exp_dict,
            unmet_demand_units=unmet_u,
            unit_price=unit_price,
        )
        if total_spot_u > 0:
            cost_breakdown["emergency_spot_cost"] = round(total_spot_u * emergency_spot_unit_cost, 2)
            cost_breakdown["total_direct_cost"] += cost_breakdown["emergency_spot_cost"]

        carbon_kg = self.objectives.compute_carbon_emissions_kg(trans_dict, exp_dict)
        gap = getattr(res, "mip_gap", None)
        solve_status = "OPTIMAL" if (gap is None or gap < 1e-4) else "FEASIBLE_GAP"
        cert = OptimizationCertificateContract(
            solve_status=solve_status, objective_value=round(float(res.fun), 2),
            best_bound=round(float(getattr(res, "best_bound", res.fun)), 2),
            mip_gap=round(float(gap), 4) if gap is not None else 0.0,
            runtime_seconds=runtime_seconds, candidate_space_version="v2.0-spord",
            scenario_set_version="v2.0-enterprise", constraint_set_hash=constraint_set_hash,
        )

        return OptimizationResult(
            status=solve_status, success=True, objective_value=round(float(res.fun), 2),
            production_allocations=prod_allocs, overtime_hours=ot_dict, transfer_shifts=shifts_list,
            expedite_units=exp_dict, unmet_demand_units=unmet_u, total_fulfilled_units=fulfilled_u,
            total_cost_usd=cost_breakdown["total_direct_cost"], total_protected_revenue_usd=round(fulfilled_u * unit_price, 2),
            carbon_emissions_kg=carbon_kg, emergency_sourcing_units=total_spot_u,
            unmet_demand_without_emergency_units=round(unmet_u + total_spot_u, 1),
            residual_unmet_demand_units=unmet_u, full_service_achieved=(unmet_u <= 1.0),
            certificate=cert,
            slack_details={
                "demand_unmet_slack": unmet_u,
                "primary_capacity_headroom": round(factory_capacities.get(primary_factory, 0) - prod_allocs.get(primary_factory, 0), 1),
                "emergency_sourcing_used": total_spot_u,
            },
        )

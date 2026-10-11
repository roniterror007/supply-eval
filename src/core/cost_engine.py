"""
PlanFlow AI - Centralized Cost Engine (Single Source of Truth)
Loads all production, overtime, transfer, and material expedite costs from
configs/business_rules.yaml so that the MILP, objectives evaluator, and agents
all reference identical cost parameters.
"""

import os
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from src.core.logger import get_logger

logger = get_logger("CostEngine")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "business_rules.yaml")


@dataclass
class FactoryCostProfile:
    """All cost and capacity parameters for a single factory."""
    name: str
    region: str
    base_capacity_units: float
    labor_rate_hourly: float  # also used as production cost per unit
    overtime_capacity_multiplier: float
    overtime_cost_multiplier: float
    transfer_costs_to: Dict[str, float] = field(default_factory=dict)

    @property
    def production_cost_per_unit(self) -> float:
        return self.labor_rate_hourly

    @property
    def overtime_cost_per_hour(self) -> float:
        return self.labor_rate_hourly * self.overtime_cost_multiplier


@dataclass
class MaterialCostProfile:
    """Cost parameters for a critical material."""
    material_id: str
    name: str
    category: str
    supplier: str
    std_lead_time_days: int
    expedite_lead_time_days: int
    expedite_cost_per_unit: float
    min_order_qty: int


class CostEngine:
    """
    Centralised cost engine loaded from business_rules.yaml.
    Every subsystem (MILP, objectives, agents) MUST use this engine
    instead of hardcoded cost constants.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path or _CONFIG_PATH
        self._rules: Dict = {}
        self.factories: Dict[str, FactoryCostProfile] = {}
        self.materials: Dict[str, MaterialCostProfile] = {}
        self.business_line_compatibility: Dict[str, List[str]] = {}
        self._load()

    def _load(self):
        path = os.path.normpath(self._config_path)
        if not os.path.exists(path):
            logger.warning(f"business_rules.yaml not found at {path}, using defaults")
            self._load_defaults()
            return

        with open(path, "r") as f:
            self._rules = yaml.safe_load(f)

        # Factories
        for fname, fdata in self._rules.get("factories", {}).items():
            self.factories[fname] = FactoryCostProfile(
                name=fname,
                region=fdata.get("region", ""),
                base_capacity_units=float(fdata.get("base_capacity_units", 12000)),
                labor_rate_hourly=float(fdata.get("labor_rate_hourly", 35.0)),
                overtime_capacity_multiplier=float(fdata.get("overtime_capacity_multiplier", 1.2)),
                overtime_cost_multiplier=float(fdata.get("overtime_cost_multiplier", 1.5)),
                transfer_costs_to={k: float(v) for k, v in fdata.get("transfer_cost_per_unit_to", {}).items()},
            )

        # Materials
        for mat in self._rules.get("critical_materials", []):
            mid = mat["id"]
            self.materials[mid] = MaterialCostProfile(
                material_id=mid,
                name=mat.get("name", mid),
                category=mat.get("category", ""),
                supplier=mat.get("supplier", ""),
                std_lead_time_days=int(mat.get("std_lead_time_days", 30)),
                expedite_lead_time_days=int(mat.get("expedite_lead_time_days", 10)),
                expedite_cost_per_unit=float(mat.get("expedite_cost_per_unit", 100.0)),
                min_order_qty=int(mat.get("min_order_qty", 100)),
            )

        # Business-line → compatible factories
        for bl in self._rules.get("business_lines", []):
            if isinstance(bl, dict):
                self.business_line_compatibility[bl["name"]] = bl.get("compatible_factories", [])

        logger.info(f"CostEngine loaded: {len(self.factories)} factories, {len(self.materials)} materials")

    def _load_defaults(self):
        """Fallback defaults when YAML is unavailable."""
        defaults = {
            "Factory A (Stuttgart)": (12000, 48.0, 1.25, 1.5),
            "Factory B (Detroit)":   (15000, 42.0, 1.20, 1.5),
            "Factory C (Yokohama)":  (10000, 38.0, 1.15, 1.4),
            "Factory D (Wroclaw)":   (14000, 26.0, 1.30, 1.35),
            "Factory E (Monterrey)": (16000, 22.0, 1.30, 1.3),
        }
        for fname, (cap, rate, ot_mult, ot_cost) in defaults.items():
            self.factories[fname] = FactoryCostProfile(
                name=fname, region="", base_capacity_units=cap,
                labor_rate_hourly=rate, overtime_capacity_multiplier=ot_mult,
                overtime_cost_multiplier=ot_cost,
            )
        self.materials = {
            "MAT-A1": MaterialCostProfile("MAT-A1", "Analytical Processor Board", "Electronics", "TechFoundry", 45, 15, 100.0, 100),
            "MAT-B2": MaterialCostProfile("MAT-B2", "High-Tensile Load Cell", "Mechanical", "ForceMetrics Inc", 30, 10, 50.0, 200),
            "MAT-C3": MaterialCostProfile("MAT-C3", "Optical Sensor Array", "Sensors", "OpticCore", 60, 20, 150.0, 50),
        }

    # ── Query APIs ──────────────────────────────────────────────────────

    def _resolve_factory(self, factory_name: str) -> Optional[FactoryCostProfile]:
        if factory_name in self.factories:
            return self.factories[factory_name]
        for fname, fp in self.factories.items():
            if factory_name in fname or fname in factory_name:
                return fp
        return None

    def get_production_cost(self, factory_name: str) -> float:
        fp = self._resolve_factory(factory_name)
        return fp.production_cost_per_unit if fp else 35.0

    def get_overtime_cost(self, factory_name: str) -> float:
        fp = self._resolve_factory(factory_name)
        return fp.overtime_cost_per_hour if fp else 52.5

    def get_transfer_cost(self, from_factory: str, to_factory: str) -> float:
        fp = self._resolve_factory(from_factory)
        fp_to = self._resolve_factory(to_factory)
        to_name = fp_to.name if fp_to else to_factory
        if fp:
            if to_name in fp.transfer_costs_to:
                return fp.transfer_costs_to[to_name]
            for dest, cost in fp.transfer_costs_to.items():
                if to_factory in dest or dest in to_factory:
                    return cost
        return 25.0

    def get_expedite_cost(self, material_id: str) -> float:
        mp = self.materials.get(material_id)
        return mp.expedite_cost_per_unit if mp else 100.0

    def get_factory_capacity(self, factory_name: str) -> float:
        fp = self.factories.get(factory_name)
        return fp.base_capacity_units if fp else 12000.0

    def get_all_factory_capacities(self) -> Dict[str, float]:
        return {n: fp.base_capacity_units for n, fp in self.factories.items()}

    def get_compatible_factories(self, business_line: str) -> List[str]:
        return self.business_line_compatibility.get(business_line, list(self.factories.keys()))

    def get_all_factory_names(self) -> List[str]:
        return list(self.factories.keys())

    def get_sla_penalty_multiplier(self) -> float:
        """Unmet demand penalty as a multiplier of unit price."""
        return 1.5

    def get_carbon_cost_per_kg(self) -> float:
        """Enterprise carbon cost in $/kg CO2e (EPA/ESG 2026 benchmark: $50/metric ton)."""
        return 0.05  # $50/metric ton = $0.05/kg


# Module-level singleton
_engine_instance: Optional[CostEngine] = None


def get_cost_engine(config_path: Optional[str] = None) -> CostEngine:
    """Returns the singleton CostEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CostEngine(config_path)
    return _engine_instance

"""
PlanFlow AI - Enterprise Database Grounding Checker
Verifies that all entity references (factories, materials, business lines, regions, suppliers)
made by peer agents are grounded in enterprise master records and not hallucinated.
"""

import os
import yaml
from typing import List, Dict, Any, Set, Optional
from src.core.logger import get_logger, log_audit_check

logger = get_logger("GroundingChecker")


class EnterpriseGroundingChecker:
    """Verifies that all agent claims and entities exist in master database records."""

    def __init__(self, config_path: str = "configs/business_rules.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config path {config_path} not found")

        with open(config_path, "r") as f:
            self.rules = yaml.safe_load(f)

        self.valid_business_lines: Set[str] = set(b["name"] for b in self.rules.get("business_lines", []))
        self.valid_regions: Set[str] = set(r["name"] for r in self.rules.get("regions", []))
        self.valid_factories: Set[str] = set(self.rules.get("factories", {}).keys())
        self.valid_materials: Set[str] = set(m["id"] for m in self.rules.get("critical_materials", []))
        self.material_master: Dict[str, Dict[str, Any]] = {
            m["id"]: m for m in self.rules.get("critical_materials", [])
        }

    def check_entities(
        self,
        business_line: str,
        region: str,
        factory_names: List[str],
        material_ids: List[str],
    ) -> Dict[str, Any]:
        """Audits grounding of enterprise entities."""
        unrecognized = []

        if business_line not in self.valid_business_lines:
            unrecognized.append(f"Business Line: '{business_line}'")

        if region not in self.valid_regions:
            unrecognized.append(f"Region: '{region}'")

        for f in factory_names:
            if f not in self.valid_factories:
                unrecognized.append(f"Factory: '{f}'")

        for m in material_ids:
            if m not in self.valid_materials:
                unrecognized.append(f"Material Code: '{m}'")

        grounded = len(unrecognized) == 0
        if grounded:
            log_audit_check("Enterprise Grounding", True, "All entities match master data catalogs")
        else:
            log_audit_check(
                "Enterprise Grounding",
                False,
                f"Unrecognized/hallucinated entities: {', '.join(unrecognized)}",
            )

        return {
            "grounded": grounded,
            "unrecognized_entities": unrecognized,
            "audited_entities_count": 2 + len(factory_names) + len(material_ids),
        }

    def get_material_info(self, mat_id: str) -> Optional[Dict[str, Any]]:
        """Returns verified master data for a material."""
        return self.material_master.get(mat_id)

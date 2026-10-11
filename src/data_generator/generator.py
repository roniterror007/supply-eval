"""
PlanFlow AI - Enterprise Supply Chain Dataset & Ground Truth Generator
Generates 10,000 highly realistic manufacturing and demand-to-supply planning records.
Splits into 8,000 reference records and 2,000 test ground-truth records with deterministic physics & economics.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from src.core.logger import get_logger

logger = get_logger("DataGenerator")


class EnterpriseDatasetGenerator:
    """Generates synthetic high-dimensional enterprise demand-to-supply data with ground truth."""

    BUSINESS_LINES = [
        "Analytical",
        "Force and Measurement",
        "Instrumentation",
    ]

    REGIONS = ["Germany", "United States", "Japan", "Poland", "Mexico"]

    FACTORY_MAP = {
        "Germany": "Factory A (Stuttgart)",
        "United States": "Factory B (Detroit)",
        "Japan": "Factory C (Yokohama)",
        "Poland": "Factory D (Wroclaw)",
        "Mexico": "Factory E (Monterrey)",
    }

    CAPACITIES = {
        "Factory A (Stuttgart)": 12000,
        "Factory B (Detroit)": 15000,
        "Factory C (Yokohama)": 10000,
        "Factory D (Wroclaw)": 14000,
        "Factory E (Monterrey)": 16000,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate(self, total_samples: int = 10000) -> pd.DataFrame:
        """Generates total_samples of realistic enterprise planning instances."""
        logger.info(f"Synthesizing {total_samples} enterprise planning records with ground truth...")

        n = total_samples

        # Categorical assignment
        business_line = np.random.choice(self.BUSINESS_LINES, size=n, p=[0.40, 0.35, 0.25])
        region = np.random.choice(self.REGIONS, size=n, p=[0.25, 0.30, 0.15, 0.15, 0.15])
        factory = np.array([self.FACTORY_MAP[r] for r in region])

        # Base capacity with periodic maintenance fluctuations (85% to 100%)
        nominal_capacity = np.array([self.CAPACITIES[f] for f in factory], dtype=float)
        maintenance_factor = np.random.uniform(0.88, 1.0, size=n)
        effective_capacity = nominal_capacity * maintenance_factor

        # Historical order patterns (monthly averages with business line multipliers)
        base_demand_scale = {
            "Analytical": 250,
            "Force and Measurement": 450,
            "Instrumentation": 120,
        }
        scales = np.array([base_demand_scale[b] for b in business_line], dtype=float)

        hist_m3 = np.maximum(20, np.random.normal(scales, scales * 0.25)) # High volatility
        hist_m2 = np.maximum(20, hist_m3 * np.random.uniform(0.85, 1.25, size=n))
        rules = yaml.safe_load(open("configs/business_rules.yaml"))
        business_lines = [b["name"] for b in rules["business_lines"]]
        regions_map = {r["name"]: r for r in rules["regions"]}
        factories_map = rules["factories"]
        
        dates = pd.date_range(start="2022-01-01", end="2026-03-01", freq="MS")

        combinations = []
        for r_name in regions_map.keys():
            for bl in business_lines:
                for dt in dates:
                    combinations.append((r_name, bl, dt))

        np.random.seed(self.seed)
        
        # We will sample from combinations to create `total_samples`
        sampled_indices = np.random.choice(len(combinations), size=total_samples, replace=True)
        
        data = []
        for idx, combo_idx in enumerate(sampled_indices):
            r_name, bl, dt = combinations[combo_idx]
            f_name = regions_map[r_name]["primary_factory"]
            market_growth = regions_map[r_name]["market_growth_trend"]
            currency = regions_map[r_name]["currency"]
            
            # Erratic B2B time series element (seasonal + noise)
            month_idx = dt.month
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month_idx / 12)
            erratic_noise = np.random.normal(1.0, 0.4) # high variance
            
            base_demand_scale = 1000 if bl == "Instrumentation" else (800 if bl == "Analytical" else 400)
            
            hist_m1 = max(10, base_demand_scale * market_growth * seasonal_factor * erratic_noise)
            hist_m2 = max(10, base_demand_scale * market_growth * (1.0 + 0.3 * np.sin(2 * np.pi * ((month_idx-1)%12)/12)) * np.random.normal(1.0, 0.4))
            hist_m3 = max(10, base_demand_scale * market_growth * (1.0 + 0.3 * np.sin(2 * np.pi * ((month_idx-2)%12)/12)) * np.random.normal(1.0, 0.4))
            
            unit_price = 450000.0 if bl == "Instrumentation" else (125000.0 if bl == "Analytical" else 75000.0)
            unit_price = unit_price * np.random.uniform(0.9, 1.1)

            # Sales Pipeline
            active_pipeline_usd = hist_m1 * unit_price * np.random.uniform(1.2, 3.5)
            stale_deals_usd = active_pipeline_usd * np.random.uniform(0.1, 0.4)
            unweighted_pipeline_usd = active_pipeline_usd + stale_deals_usd
            
            pipeline_units_equiv = active_pipeline_usd / unit_price
            true_win_rate = np.clip(np.random.normal(0.38, 0.08), 0.10, 0.70)
            pipeline_realized_units = pipeline_units_equiv * true_win_rate

            surge_pct = np.random.choice([0.0, -0.25, 0.15, 0.45, 0.80], p=[0.40, 0.20, 0.15, 0.15, 0.10])
            market_index = np.clip(np.random.normal(105, 12), 70, 140)
            backlog_units = np.random.uniform(50, 300)

            # BOM
            mat_A1_req = np.random.choice([1.0, 2.0, 4.0])
            mat_B2_req = np.random.choice([0.5, 1.0, 2.0])
            mat_C3_req = np.random.choice([2.0, 5.0, 10.0])
            
            mat_A1_stock = hist_m1 * mat_A1_req * np.random.uniform(0.5, 1.2)
            mat_B2_stock = hist_m1 * mat_B2_req * np.random.uniform(0.3, 0.9)
            mat_C3_stock = hist_m1 * mat_C3_req * np.random.uniform(0.8, 1.5)
            
            mat_A1_in_transit = np.random.uniform(10, 500)
            mat_B2_in_transit = np.random.uniform(10, 500)
            mat_C3_in_transit = np.random.uniform(50, 1000)
            
            mat_A1_delay = np.random.choice([0, 10, 25, 45], p=[0.50, 0.25, 0.15, 0.10])
            mat_B2_delay = np.random.choice([0, 5, 15, 30], p=[0.40, 0.30, 0.20, 0.10])
            mat_C3_delay = np.random.choice([0, 14, 40], p=[0.70, 0.20, 0.10])

            # True Demand
            macro_factor = market_index / 100.0
            base_trend = (hist_m1 * 0.5 + hist_m2 * 0.3 + hist_m3 * 0.2) * macro_factor
            true_demand = (base_trend + (pipeline_realized_units * 0.6)) * (1.0 + surge_pct)
            true_demand = max(np.round(true_demand, 1), backlog_units)

            # Returns
            returns_rate = np.random.uniform(0.01, 0.05)
            returns_units = np.round(true_demand * returns_rate, 1)
            returns_usd = np.round(returns_units * unit_price, 2)

            # True factory util
            eff_cap = factories_map[f_name]["base_capacity_units"] * np.random.uniform(0.85, 1.05)
            true_util = np.round((true_demand / eff_cap) * 100.0, 1)
            true_overload = 1 if true_util > 100.0 else 0

            # Detailed Costs
            labor_rate = factories_map[f_name]["labor_rate_hourly"]
            labor_hrs_per_unit = np.random.uniform(10.0, 40.0)
            labor_cost = true_demand * labor_hrs_per_unit * labor_rate
            electricity_cost = true_demand * np.random.uniform(100.0, 500.0)
            material_cost = true_demand * (mat_A1_req * 100.0 + mat_B2_req * 50.0 + mat_C3_req * 150.0) * np.random.uniform(0.9, 1.1)
            overhead_cost = true_demand * np.random.uniform(500.0, 2000.0)
            production_cost = labor_cost + electricity_cost + material_cost + overhead_cost

            # Material shortages
            mat_A1_req_tot = true_demand * mat_A1_req
            mat_B2_req_tot = true_demand * mat_B2_req
            mat_C3_req_tot = true_demand * mat_C3_req
            
            eff_stock_A1 = mat_A1_stock + (mat_A1_in_transit if mat_A1_delay <= 7 else 0)
            eff_stock_B2 = mat_B2_stock + (mat_B2_in_transit if mat_B2_delay <= 7 else 0)
            eff_stock_C3 = mat_C3_stock + (mat_C3_in_transit if mat_C3_delay <= 7 else 0)
            
            short_A1 = max(0, mat_A1_req_tot - eff_stock_A1)
            short_B2 = max(0, mat_B2_req_tot - eff_stock_B2)
            short_C3 = max(0, mat_C3_req_tot - eff_stock_C3)
            tot_short = short_A1 + short_B2 + short_C3
            true_shortage_detected = 1 if tot_short > 0 else 0

            # Delivery Risk
            if true_overload and true_shortage_detected:
                risk = "CRITICAL"
            elif true_overload or (tot_short > 50):
                risk = "HIGH"
            elif tot_short > 0 or true_util > 92.0:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            # Action logic
            if risk == "CRITICAL":
                opt_act = "EXPEDITE_AND_SHIFT_FACTORY"
                mit_cost = (tot_short * 500.0) + (max(0, true_demand - eff_cap) * 1500.0)
                saved = (true_demand * unit_price) * 0.35
            elif risk == "HIGH" and true_overload:
                opt_act = "SHIFT_FACTORY_TRANSFER"
                mit_cost = max(0, true_demand - eff_cap) * 1500.0
                saved = max(0, true_demand - eff_cap) * unit_price
            elif risk == "HIGH" and true_shortage_detected:
                opt_act = "EXPEDITE_CRITICAL_SUPPLIER"
                mit_cost = tot_short * 500.0
                saved = (true_demand * unit_price) * 0.15
            elif risk == "MEDIUM":
                opt_act = "SCHEDULE_OVERTIME_PREBUILD"
                mit_cost = 50000.0
                saved = (true_demand * unit_price) * 0.08
            else:
                opt_act = "MAINTAIN_NORMAL_PRODUCTION"
                mit_cost = 0.0
                saved = 0.0

            signal_noise = np.abs(hist_m1 - hist_m2) / (hist_m1 + 1e-5)
            true_calibrated_confidence = np.clip(1.0 - (signal_noise * 1.5) - (surge_pct * 0.3), 0.52, 0.98)

            data.append({
                "scenario_id": f"SCN-{idx:06d}",
                "date": dt.strftime("%Y-%m-%d"),
                "business_line": bl,
                "region": r_name,
                "factory": f_name,
                "currency": currency,
                "hist_orders_m3": round(hist_m3, 1),
                "hist_orders_m2": round(hist_m2, 1),
                "hist_orders_m1": round(hist_m1, 1),
                "market_index": round(market_index, 2),
                "backlog_units": round(backlog_units, 1),
                "unweighted_pipeline_usd": round(unweighted_pipeline_usd, 2),
                "stale_deals_usd": round(stale_deals_usd, 2),
                "active_pipeline_usd": round(active_pipeline_usd, 2),
                "unit_price_usd": round(unit_price, 2),
                "nominal_capacity": factories_map[f_name]["base_capacity_units"],
                "effective_capacity": round(eff_cap, 1),
                "surge_pct": surge_pct,
                "mat_A1_req": mat_A1_req,
                "mat_B2_req": mat_B2_req,
                "mat_C3_req": mat_C3_req,
                "mat_A1_initial_stock": round(mat_A1_stock, 1),
                "mat_B2_initial_stock": round(mat_B2_stock, 1),
                "mat_C3_initial_stock": round(mat_C3_stock, 1),
                "mat_A1_in_transit": round(mat_A1_in_transit, 1),
                "mat_B2_in_transit": round(mat_B2_in_transit, 1),
                "mat_C3_in_transit": round(mat_C3_in_transit, 1),
                "mat_A1_delay_days": mat_A1_delay,
                "mat_B2_delay_days": mat_B2_delay,
                "mat_C3_delay_days": mat_C3_delay,
                "returns_units": returns_units,
                "returns_usd": returns_usd,
                "labor_cost_usd": round(labor_cost, 2),
                "electricity_cost_usd": round(electricity_cost, 2),
                "material_cost_usd": round(material_cost, 2),
                "overhead_cost_usd": round(overhead_cost, 2),
                "production_cost_usd": round(production_cost, 2),
                "ground_truth_demand": true_demand,
                "ground_truth_revenue_usd": round(true_demand * unit_price, 2),
                "ground_truth_utilization_pct": true_util,
                "ground_truth_overload": true_overload,
                "ground_truth_mat_A1_shortage": round(short_A1, 1),
                "ground_truth_mat_B2_shortage": round(short_B2, 1),
                "ground_truth_mat_C3_shortage": round(short_C3, 1),
                "ground_truth_shortage_detected": true_shortage_detected,
                "ground_truth_delivery_risk": risk,
                "ground_truth_optimal_action": opt_act,
                "ground_truth_mitigation_cost_usd": round(mit_cost, 2),
                "ground_truth_revenue_protected_usd": round(saved, 2),
                "ground_truth_confidence": np.round(true_calibrated_confidence, 3),
            })

        df = pd.DataFrame(data)
        logger.info(f"Dataset generated successfully with {len(df)} rows and {len(df.columns)} features.")
        return df


def generate_enterprise_datasets(
    output_dir: str = "data",
    total_samples: int = 10000,
    reference_samples: int = 8000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates and persists the 8,000 reference and 2,000 ground-truth test datasets."""
    os.makedirs(output_dir, exist_ok=True)
    generator = EnterpriseDatasetGenerator(seed=seed)
    df_all = generator.generate(total_samples=total_samples)

    # Deterministic split: first reference_samples as reference, rest as test ground truth
    df_ref = df_all.iloc[:reference_samples].copy()
    df_test = df_all.iloc[reference_samples:].copy().reset_index(drop=True)

    ref_path = os.path.join(output_dir, "enterprise_reference_8k.parquet")
    test_path = os.path.join(output_dir, "test_ground_truth_2k.parquet")
    meta_path = os.path.join(output_dir, "dataset_metadata.json")

    df_ref.to_parquet(ref_path, index=False)
    df_test.to_parquet(test_path, index=False)

    metadata = {
        "dataset_name": "PlanFlow AI Enterprise Planning Benchmark",
        "total_samples": total_samples,
        "reference_samples": len(df_ref),
        "test_ground_truth_samples": len(df_test),
        "columns": list(df_all.columns),
        "ground_truth_columns": [c for c in df_all.columns if c.startswith("ground_truth_")],
        "business_lines": list(df_all["business_line"].unique()),
        "regions": list(df_all["region"].unique()),
        "factories": list(df_all["factory"].unique()),
        "overload_prevalence_pct": round(float(df_all["ground_truth_overload"].mean()) * 100, 2),
        "shortage_prevalence_pct": round(float(df_all["ground_truth_shortage_detected"].mean()) * 100, 2),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[green]Saved reference dataset ({len(df_ref)} rows) to {ref_path}[/green]")
    logger.info(f"[green]Saved test ground-truth dataset ({len(df_test)} rows) to {test_path}[/green]")
    logger.info(f"[green]Saved dataset metadata to {meta_path}[/green]")

    return df_ref, df_test

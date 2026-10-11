"""
PlanFlow AI - Enterprise Benchmark Runner
Executes comprehensive evaluation against the 2,000 test ground-truth dataset.
Scores accuracy, precision, recall, F1, Expected Calibration Error (ECE), and Brier score.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from src.core.logger import get_logger
from src.models.neural_surrogate import ModelPredictor
from src.evaluation.metrics import calculate_regression_metrics, calculate_classification_metrics
from src.evaluation.calibration import calculate_ece_and_brier
from src.agents.orchestrator import PlanFlowOrchestrator

logger = get_logger("BenchmarkRunner")


class EnterpriseBenchmarkRunner:
    """Automated benchmark scoring suite against ground-truth test data."""

    def __init__(
        self,
        test_dataset_path: str = "data/test_ground_truth_2k.parquet",
        model_checkpoint: str = "checkpoints/best_surrogate_model.pt",
    ):
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Test ground truth dataset not found at {test_dataset_path}")

        self.test_df = pd.read_parquet(test_dataset_path)
        self.predictor = ModelPredictor(model_checkpoint)
        self.orchestrator = PlanFlowOrchestrator(model_checkpoint)

    def run_full_benchmark(self, agent_eval_samples: int = 50) -> Dict[str, Any]:
        """
        Executes full benchmark evaluation across all 2,000 test samples on GPU.
        Also runs multi-agent cycle on a subset to measure contract and verification compliance.
        """
        logger.info(f"Running GPU benchmark across {len(self.test_df)} ground-truth test instances...")

        # 1. GPU Batch Inference on all 2,000 test records
        preds = self.predictor.predict_batch(self.test_df)

        y_true_demand = self.test_df["ground_truth_demand"].values
        y_pred_demand = preds["p50"]

        y_true_util = self.test_df["ground_truth_utilization_pct"].values
        y_pred_util = preds["utilization"]

        y_true_overload = self.test_df["ground_truth_overload"].values
        y_pred_overload = (preds["overload_prob"] >= 0.5).astype(int)

        y_true_shortage = self.test_df["ground_truth_shortage_detected"].values
        y_pred_shortage = (preds["shortage_prob"] >= 0.5).astype(int)

        # 2. Compute Metrics
        demand_metrics = calculate_regression_metrics(y_true_demand, y_pred_demand)
        util_metrics = calculate_regression_metrics(y_true_util, y_pred_util)

        overload_metrics = calculate_classification_metrics(y_true_overload, y_pred_overload)
        shortage_metrics = calculate_classification_metrics(y_true_shortage, y_pred_shortage)

        # 3. Confidence Calibration (ECE & Brier Score)
        # Check calibration of overload probability vs actual overload
        overload_calibration = calculate_ece_and_brier(preds["overload_prob"], y_true_overload, n_bins=10)
        shortage_calibration = calculate_ece_and_brier(preds["shortage_prob"], y_true_shortage, n_bins=10)

        # 4. Multi-Agent End-to-End Hierarchy Audit
        logger.info(f"Auditing multi-agent planning hierarchy on {agent_eval_samples} diverse scenarios...")
        agent_audit_scores = []
        contracts_valid_count = 0
        boss_approvals_count = 0
        corrections_triggered = 0

        eval_subset = self.test_df.head(agent_eval_samples)
        for _, row in eval_subset.iterrows():
            row_dict = row.to_dict()
            try:
                res = self.orchestrator.run_planning_cycle(
                    row_dict=row_dict,
                    scenario_name="Benchmark Audit",
                    demand_multiplier=1.0 + float(row_dict.get("surge_pct", 0.0)),
                    supplier_delay_days=int(row_dict.get("supplier_delay_days", 0)),
                )
                m = res["manager_audit"]
                b = res["boss_decision"]

                agent_audit_scores.append(m.verification_score)
                if m.all_contracts_passed:
                    contracts_valid_count += 1
                if b.status.value == "APPROVED":
                    boss_approvals_count += 1
                corrections_triggered += len([msg for msg in res["dialogue_messages"] if msg.is_correction])
            except Exception as e:
                logger.error(f"Error during agent cycle on scenario {row_dict.get('scenario_id')}: {str(e)}")

        avg_audit_score = float(np.mean(agent_audit_scores)) if agent_audit_scores else 0.0
        contract_compliance_pct = (contracts_valid_count / max(1, agent_eval_samples)) * 100.0
        boss_approval_rate_pct = (boss_approvals_count / max(1, agent_eval_samples)) * 100.0

        results = {
            "total_test_samples": len(self.test_df),
            "demand_forecast_metrics": demand_metrics,
            "utilization_metrics": util_metrics,
            "overload_detection_metrics": overload_metrics,
            "shortage_detection_metrics": shortage_metrics,
            "overload_calibration": overload_calibration,
            "shortage_calibration": shortage_calibration,
            "multi_agent_governance": {
                "scenarios_audited": agent_eval_samples,
                "contract_compliance_rate_pct": round(contract_compliance_pct, 1),
                "mean_verification_score": round(avg_audit_score, 1),
                "boss_approval_rate_pct": round(boss_approval_rate_pct, 1),
                "total_peer_corrections_resolved": corrections_triggered,
            },
        }

        logger.info(
            f"[bold green]Benchmark Complete![/bold green] "
            f"Demand MAE: {demand_metrics['mae']} units | "
            f"Overload F1: {overload_metrics['f1_score']} | "
            f"Overload ECE: {overload_calibration['ece_pct']}% | "
            f"Contract Compliance: {contract_compliance_pct}%"
        )
        return results

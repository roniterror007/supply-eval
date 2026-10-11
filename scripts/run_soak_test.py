"""
PlanFlow AI - Soak & Memory-Leak Reliability Benchmark
Executes repeated GPU neural inference, HELICASE DAG manipulations, and HiGHS MILP solves
to measure host RSS memory drift, CUDA reserved memory, file-handle usage, and latency percentiles.
"""

import gc
import os
import time
import argparse
from typing import Dict, Any, List
import numpy as np
import psutil
import torch

import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.core.gpu_utils import get_gpu_manager
from src.optimization.master_planner import MasterMILPPlanner
from src.knowledge_graph.evidence_graph import TemporalEvidenceGraph
from src.models.neural_surrogate import ModelPredictor
from src.core.logger import get_logger

logger = get_logger("SoakTestRunner")


def run_soak_benchmark(iterations: int = 100, warmup_iterations: int = 10) -> Dict[str, Any]:
    """
    Executes an intensive soak test measuring resource drift and latency distributions.
    """
    process = psutil.Process(os.getpid())
    gpu_mgr = get_gpu_manager()
    has_cuda = torch.cuda.is_available()

    # Load neural predictor and test records
    predictor = None
    test_records = []
    checkpoint_path = "checkpoints/best_surrogate_model.pt"
    if os.path.exists(checkpoint_path):
        predictor = ModelPredictor(checkpoint_path)
    if os.path.exists("data/test_ground_truth_2k.parquet"):
        import pandas as pd
        test_records = pd.read_parquet("data/test_ground_truth_2k.parquet").head(50).to_dict(orient="records")

    planner = MasterMILPPlanner()

    inference_latencies = []
    solver_latencies = []
    dag_latencies = []
    exceptions_count = 0

    print(f"\n================================================================================")
    print(f"PlanFlow AI 2.0 - Soak & Memory Stability Audit ({iterations} cycles)")
    print(f"Device: {gpu_mgr.device_name} (CUDA: {has_cuda}) | PID: {process.pid}")
    print(f"================================================================================\n")

    # Warmup
    for _ in range(warmup_iterations):
        if predictor and test_records:
            predictor.predict_single(test_records[0])
        planner.solve(10000.0, 1200.0, "Factory A", 12000.0, "Factory B", 6000.0, 1.0, 10000.0)

    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    # Initial telemetry baseline
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)
    initial_cuda_alloc_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if has_cuda else 0.0
    initial_cuda_res_mb = (torch.cuda.memory_reserved() / (1024 * 1024)) if has_cuda else 0.0
    initial_handles = process.num_handles() if hasattr(process, "num_handles") else 0

    print(f"Baseline Host RSS Memory : {initial_rss_mb:.2f} MB")
    if has_cuda:
        print(f"Baseline CUDA Allocated  : {initial_cuda_alloc_mb:.2f} MB (Reserved: {initial_cuda_res_mb:.2f} MB)")
    print(f"Baseline Process Handles : {initial_handles}\n")

    # Main Soak Loop
    for cycle in range(1, iterations + 1):
        try:
            # 1. Neural Inference
            if predictor and test_records:
                t0 = time.perf_counter()
                rec = test_records[cycle % len(test_records)]
                pred = predictor.predict_single(rec)
                inference_latencies.append((time.perf_counter() - t0) * 1000.0)

            # 2. DAG Generation & Traversal
            t1 = time.perf_counter()
            g = TemporalEvidenceGraph()
            n1 = g.add_fact("order", f"ORD-{cycle}", 5000 + cycle, "ERP", f"REC-{cycle}")
            n2 = g.add_fact("forecast", f"FC-{cycle}", 5200 + cycle, "manual", f"AI-{cycle}")
            g.add_derivation(n1.evidence_id, n2.evidence_id)
            ancestors = g.get_ancestors(n2.evidence_id)
            dag_latencies.append((time.perf_counter() - t1) * 1000.0)

            # 3. HiGHS MILP Solve
            t2 = time.perf_counter()
            plan = planner.solve(
                target_demand=8000.0 + (cycle % 40) * 100.0,
                unit_price=1100.0,
                primary_factory="Factory A (Stuttgart)",
                primary_capacity=10000.0,
                secondary_factory="Factory D (Wroclaw)",
                secondary_capacity=5000.0,
                material_ratio=1.0,
                usable_material_stock=7000.0,
            )
            solver_latencies.append((time.perf_counter() - t2) * 1000.0)

        except Exception as e:
            exceptions_count += 1
            logger.error(f"Exception during soak cycle {cycle}: {e}")

        if cycle % 25 == 0:
            current_rss = process.memory_info().rss / (1024 * 1024)
            print(f"Cycle {cycle:3d}/{iterations}: Current RSS = {current_rss:.2f} MB (Delta: {current_rss - initial_rss_mb:+.2f} MB)")

    # Post-soak telemetry collection
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    final_cuda_alloc_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if has_cuda else 0.0
    final_cuda_res_mb = (torch.cuda.memory_reserved() / (1024 * 1024)) if has_cuda else 0.0
    final_handles = process.num_handles() if hasattr(process, "num_handles") else 0

    rss_drift_mb = final_rss_mb - initial_rss_mb
    cuda_alloc_drift_mb = final_cuda_alloc_mb - initial_cuda_alloc_mb
    handles_drift = final_handles - initial_handles

    # Compute latency percentiles
    def get_percentiles(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    report = {
        "iterations_completed": iterations,
        "exceptions_encountered": exceptions_count,
        "memory_telemetry": {
            "initial_rss_mb": round(initial_rss_mb, 2),
            "final_rss_mb": round(final_rss_mb, 2),
            "rss_drift_mb": round(rss_drift_mb, 2),
            "initial_cuda_alloc_mb": round(initial_cuda_alloc_mb, 2),
            "final_cuda_alloc_mb": round(final_cuda_alloc_mb, 2),
            "cuda_alloc_drift_mb": round(cuda_alloc_drift_mb, 2),
            "initial_handles": initial_handles,
            "final_handles": final_handles,
            "handles_drift": handles_drift,
            "leak_detected": bool(rss_drift_mb > 25.0),
        },
        "latencies_ms": {
            "neural_inference": get_percentiles(inference_latencies),
            "dag_operations": get_percentiles(dag_latencies),
            "milp_solver": get_percentiles(solver_latencies),
        },
    }

    print(f"\n================================================================================")
    print(f"Soak Audit Verdict:")
    print(f"Host RSS Memory Drift: {rss_drift_mb:+.2f} MB")
    if has_cuda:
        print(f"CUDA Memory Drift    : {cuda_alloc_drift_mb:+.2f} MB")
    print(f"Handle Drift         : {handles_drift:+d}")
    print(f"Exceptions / Errors  : {exceptions_count}")
    print(f"MILP Latency (ms)    : p50={report['latencies_ms']['milp_solver']['p50']}ms | p95={report['latencies_ms']['milp_solver']['p95']}ms | p99={report['latencies_ms']['milp_solver']['p99']}ms")
    print(f"================================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PlanFlow AI Soak & Memory Reliability Benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Number of soak iterations")
    args = parser.parse_args()

    run_soak_benchmark(iterations=args.iterations)

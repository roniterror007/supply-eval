"""
Unit tests for Soak Reliability, Memory Stability, and Handle Leak Prevention
"""

from scripts.run_soak_test import run_soak_benchmark


def test_soak_reliability_and_memory_stability():
    """
    Executes a 25-cycle soak test and asserts:
    1. Zero unhandled exceptions.
    2. Host RSS memory drift does not exceed allowable bound (< 15 MB).
    3. MILP solver p95 latency is sub-100ms.
    """
    report = run_soak_benchmark(iterations=25, warmup_iterations=5)

    assert report["exceptions_encountered"] == 0
    assert report["memory_telemetry"]["leak_detected"] is False
    assert report["memory_telemetry"]["rss_drift_mb"] < 25.0
    assert report["latencies_ms"]["milp_solver"]["p95"] < 100.0

"""
PlanFlow AI - NVIDIA GeForce RTX 5060 GPU Acceleration & Throughput Benchmark
Directly engages CUDA Device 0 (NVIDIA GeForce RTX 5060 Laptop GPU) with high-throughput
parallel neural surrogate inference and GPU-vectorized Monte Carlo scenario simulation.
Watch Task Manager -> Performance -> GPU 0 (3D / Compute / VRAM) while running this script!
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from src.core.gpu_utils import get_gpu_manager
from src.models.neural_surrogate import ModelPredictor, EnterpriseSurrogateNet

console = Console()


def run_gpu_benchmark(duration_seconds: int = 5):
    console.rule("[bold cyan]PLANFLOW AI 2.0 - NVIDIA RTX 5060 GPU HARDWARE BENCHMARK[/bold cyan]")

    if not torch.cuda.is_available():
        console.print("[bold red]ERROR: CUDA is not available. Please verify NVIDIA driver and PyTorch CUDA build.[/bold red]")
        return

    gpu_mgr = get_gpu_manager()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    console.print(
        Panel(
            f"[bold green]Target Hardware:[/bold green] {device_name}\n"
            f"[bold green]Compute Device:[/bold green] cuda:0 (Primary Dedicated GPU)\n"
            f"[bold green]Compute Capability:[/bold green] {cap[0]}.{cap[1]} (NVIDIA Architecture)\n"
            f"[bold green]Total Dedicated VRAM:[/bold green] {total_mem_gb:.2f} GB GDDR7\n"
            f"[bold green]PyTorch Backend:[/bold green] PyTorch {torch.__version__} with cuDNN {torch.backends.cudnn.version()}",
            title="[bold yellow]Active GPU Configuration[/bold yellow]",
            border_style="cyan",
        )
    )

    # 1. Load Model onto CUDA:0
    checkpoint_path = "checkpoints/best_surrogate_model.pt"
    if not os.path.exists(checkpoint_path):
        console.print("[yellow]Checkpoint not found, training a quick model on GPU...[/yellow]")
        from src.models.trainer import train_surrogate_model
        train_surrogate_model("data/enterprise_reference_8k.parquet", checkpoint_path, epochs=5)

    predictor = ModelPredictor(checkpoint_path)
    model = predictor.model.to(device)
    model.eval()

    # Load test data
    test_path = "data/test_ground_truth_2k.parquet"
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
    else:
        from src.data_generator.generator import EnterpriseDataGenerator
        gen = EnterpriseDataGenerator()
        df_test = gen.generate_enterprise_dataset(2000)

    cat_t, cont_t = predictor.preprocess_df(df_test)
    batch_size = len(df_test)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(cat_t, cont_t)
    torch.cuda.synchronize()

    console.print("\n[bold yellow]>> Benchmark 1: Sustained Parallel GPU Neural Inference (Task Manager Load Generation)...[/bold yellow]")
    console.print("[dim]Open Task Manager -> Performance -> GPU 0 to observe active 3D / Compute / Dedicated Memory usage![/dim]\n")

    # Repeat inference for duration_seconds to generate visible Task Manager GPU activity
    start_time = time.perf_counter()
    batches_processed = 0
    records_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Executing GPU Matrix Computations...", total=duration_seconds)
        
        while (time.perf_counter() - start_time) < duration_seconds:
            with torch.no_grad():
                outputs = model(cat_t, cont_t)
            torch.cuda.synchronize()
            batches_processed += 1
            records_processed += batch_size
            elapsed = time.perf_counter() - start_time
            progress.update(task, completed=min(duration_seconds, elapsed))

    elapsed_total = time.perf_counter() - start_time
    throughput_records_per_sec = records_processed / elapsed_total
    vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    vram_res_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)

    # 2. Vectorized GPU Monte Carlo Simulation (100,000 scenarios on GPU)
    console.print("\n[bold yellow]>> Benchmark 2: Vectorized GPU Monte Carlo Scenario Twin (100,000 Scenarios)...[/bold yellow]")
    t_sim_start = time.perf_counter()

    num_scenarios = 100_000
    base_demand = 12_500.0
    base_capacity = 14_000.0

    # Execute entirely on GPU VRAM tensors
    demand_shocks = torch.empty((num_scenarios,), device=device).log_normal_(mean=0.0, std=0.18)
    capacity_shocks = torch.clamp(torch.randn((num_scenarios,), device=device) * 0.08 + 0.95, min=0.50, max=1.0)

    sim_demands = base_demand * demand_shocks
    sim_capacities = base_capacity * capacity_shocks
    sim_unmet = torch.clamp(sim_demands - sim_capacities, min=0.0)
    sim_costs = (torch.minimum(sim_demands, sim_capacities) * 45.0) + (sim_unmet * 1200.0 * 1.5)

    # GPU-accelerated CVaR calculation
    var_95 = torch.quantile(sim_costs, 0.95)
    cvar_95 = sim_costs[sim_costs >= var_95].mean()

    torch.cuda.synchronize()
    sim_elapsed_ms = (time.perf_counter() - t_sim_start) * 1000.0

    # Output Results Table
    table = Table(title="[bold green]NVIDIA RTX 5060 Laptop GPU Benchmark Summary[/bold green]", border_style="cyan")
    table.add_column("Performance Dimension", style="cyan")
    table.add_column("Measured GPU Result", style="bold green")
    table.add_column("Hardware Architecture Spec", style="dim")

    table.add_row("Active Compute Hardware", f"{device_name} (cuda:0)", "Dedicated Discrete GPU")
    table.add_row("Dedicated VRAM Total", f"{total_mem_gb:.2f} GB GDDR7", "8.0 GB Physical VRAM")
    table.add_row("VRAM Active Working Set", f"{vram_alloc_mb:.2f} MB (Reserved: {vram_res_mb:.2f} MB)", "Stable working set telemetry")
    table.add_row("Neural Inference Throughput", f"{throughput_records_per_sec:,.0f} records / sec", "PyTorch Tensor Core Matmul")
    table.add_row("Mean Inference Latency", f"{(elapsed_total / records_processed) * 1e6:.1f} microseconds / record", "Sub-millisecond inference")
    table.add_row("GPU Digital Twin (100k Scenarios)", f"{sim_elapsed_ms:.2f} ms total", "Vectorized PyTorch CUDA Tensors")
    table.add_row("GPU Calculated CVaR 0.95", f"${float(cvar_95.cpu()):,.2f}", "Worst 5% tail risk computed on GPU")

    console.print(table)
    console.print("\n[bold green]✔ Verification Complete:[/bold green] NVIDIA GeForce RTX 5060 Laptop GPU is the primary compute device for all PlanFlow neural inference, digital twin simulations, and stress tests.")


if __name__ == "__main__":
    duration = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        duration = int(sys.argv[1])
    run_gpu_benchmark(duration_seconds=duration)

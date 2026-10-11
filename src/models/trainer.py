"""
PlanFlow AI - PyTorch CUDA Training Pipeline
Trains the EnterpriseSurrogateNet on the 8,000 reference records with GPU acceleration.
Uses Pinball/Quantile loss for uncertainty estimation and multi-task gradient descent.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.core.gpu_utils import get_gpu_manager
from src.core.logger import get_logger
from src.models.neural_surrogate import EnterpriseSurrogateNet, ModelPredictor

logger = get_logger("ModelTrainer")


def quantile_loss(preds: torch.Tensor, targets: torch.Tensor, q: float) -> torch.Tensor:
    """Calculates pinball / quantile loss for quantile q in (0, 1)."""
    errors = targets - preds
    return torch.max((q - 1) * errors, q * errors).mean()


def train_surrogate_model(
    reference_data_path: str = "data/enterprise_reference_8k.parquet",
    output_checkpoint: str = "checkpoints/best_surrogate_model.pt",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.003,
) -> ModelPredictor:
    """Trains the surrogate deep learning model on GPU and returns trained ModelPredictor."""
    gpu_manager = get_gpu_manager()
    device = gpu_manager.get_device()
    logger.info(f"Starting model training on compute device: [bold cyan]{device}[/bold cyan]")

    if not os.path.exists(reference_data_path):
        raise FileNotFoundError(f"Reference dataset not found at {reference_data_path}")

    df = pd.read_parquet(reference_data_path)
    predictor = ModelPredictor()

    # Calculate normalization statistics on training data
    cont_vals = df[predictor.CONT_FEATURES].values.astype(np.float32)
    mean = np.mean(cont_vals, axis=0)
    std = np.std(cont_vals, axis=0)
    predictor.set_scaler_params(mean, std)

    cat_tensor, cont_tensor = predictor.preprocess_df(df)

    # Targets
    target_demand = torch.tensor(df["ground_truth_demand"].values, dtype=torch.float32, device=device)
    target_util = torch.tensor(df["ground_truth_utilization_pct"].values, dtype=torch.float32, device=device)
    target_overload = torch.tensor(df["ground_truth_overload"].values, dtype=torch.float32, device=device)
    target_shortage = torch.tensor(df["ground_truth_shortage_detected"].values, dtype=torch.float32, device=device)

    dataset = TensorDataset(cat_tensor, cont_tensor, target_demand, target_util, target_overload, target_shortage)

    # Split 85% train, 15% validation
    n_total = len(dataset)
    n_train = int(n_total * 0.85)
    n_val = n_total - n_train

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = predictor.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    logger.info(f"Training across {epochs} epochs with batch size {batch_size}...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for b_cat, b_cont, b_demand, b_util, b_overload, b_shortage in train_loader:
            optimizer.zero_grad()
            out = model(b_cat, b_cont)

            # Quantile Pinball Loss (normalized by 1000)
            l_p10 = quantile_loss(out["p10"] / 1000.0, b_demand / 1000.0, 0.1)
            l_p50 = quantile_loss(out["p50"] / 1000.0, b_demand / 1000.0, 0.5)
            l_p90 = quantile_loss(out["p90"] / 1000.0, b_demand / 1000.0, 0.9)
            loss_demand = l_p10 + 2.0 * l_p50 + l_p90

            loss_util = mse_loss_fn(out["utilization"] / 100.0, b_util / 100.0)
            loss_overload = bce_loss_fn(out["overload_logits"], b_overload)
            loss_shortage = bce_loss_fn(out["shortage_logits"], b_shortage)

            # Combined Multi-Task Loss with balanced gradient contributions
            total_loss = loss_demand * 2.0 + loss_util * 1.0 + loss_overload * 0.5 + loss_shortage * 0.5

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(total_loss.item())

        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = []
        val_demand_mae = []
        with torch.no_grad():
            for b_cat, b_cont, b_demand, b_util, b_overload, b_shortage in val_loader:
                out = model(b_cat, b_cont)
                val_loss = (
                    quantile_loss(out["p50"] / 1000.0, b_demand / 1000.0, 0.5) * 2.0
                    + mse_loss_fn(out["utilization"] / 100.0, b_util / 100.0) * 1.0
                    + bce_loss_fn(out["overload_logits"], b_overload) * 0.5
                    + bce_loss_fn(out["shortage_logits"], b_shortage) * 0.5
                )
                val_losses.append(val_loss.item())
                val_demand_mae.append(torch.abs(out["p50"] - b_demand).mean().item())

        mean_val_loss = np.mean(val_losses)
        mean_val_mae = np.mean(val_demand_mae)

        if epoch % 5 == 0 or epoch == epochs:
            telemetry = gpu_manager.get_telemetry()
            vram_mb = telemetry.get("vram_allocated_mb", 0)
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {np.mean(train_losses):.4f} - "
                f"Val Loss: {mean_val_loss:.4f} - Val Demand MAE: {mean_val_mae:.1f} units - "
                f"GPU VRAM: {vram_mb} MB"
            )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            predictor.save(output_checkpoint)

    logger.info(f"[bold green]Model training complete. Checkpoint saved to {output_checkpoint}[/bold green]")
    return predictor

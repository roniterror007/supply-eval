"""
PlanFlow AI - GPU Training CLI Script
Trains the PyTorch multi-task surrogate network on the 8,000 reference dataset using CUDA.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.trainer import train_surrogate_model
from src.core.logger import get_logger

logger = get_logger("TrainCLI")

if __name__ == "__main__":
    logger.info("Initiating GPU training on reference dataset...")
    predictor = train_surrogate_model(
        reference_data_path="data/enterprise_reference_8k.parquet",
        output_checkpoint="checkpoints/best_surrogate_model.pt",
        epochs=25,
        batch_size=128,
        learning_rate=0.003,
    )
    logger.info("Model successfully trained and persisted.")

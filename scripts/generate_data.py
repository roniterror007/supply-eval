"""
PlanFlow AI - Data Generation CLI Script
Generates 10,000 enterprise planning instances split into:
- 8,000 reference records (data/enterprise_reference_8k.parquet)
- 2,000 test ground-truth records (data/test_ground_truth_2k.parquet)
"""

import sys
import os

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator.generator import generate_enterprise_datasets
from src.core.logger import get_logger

logger = get_logger("DataGenCLI")

if __name__ == "__main__":
    logger.info("Initializing PlanFlow AI 10,000 dataset generation pipeline...")
    df_ref, df_test = generate_enterprise_datasets(
        output_dir="data",
        total_samples=10000,
        reference_samples=8000,
        seed=42,
    )
    logger.info(f"Successfully created Reference Set: {df_ref.shape} and Test Ground Truth: {df_test.shape}")

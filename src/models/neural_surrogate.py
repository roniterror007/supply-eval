"""
PlanFlow AI - GPU Neural Demand & Digital Twin Surrogate Network
PyTorch deep learning model with multi-task architecture for:
1. Demand quantile forecasting (P10, P50, P90)
2. Factory utilization % regression
3. Bottleneck overload probability logits
4. Critical material shortage probability logits
Optimized for CUDA execution on NVIDIA GeForce RTX 5060 Laptop GPU.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.gpu_utils import get_gpu_manager
from src.core.logger import get_logger

logger = get_logger("NeuralSurrogate")


class EnterpriseSurrogateNet(nn.Module):
    """Deep Multi-Task Neural Surrogate for Demand Forecasting & Digital Twin Simulation."""

    def __init__(
        self,
        num_continuous_features: int = 14,
        embedding_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.15,
    ):
        super(EnterpriseSurrogateNet, self).__init__()

        # Categorical Embeddings
        # Default: business_line (4 classes -> dim 8), region (5 classes -> dim 8), factory (5 classes -> dim 8)
        if embedding_dims is None:
            embedding_dims = {
                "business_line": (3, 8),
                "region": (5, 8),
                "factory": (5, 8),
            }

        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(num_classes, emb_dim) for name, (num_classes, emb_dim) in embedding_dims.items()}
        )

        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_dims.values())
        input_dim = num_continuous_features + total_emb_dim

        # Backbone Shared Feature Extractor
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())  # Smooth Swish activation
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Multi-Task Heads
        # 1. Demand Quantiles (P10, P50, P90)
        self.demand_p50_head = nn.Linear(in_dim, 1)
        self.demand_p10_head = nn.Linear(in_dim, 1)
        self.demand_p90_head = nn.Linear(in_dim, 1)

        # 2. Factory Utilization Head (Continuous %)
        self.utilization_head = nn.Linear(in_dim, 1)

        # 3. Overload Probability Logit Head
        self.overload_head = nn.Linear(in_dim, 1)

        # 4. Material Shortage Probability Logit Head
        self.shortage_head = nn.Linear(in_dim, 1)

        # Initialize biases to realistic enterprise scale
        self.demand_p50_head.bias.data.fill_(1.0)  # 1.0 * 10000 = 10,000 units
        self.demand_p10_head.bias.data.fill_(0.15)  # 15% lower
        self.demand_p90_head.bias.data.fill_(0.15)  # 15% higher
        self.utilization_head.bias.data.fill_(0.85)  # 85% utilization

    def forward(
        self,
        cat_features: torch.Tensor,
        cont_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        cat_features: Tensor of shape (B, 3) [business_line_idx, region_idx, factory_idx]
        cont_features: Tensor of shape (B, num_continuous_features)
        """
        b_idx, r_idx, f_idx = cat_features[:, 0], cat_features[:, 1], cat_features[:, 2]
        emb_b = self.embeddings["business_line"](b_idx)
        emb_r = self.embeddings["region"](r_idx)
        emb_f = self.embeddings["factory"](f_idx)

        combined_input = torch.cat([cont_features, emb_b, emb_r, emb_f], dim=1)
        features = self.backbone(combined_input)

        # Ensure quantile monotonicity by predicting positive offsets:
        # Scale by 1000 to match raw units (B2B scale)
        p50 = F.softplus(self.demand_p50_head(features)) * 1000.0
        p10_offset = F.softplus(self.demand_p10_head(features)) * 1000.0
        p90_offset = F.softplus(self.demand_p90_head(features)) * 1000.0

        p10 = torch.clamp(p50 - p10_offset, min=0.0)
        p90 = p50 + p90_offset

        utilization = F.relu(self.utilization_head(features)) * 100.0
        overload_logits = self.overload_head(features)
        shortage_logits = self.shortage_head(features)

        return {
            "p10": p10.squeeze(-1),
            "p50": p50.squeeze(-1),
            "p90": p90.squeeze(-1),
            "utilization": utilization.squeeze(-1),
            "overload_logits": overload_logits.squeeze(-1),
            "shortage_logits": shortage_logits.squeeze(-1),
            "overload_prob": torch.sigmoid(overload_logits).squeeze(-1),
            "shortage_prob": torch.sigmoid(shortage_logits).squeeze(-1),
        }


class ModelPredictor:
    """High-level wrapper for CUDA inference, scenario simulation, and preprocessing."""

    CONT_FEATURES = [
        "hist_orders_m3",
        "hist_orders_m2",
        "hist_orders_m1",
        "market_index",
        "backlog_units",
        "unweighted_pipeline_usd",
        "stale_deals_usd",
        "active_pipeline_usd",
        "unit_price_usd",
        "effective_capacity",
        "surge_pct",
        "mat_A1_req",
        "mat_B2_req",
        "mat_C3_req",
        "mat_A1_initial_stock",
        "mat_B2_initial_stock",
        "mat_C3_initial_stock",
        "mat_A1_in_transit",
        "mat_B2_in_transit",
        "mat_C3_in_transit",
    ]

    BL_MAP = {
        "Analytical": 0,
        "Force and Measurement": 1,
        "Instrumentation": 2,
    }

    REGION_MAP = {
        "Germany": 0,
        "United States": 1,
        "Japan": 2,
        "Poland": 3,
        "Mexico": 4,
    }

    FACTORY_MAP = {
        "Factory A (Stuttgart)": 0,
        "Factory B (Detroit)": 1,
        "Factory C (Yokohama)": 2,
        "Factory D (Wroclaw)": 3,
        "Factory E (Monterrey)": 4,
    }

    def __init__(self, model_path: Optional[str] = None):
        self.gpu_manager = get_gpu_manager()
        self.device = self.gpu_manager.get_device()
        self.model = EnterpriseSurrogateNet(num_continuous_features=len(self.CONT_FEATURES))

        # Fit scaling stats placeholders
        self.cont_mean = np.zeros(len(self.CONT_FEATURES), dtype=np.float32)
        self.cont_std = np.ones(len(self.CONT_FEATURES), dtype=np.float32)

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.model.to(self.device)

    def set_scaler_params(self, mean: np.ndarray, std: np.ndarray):
        """Sets z-score standardization statistics."""
        self.cont_mean = mean.astype(np.float32)
        self.cont_std = np.where(std == 0, 1.0, std).astype(np.float32)

    def save(self, filepath: str):
        """Saves model weights and scaler statistics."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "cont_mean": self.cont_mean,
            "cont_std": self.cont_std,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")

    def load(self, filepath: str):
        """Loads model weights and scaler statistics."""
        if not os.path.exists(filepath):
            return
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.cont_mean = checkpoint["cont_mean"]
        self.cont_std = checkpoint["cont_std"]
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {filepath} on device {self.device}")

    def preprocess_df(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses a DataFrame into CUDA tensors."""
        b_idx = df["business_line"].map(self.BL_MAP).fillna(0).values.astype(np.int64)
        r_idx = df["region"].map(self.REGION_MAP).fillna(0).values.astype(np.int64)
        f_idx = df["factory"].map(self.FACTORY_MAP).fillna(0).values.astype(np.int64)
        cat_arr = np.stack([b_idx, r_idx, f_idx], axis=1)

        cont_arr = df[self.CONT_FEATURES].values.astype(np.float32)
        cont_norm = (cont_arr - self.cont_mean) / self.cont_std

        cat_tensor = torch.tensor(cat_arr, dtype=torch.long, device=self.device)
        cont_tensor = torch.tensor(cont_norm, dtype=torch.float32, device=self.device)
        return cat_tensor, cont_tensor

    @torch.no_grad()
    def predict_batch(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Runs fast GPU batch inference for a given DataFrame."""
        self.model.eval()
        cat_t, cont_t = self.preprocess_df(df)
        outputs = self.model(cat_t, cont_t)

        return {
            "p10": outputs["p10"].cpu().numpy(),
            "p50": outputs["p50"].cpu().numpy(),
            "p90": outputs["p90"].cpu().numpy(),
            "utilization": outputs["utilization"].cpu().numpy(),
            "overload_prob": outputs["overload_prob"].cpu().numpy(),
            "shortage_prob": outputs["shortage_prob"].cpu().numpy(),
        }

    @torch.no_grad()
    def predict_single(self, row_dict: Dict[str, Any]) -> Dict[str, float]:
        """Predicts single row instance."""
        df = pd.DataFrame([row_dict])
        preds = self.predict_batch(df)
        return {k: float(v[0]) for k, v in preds.items()}

"""
PlanFlow AI - GPU Hardware Acceleration Manager
Optimized for NVIDIA GeForce RTX 5060 Laptop GPU (CUDA).
Provides device allocation, VRAM telemetry, FP16/mixed precision management, and tensor utilities.
"""

import os
from typing import Dict, Any, Optional
import torch
from src.core.logger import get_logger

logger = get_logger("GPUManager")


class GPUManager:
    """Manages GPU resource allocation, CUDA streams, and device synchronization."""

    _instance: Optional["GPUManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.device("cuda:0")
            self.device_name = torch.cuda.get_device_name(0)
            self.compute_capability = torch.cuda.get_device_capability(0)
            # Set cuDNN benchmark for optimized convolutions / linear layers
            torch.backends.cudnn.benchmark = True
            logger.info(
                f"[bold green]GPU Acceleration Active[/bold green]: {self.device_name} "
                f"(Compute Capability {self.compute_capability[0]}.{self.compute_capability[1]})"
            )
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU (Fallback)"
            self.compute_capability = (0, 0)
            logger.warning("[yellow]CUDA not detected or unavailable. Falling back to CPU execution.[/yellow]")

    def get_device(self) -> torch.device:
        """Returns the active PyTorch compute device."""
        return self.device

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns current GPU VRAM utilization and device status."""
        if not self.cuda_available:
            return {
                "device": "CPU",
                "cuda_available": False,
                "vram_allocated_mb": 0.0,
                "vram_reserved_mb": 0.0,
                "vram_total_mb": 0.0,
            }

        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        return {
            "device": self.device_name,
            "cuda_available": True,
            "vram_allocated_mb": round(allocated, 2),
            "vram_reserved_mb": round(reserved, 2),
            "vram_total_mb": round(total, 2),
            "vram_utilization_pct": round((allocated / total) * 100, 2),
        }

    def clear_cache(self):
        """Releases cached memory on CUDA device."""
        if self.cuda_available:
            torch.cuda.empty_cache()

    def to_device(self, tensor_or_module: Any) -> Any:
        """Transfers tensor or PyTorch nn.Module to the active device."""
        return tensor_or_module.to(self.device)


def get_gpu_manager() -> GPUManager:
    """Convenience accessor for singleton GPUManager."""
    return GPUManager()

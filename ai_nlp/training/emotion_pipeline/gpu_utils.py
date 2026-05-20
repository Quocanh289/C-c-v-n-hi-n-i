"""
GPU Utilities Module
=====================
Hardware utilities for optimal GPU usage:
- Device detection (CUDA, MPS, CPU)
- Mixed precision training
- Memory optimization
- Gradient scaling
"""

import os
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"CUDA memory: {props.total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def optimize_memory(device: torch.device):
    """
    Apply memory optimizations based on device.
    """
    if device.type == "cuda":
        # Enable TF32 for Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Ampere GPU")
        
        # Set cuDNN benchmark for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter:
    """Displays training progress."""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmt = self._get_batch_fmt(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def _get_batch_fmt(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = f"[{{:>{num_digits}}}/{num_digits > 0 and num_batches or 1}]"
        return fmt
    
    def display(self, batch: int):
        entries = [f"{self.prefix} Batch {self.batch_fmt.format(batch)}"]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))
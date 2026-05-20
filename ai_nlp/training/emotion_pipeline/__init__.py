"""
Emotion Lens - GoEmotions Training Pipeline
=============================================

Production-grade training pipeline for fine-tuning XLM-RoBERTa 
on the GoEmotions dataset with:

- 27 fine-grained → 9 coarse emotion mapping
- Mixed precision training (FP16)
- Advanced loss functions (Focal Loss, ASL)
- Comprehensive metrics (per-class, macro, weighted)
- Threshold optimization for best F1
- Model checkpointing & early stopping
- Gradient accumulation
- Data augmentation (EDA for low-resource classes)

Usage:
    from emotion_pipeline import train_pipeline
    train_pipeline()
    
Or via CLI:
    python -m ai_nlp.training.emotion_pipeline.run
"""

from .config import TrainingConfig, GoEmotionsConfig
from .dataset import GoEmotionsDataset, get_kaggle_goemotions, EmotionLabelMapper
from .model import GoEmotionsModel, get_model_and_tokenizer
from .losses import FocalLoss, ASLLoss, CombinedLoss
from .metrics import EmotionMetrics, compute_all_metrics
from .trainer import Trainer, TrainingResult
from .gpu_utils import get_device, optimize_memory
from .threshold_optimizer import ThresholdOptimizer
from .pipeline import train_pipeline

__version__ = "1.0.0"
__all__ = [
    "TrainingConfig",
    "GoEmotionsConfig",
    "GoEmotionsDataset",
    "get_kaggle_goemotions",
    "EmotionLabelMapper",
    "GoEmotionsModel",
    "get_model_and_tokenizer",
    "FocalLoss",
    "ASLLoss",
    "CombinedLoss",
    "EmotionMetrics",
    "compute_all_metrics",
    "Trainer",
    "TrainingResult",
    "get_device",
    "optimize_memory",
    "ThresholdOptimizer",
    "train_pipeline",
]
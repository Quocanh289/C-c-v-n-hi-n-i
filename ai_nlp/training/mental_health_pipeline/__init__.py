"""
Mental Health Text Classification Pipeline
=============================================
Production-grade training pipeline for detecting 7 mental health conditions
from Reddit text using Transformer models.

Labels:
- Normal, Depression, Anxiety, Bipolar, Stress, Suicidal, Personality_disorder

Architecture:
- DeBERTa-v3-base + LoRA (Parameter Efficient Fine-Tuning)
- Custom PyTorch training loop with mixed precision (fp16)
- Focal Loss for class imbalance
- Gradient accumulation + checkpointing for RTX 4060 16GB
- Per-class threshold optimization with Optuna
- Comprehensive evaluation with macro F1, confusion matrix, per-class analysis

Reference:
  Dataset: https://www.kaggle.com/datasets/maazkareem/sentiment-and-mental-health-dataset-reddit-based
"""

from .config import (
    MENTAL_HEALTH_LABELS,
    MENTAL_HEALTH_LABELS_TO_IDX,
    IDX_TO_MH_LABELS,
    TrainingConfig,
    MHConfig,
)
from .preprocessor import RedditTextPreprocessor
from .dataset import MentalHealthDataset, create_dataloaders
from .model import (
    MentalHealthClassifier,
    EnsembleMentalHealthClassifier,
)
from .losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    PerClassGammaFocalLoss,
    ConfusionFocalLoss,
    LabelSmoothingCrossEntropy,
)
from .metrics import MentalHealthMetrics
from .trainer import MentalHealthTrainer
from .pipeline import MentalHealthPipeline

__version__ = "1.0.0"
__all__ = [
    "MENTAL_HEALTH_LABELS",
    "MENTAL_HEALTH_LABELS_TO_IDX",
    "IDX_TO_MH_LABELS",
    "TrainingConfig",
    "MHConfig",
    "RedditTextPreprocessor",
    "MentalHealthDataset",
    "create_dataloaders",
    "MentalHealthClassifier",
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "MentalHealthMetrics",
    "MentalHealthTrainer",
    "MentalHealthPipeline",
]
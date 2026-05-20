"""
Emotion Lens - GoEmotions Multi-Label Training Pipeline
========================================================

Production-grade training pipeline for fine-tuning transformer models (XLM-RoBERTa, RoBERTa)
on the GoEmotions dataset with true multi-label support (28 labels: 27 emotions + neutral).

Key Features:
- True multi-label classification preserving all 28 GoEmotions labels
- Dual-head architecture: 28 fine-grained + 9 coarse aggregated outputs
- Asymmetric Loss (ASL) for multi-label class imbalance
- Reddit/internet text preprocessing (emoji, slang, repeated chars)
- LoRA fine-tuning with gradient checkpointing
- Per-label threshold optimization
- Comprehensive multi-label metrics (macro/micro F1, Hamming loss, ECE)
- Overfitting & data leakage detection
- RoBERTa vs XLM-RoBERTa comparison utilities
- ONNX export support

Usage:
    from emotion_pipeline import train_pipeline
    train_pipeline()
    
Or via CLI:
    python -m ai_nlp.training.emotion_pipeline.run --help
    
Config-driven:
    from emotion_pipeline.config import TrainingConfig
    config = TrainingConfig(
        task_type="multi_label",
        model_name="FacebookAI/xlm-roberta-base",
        use_lora=True,
        loss_type="asl",
    )
    result = train_pipeline(config=config)
"""

from .config import (
    TrainingConfig,
    GoEmotionsConfig,
    GOEMOTIONS_28,
    COARSE_EMOTIONS,
    GOEMOTIONS_28_TO_IDX,
    COARSE_TO_IDX,
    AGGREGATION_MATRIX,
)
from .dataset import (
    GoEmotionsDataset,
    get_kaggle_goemotions,
    MultiLabelDataCollator,
    TextAugmenter,
    create_multi_label_loaders,
)
from .model import GoEmotionsModel, get_model_and_tokenizer, ModelComparisonResult
from .losses import (
    MultiLabelBCEWithLogitsLoss,
    MultiLabelFocalLoss,
    AsymmetricLoss,
    CombinedLoss,
    get_loss_function,
)
from .metrics import (
    EmotionMetrics,
    compute_multi_label_metrics,
    compute_all_metrics,
    detect_overfitting,
    detect_data_leakage,
    detect_distribution_shift,
)
from .trainer import Trainer, TrainingResult
from .gpu_utils import get_device, optimize_memory
from .threshold_optimizer import ThresholdOptimizer
from .pipeline import train_pipeline, evaluate_model
from .preprocessing import (
    RedditTextNormalizer,
    MultiLabelPreprocessor,
    RobustnessEvaluator,
)

__version__ = "2.0.0"
__all__ = [
    # Config
    "TrainingConfig",
    "GoEmotionsConfig",
    "GOEMOTIONS_28",
    "COARSE_EMOTIONS",
    "GOEMOTIONS_28_TO_IDX",
    "COARSE_TO_IDX",
    "AGGREGATION_MATRIX",
    # Dataset
    "GoEmotionsDataset",
    "get_kaggle_goemotions",
    "MultiLabelDataCollator",
    "TextAugmenter",
    "create_multi_label_loaders",
    # Model
    "GoEmotionsModel",
    "get_model_and_tokenizer",
    "ModelComparisonResult",
    # Losses
    "MultiLabelBCEWithLogitsLoss",
    "MultiLabelFocalLoss",
    "AsymmetricLoss",
    "CombinedLoss",
    "get_loss_function",
    # Metrics
    "EmotionMetrics",
    "compute_multi_label_metrics",
    "compute_all_metrics",
    "detect_overfitting",
    "detect_data_leakage",
    "detect_distribution_shift",
    # Training
    "Trainer",
    "TrainingResult",
    "train_pipeline",
    "evaluate_model",
    # Utils
    "get_device",
    "optimize_memory",
    "ThresholdOptimizer",
    # Preprocessing
    "RedditTextNormalizer",
    "MultiLabelPreprocessor",
    "RobustnessEvaluator",
]
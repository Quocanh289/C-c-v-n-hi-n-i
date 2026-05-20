"""
Configuration Module
=====================
Central configuration for the GoEmotions training pipeline.
All hyperparameters, paths, and model settings are defined here.
"""

import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple


# ============================================================
# GoEmotions 27 → 9 Emotion Mapping
# ============================================================

# The GoEmotions dataset has 27 fine-grained emotions.
# We map them to 9 coarse groups for the multi-task emotion model.

GOEMOTIONS_27 = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise",
]

# Mapping from GoEmotions 27 labels to 9 coarse emotions
EMOTION_27_TO_9_MAP = {
    "admiration": "admiration",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "admiration",
    "caring": "love",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "admiration",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "admiration",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "anxiety",
    "optimism": "admiration",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
}

# The 9 coarse emotion labels used in the project
COARSE_EMOTIONS = [
    "admiration", "anger", "anxiety", "fear", "joy", "love", "sadness", "surprise", "neutral"
]

# Neutral is the "no emotion" category in GoEmotions
NEUTRAL_LABEL_IDX = 8  # Position in COARSE_EMOTIONS

COARSE_TO_IDX = {label: idx for idx, label in enumerate(COARSE_EMOTIONS)}
IDX_TO_COARSE = {idx: label for idx, label in enumerate(COARSE_EMOTIONS)}


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """Master configuration for the complete training pipeline."""
    
    # ---------- Dataset ----------
    data_dir: str = os.path.join("ai_nlp", "training", "data")
    output_dir: str = os.path.join("ai_nlp", "training", "outputs")
    checkpoint_dir: str = os.path.join("ai_nlp", "training", "checkpoints")
    log_dir: str = os.path.join("ai_nlp", "training", "logs")
    
    # Kaggle dataset info
    kaggle_dataset: str = "debarshichanda/goemotions"
    use_kaggle: bool = True
    
    # ---------- Model ----------
    model_name: str = "xlm-roberta-base"
    # Alternatives: "microsoft/deberta-v3-base", "FacebookAI/xlm-roberta-base", "bert-base-multilingual-cased"
    num_labels: int = 9  # 8 emotion + 1 neutral
    max_seq_length: int = 128
    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # ---------- LoRA ----------
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # LoRA target modules
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value", "key", "output.dense"])
    
    # ---------- Training Hyperparameters ----------
    num_epochs: int = 15
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If 0, calculated from warmup_ratio * total_steps
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    
    # ---------- Scheduler ----------
    scheduler: str = "cosine"  # "linear", "cosine", "cosine_with_restarts"
    num_cycles: float = 0.5
    
    # ---------- Loss ----------
    primary_loss: str = "focal"  # "bce", "focal", "asl", "combined"
    secondary_loss: str = "bce"
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    focal_gamma: float = 2.0
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    label_smoothing: float = 0.1
    class_weights: bool = True  # Compute inverse frequency weights
    
    # ---------- Optimization ----------
    mixed_precision: str = "fp16"  # "fp16", "bf16", "no"
    use_ema: bool = False  # Exponential Moving Average
    ema_decay: float = 0.999
    use_fgm: bool = False  # Fast Gradient Method adversarial training
    fgm_epsilon: float = 0.5
    use_rdrop: bool = False  # R-Drop regularization
    rdrop_alpha: float = 4.0
    compile: bool = False  # torch.compile (PyTorch 2.0+)
    gradient_checkpointing: bool = True
    
    # ---------- Early Stopping ----------
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # ---------- Evaluation ----------
    eval_strategy: str = "epoch"  # "epoch", "steps"
    eval_steps: int = 100
    logging_steps: int = 10
    save_strategy: str = "epoch"  # "epoch", "steps", "best"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    
    # ---------- Data Augmentation ----------
    use_augmentation: bool = True
    aug_synonym_prob: float = 0.3
    aug_random_swap: int = 2
    aug_random_delete_prob: float = 0.1
    aug_back_translate: bool = False  # Requires API calls
    
    # ---------- Threshold Optimization ----------
    optimize_thresholds: bool = True
    threshold_optimization_metric: str = "macro_f1"
    threshold_n_trials: int = 100
    
    # ---------- Reproducibility ----------
    seed: int = 42
    
    # ---------- Hardware ----------
    num_workers: int = 2  # DataLoader workers
    
    # ---------- Paths (computed) ----------
    
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_checkpoint_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, "emotion_model")
    
    def get_output_model_dir(self) -> str:
        return os.path.join(self.output_dir, "emotion_model")
    
    def save(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GoEmotionsConfig:
    """Configuration specific to GoEmotions dataset processing."""
    
    # Labels
    num_fine_emotions: int = 27
    num_coarse_emotions: int = 9
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Kaggle CSV columns
    text_column: str = "text"
    label_columns: List[str] = field(default_factory=lambda: [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise",
    ])
    
    # Filter: remove samples with "neutral" (no emotion) to focus on emotional ones
    # Set to False to include neutral samples
    filter_neutral: bool = True
    
    # Minimum samples per class threshold (for validation)
    min_samples_per_class: int = 5
    
    # Maximum samples to use (for quick testing, None = all)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
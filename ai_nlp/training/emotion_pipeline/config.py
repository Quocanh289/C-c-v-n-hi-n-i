"""
Configuration Module
=====================
Central configuration for the GoEmotions multi-label emotion detection pipeline.
All hyperparameters, paths, and model settings are defined here.

Architecture:
- True multi-label classification over 28 labels (27 GoEmotions + neutral)
- Dual-head output: 28 fine-grained + 9 coarse aggregated
- Supports both RoBERTa-base and XLM-RoBERTa-base comparisons
"""

import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Literal


# ============================================================
# GoEmotions Label Definitions
# ============================================================

# The full 28-label GoEmotions set (27 emotions + "neutral")
GOEMOTIONS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

GOEMOTIONS_28_TO_IDX = {l: i for i, l in enumerate(GOEMOTIONS_28)}
GOEMOTIONS_IDX_TO_28 = {i: l for i, l in enumerate(GOEMOTIONS_28)}

# 9 coarse emotions for backward compatibility
COARSE_EMOTIONS = [
    "admiration", "anger", "anxiety", "fear", "joy",
    "love", "sadness", "surprise", "neutral"
]
COARSE_TO_IDX = {label: idx for idx, label in enumerate(COARSE_EMOTIONS)}
IDX_TO_COARSE = {idx: label for idx, label in enumerate(COARSE_EMOTIONS)}
NEUTRAL_LABEL_IDX = 8  # Position in COARSE_EMOTIONS

# 27 fine → 9 coarse emotion mapping (aggregation rules)
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

# 28-label to 9-coarse aggregation matrix (used in dual-head model)
# Shape: [9_coarse, 28_fine] — row-normalized averages
def _build_aggregation_matrix() -> torch.Tensor:
    mat = torch.zeros((len(COARSE_EMOTIONS), len(GOEMOTIONS_28)), dtype=torch.float32)
    for i, fine_label in enumerate(GOEMOTIONS_28):
        if fine_label == "neutral":
            mat[COARSE_TO_IDX["neutral"], i] = 1.0
        elif fine_label in EMOTION_27_TO_9_MAP:
            coarse = EMOTION_27_TO_9_MAP[fine_label]
            mat[COARSE_TO_IDX[coarse], i] = 1.0
    # Row-normalize (each coarse class gets equal weight from its fine sub-classes)
    row_sums = mat.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    return mat

AGGREGATION_MATRIX = _build_aggregation_matrix()

# Priority order if multi-label → single-label fallback needed
PRIORITY_ORDER = [
    "anger", "sadness", "fear", "anxiety", "surprise",
    "joy", "love", "admiration", "neutral"
]

# Label groups for analysis
NEGATIVE_EMOTIONS = {"anger", "annoyance", "disappointment", "disapproval", "disgust",
                     "fear", "grief", "nervousness", "remorse", "sadness"}
POSITIVE_EMOTIONS = {"admiration", "amusement", "approval", "caring", "desire",
                     "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
AMBIGUOUS_EMOTIONS = {"confusion", "curiosity", "embarrassment", "realization", "surprise"}

# Standard GoEmotions CSV column names (Kaggle format)
GOEMOTIONS_CSV_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """Master configuration for the complete multi-label training pipeline."""

    # ---------- Task Type ----------
    task_type: str = "multi_label"  # "multi_label" (28), "multi_class" (9), "dual_head" (28+9)
    
    # ---------- Dataset ----------
    data_dir: str = os.path.join("ai_nlp", "training", "data")
    output_dir: str = os.path.join("ai_nlp", "training", "outputs")
    checkpoint_dir: str = os.path.join("ai_nlp", "training", "checkpoints")
    log_dir: str = os.path.join("ai_nlp", "training", "logs")
    
    # Kaggle dataset info
    kaggle_dataset: str = "debarshichanda/goemotions"
    use_kaggle: bool = True
    
    # ---------- Model ----------
    model_name: str = "FacebookAI/xlm-roberta-base"
    # Alternatives: "roberta-base", "FacebookAI/xlm-roberta-base"
    # Comparison mode: set model_name_alt to compare
    model_name_alt: Optional[str] = "roberta-base"
    
    num_labels: int = 28  # 27 emotions + neutral (full GoEmotions)
    num_coarse_labels: int = 9  # Coarse emotions for backward compat
    
    max_seq_length: int = 128
    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Aggregation: how to produce 9-class from outputs
    # "learned": separate coarse head, "projected": matmul with AGGREGATION_MATRIX
    aggregation_method: str = "projected"
    
    # ---------- LoRA ----------
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value", "key", "output.dense"])
    
    # ---------- Training Hyperparameters ----------
    num_epochs: int = 30
    batch_size: int = 16  # RTX 4060 8GB: batch 16 fits with LoRA
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If 0, calculated from warmup_ratio * total_steps
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2  # Effective batch = 32 with 8GB VRAM
    
    # ---------- Scheduler ----------
    scheduler: str = "cosine"  # "linear", "cosine", "cosine_with_restarts"
    num_cycles: float = 0.5
    
    # ---------- Loss ----------
    # For multi-label: "bce", "focal", "asl", "combined"
    # For multi-class: "ce", "focal"
    loss_type: str = "asl"  # Asymmetric Loss is best for multi-label with imbalance
    primary_loss: str = "asl"
    secondary_loss: str = "bce"
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None
    asl_gamma_neg: float = 4.0  # Down-weight easy negatives heavily
    asl_gamma_pos: float = 0.0  # Don't down-weight positives
    asl_clip: float = 0.05
    label_smoothing: float = 0.0  # No smoothing for multi-label with ASL
    class_weights: bool = True
    
    # ---------- Optimization ----------
    mixed_precision: str = "fp16"  # Use fp16 on GPU for 2x speedup
    use_ema: bool = False
    ema_decay: float = 0.999
    use_fgm: bool = False
    fgm_epsilon: float = 0.5
    use_rdrop: bool = False
    rdrop_alpha: float = 4.0
    compile: bool = False
    gradient_checkpointing: bool = True  # Saves VRAM on 8GB GPU
    
    # ---------- Early Stopping ----------
    early_stopping_patience: int = 7
    early_stopping_threshold: float = 0.001
    # Metric to monitor for multi-label
    early_stopping_metric: str = "macro_f1_micro_avg"  # Use micro-averaged F1 for multi-label
    
    # ---------- Evaluation ----------
    eval_strategy: str = "epoch"
    eval_steps: int = 100
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1_micro_avg"  # Multi-label macro F1
    greater_is_better: bool = True
    
    # ---------- Data Augmentation ----------
    use_augmentation: bool = True  # Re-enabled with GPU
    aug_synonym_prob: float = 0.3
    aug_random_swap: int = 2
    aug_random_delete_prob: float = 0.1
    
    # ---------- Preprocessing ----------
    normalize_unicode: bool = True
    handle_emojis: bool = True
    handle_repeated_chars: bool = True
    handle_urls: bool = True
    
    # ---------- Threshold Optimization ----------
    optimize_thresholds: bool = True  # Re-enabled with GPU
    threshold_optimization_metric: str = "macro_f1"
    threshold_n_trials: int = 100
    
    # ---------- Reproducibility ----------
    seed: int = 42
    
    # ---------- Hardware ----------
    num_workers: int = 4  # 4 workers for GPU data loading
    
    # ---------- Class Balancing ----------
    use_balanced_sampling: bool = True
    balancing_strategy: str = "labels"
    
    # ---------- Paths (computed) ----------

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def num_outputs(self) -> int:
        """Number of output logits based on task type."""
        if self.task_type == "multi_label":
            return self.num_labels  # 28
        elif self.task_type == "multi_class":
            return self.num_coarse_labels  # 9
        elif self.task_type == "dual_head":
            return self.num_labels + self.num_coarse_labels  # 28 + 9 = 37
        return self.num_labels
    
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

    num_fine_emotions: int = 28  # 27 + neutral
    num_coarse_emotions: int = 9
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    text_column: str = "text"
    label_columns: List[str] = field(default_factory=lambda: GOEMOTIONS_CSV_COLUMNS)
    
    # Whether to include neutral samples (YES - neutral is one of 28 outputs)
    filter_neutral: bool = False
    
    min_samples_per_class: int = 5
    
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    
    # How to handle overlap between fine and coarse
    # "both": output both 28 & 9, "fine_only": output 28 only
    output_mode: str = "both"
    
    # Minimum label frequency to keep (for filtering rare labels)
    min_label_frequency: int = 10


# ============================================================
# Model Comparison Configuration
# ============================================================

@dataclass
class ModelComparisonConfig:
    """Configuration for comparing RoBERTa-base vs XLM-RoBERTa-base."""
    
    models_to_compare: List[str] = field(default_factory=lambda: [
        "roberta-base",
        "FacebookAI/xlm-roberta-base",
    ])
    
    metrics_to_compare: List[str] = field(default_factory=lambda: [
        "macro_f1_micro_avg", "micro_f1", "hamming_loss", "subset_accuracy"
    ])
    
    # Test on internet language robustness
    test_robustness_samples: List[str] = field(default_factory=lambda: [
        "bro cooked 💀",
        "nah this is insane 😭",
        "W take honestly",
        "i'm so done bro",
        "great... just great 🙂",
        "this is lit af 🔥",
        "no cap fr fr",
        "im dead 💀💀💀",
        "that's based as hell",
        "ong this is fire",
    ])
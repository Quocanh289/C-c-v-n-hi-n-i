"""
Configuration Module
=====================
Central configuration for mental health text classification pipeline.
All hyperparameters, paths, labels, and model settings are defined here.

Designed for:
  - GPU: RTX 4060 16GB VRAM
  - Model: DeBERTa-v3-base + LoRA
  - Dataset: Kaggle Reddit mental health (26K samples, 7 classes)

Label distribution (imbalanced):
  Suicidal:            5600  (21.2%)
  Anxiety:             4704  (17.9%)
  Normal:              4575  (17.4%)
  Depression:          4111  (15.6%)
  Personality_disorder: 2817  (10.7%)
  Bipolar:             2720  (10.3%)
  Stress:              1823  (6.9%)
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Literal


# ============================================================
# Mental Health Label Definitions
# ============================================================

MENTAL_HEALTH_LABELS = [
    "Normal",
    "Depression",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Suicidal",
    "Personality_disorder",
]

MENTAL_HEALTH_LABELS_TO_IDX = {l: i for i, l in enumerate(MENTAL_HEALTH_LABELS)}
IDX_TO_MH_LABELS = {i: l for i, l in enumerate(MENTAL_HEALTH_LABELS)}

NUM_MH_CLASSES = len(MENTAL_HEALTH_LABELS)  # 7

# Severity mapping for risk assessment
MH_SEVERITY_MAP = {
    "Normal": 0,                # healthy
    "Stress": 1,                # mild
    "Anxiety": 2,               # moderate
    "Personality_disorder": 2,  # moderate
    "Bipolar": 3,               # moderate-high
    "Depression": 4,            # severe
    "Suicidal": 5,              # critical
}

MH_SEVERITY_LABELS = {0: "healthy", 1: "mild", 2: "moderate", 3: "moderate-high", 4: "severe", 5: "critical"}

# Label counts from the dataset for computing class weights
MH_CLASS_COUNTS = {
    "Normal": 4575,
    "Depression": 4111,
    "Anxiety": 4704,
    "Bipolar": 2720,
    "Stress": 1823,
    "Suicidal": 5600,
    "Personality_disorder": 2817,
}

# Class weights = total_samples / (n_classes * class_count)  (balanced)
TOTAL_SAMPLES = sum(MH_CLASS_COUNTS.values())  # 26350
MH_CLASS_WEIGHTS = {
    label: TOTAL_SAMPLES / (NUM_MH_CLASSES * count)
    for label, count in MH_CLASS_COUNTS.items()
}
MH_CLASS_WEIGHTS_TENSOR = torch.tensor(
    [MH_CLASS_WEIGHTS[label] for label in MENTAL_HEALTH_LABELS],
    dtype=torch.float32,
)

# Keywords for fallback detection (only used when model unavailable)
MH_KEYWORDS = {
    "Suicidal": [
        "kill myself", "end my life", "want to die", "better off dead",
        "suicide", "no reason to live", "can't go on", "end it all",
        "wish i was dead", "take my own life", "suicidal", "ending it",
        "don't want to live", "i give up", "nothing matters anymore",
        "ready to die", "just want peace", "i want out",
    ],
    "Depression": [
        "depressed", "depression", "hopeless", "worthless", "empty",
        "miserable", "numb", "no energy", "can't get out of bed",
        "no motivation", "feel like a failure", "i hate myself", "meaningless",
        "nothing brings joy", "anhedonia", "tired all the time",
        "can't do anything", "what's the point", "feeling low",
        "down all the time", "sad all the time", "crying everyday",
    ],
    "Anxiety": [
        "anxious", "anxiety", "panic", "worried", "nervous", "overthinking",
        "can't stop thinking", "restless", "tense", "on edge", "dread",
        "scared", "terrified", "heart racing", "can't breathe", "panic attack",
        "racing thoughts", "constant worry", "social anxiety", "fearful",
        "paranoid", "overthink", "sweating", "shaking",
    ],
    "Bipolar": [
        "manic", "bipolar", "hypomania", "racing thoughts", "grandiose",
        "mood swings", "impulsive", "spending spree", "not sleeping",
        "too much energy", "rapid cycling", "mania", "hypomanic",
        "inappropriate laughter", "pressured speech", "flight of ideas",
    ],
    "Stress": [
        "stressed", "overwhelmed", "burnout", "so much to do", "pressure",
        "can't cope", "stressing out", "freaking out", "deadlines",
        "stressed out", "so much work", "exhausted", "burned out",
        "can't handle it", "too much pressure", "overworked",
    ],
    "Personality_disorder": [
        "bpd", "borderline", "personality disorder", "split",
        "abandonment issues", "unstable relationships", "identity disturbance",
        "impulsive behavior", "emotional dysregulation", "npd", "narcissistic",
        "dissociative", "depersonalization", "derealization",
        "avoidant personality", "dependent personality", "schizoid",
    ],
    "Normal": [],
}


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """Master configuration for mental health training pipeline."""

    # ---------- Task Type ----------
    task_type: str = "multi_class"  # 7-class single-label classification
    num_labels: int = NUM_MH_CLASSES  # 7

    # ---------- Dataset ----------
    data_path: str = os.path.join("ai_nlp", "training", "data", "mental_health")
    dataset_filename: str = "Sentiment_Mental_health_dataset.csv"
    output_dir: str = os.path.join("ai_nlp", "training", "outputs")
    checkpoint_dir: str = os.path.join("ai_nlp", "training", "checkpoints")
    log_dir: str = os.path.join("ai_nlp", "training", "logs")

    text_column: str = "statement"
    label_column: str = "status"

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42

    # ---------- Model ----------
    # DeBERTa-v3-base is RECOMMENDED over XLM-RoBERTa because:
    # 1. DeBERTa-v3 uses disentangled attention + enhanced masked decoder
    #    → better at capturing nuanced mental health language patterns
    # 2. DeBERTa-v3-base has 12 layers, 768 hidden, 12 heads → 184M params
    #    → comparable size to XLM-RoBERTa-base (279M) but MORE EFFICIENT
    # 3. DeBERTa-v3 consistently outperforms RoBERTa on GLUE/SuperGLUE
    # 4. For English-only mental health detection, XLM's multilingual
    #    advantage is unnecessary → DeBERTa is strictly better
    # 5. DeBERTa-v3-base uses ∼5.4GB VRAM with batch 16 + LoRA
    #    → fits well on RTX 4060 16GB
    #
    # XLM-RoBERTa-base alternative: "FacebookAI/xlm-roberta-base"
    #   - Good if you plan to extend to multilingual later (e.g., Vietnamese)
    #   - Slightly more VRAM (~6.2GB with same settings)
    #   - Slightly lower F1 on English-only tasks
    #
    # PhoBERT alternative for Vietnamese adaptation:
    #   - "vinai/phobert-base" for Vietnamese-only
    #   - "vinai/phobert-large" for higher accuracy
    #   - Requires Vietnamese tokenizer (replaces WordPiece with BPE-syllable)
    #   - Best practice: fine-tune PhoBERT separately on Vietnamese data
    model_name: str = "microsoft/deberta-v3-base"
    # Alternative: "FacebookAI/xlm-roberta-base" (for multilingual later)

    num_labels: int = 7

    max_seq_length: int = 256  # KEY: 256 > 128 for Reddit mental health posts
    """
    Why 256 tokens instead of 512?

    RTX 4060 16GB VRAM analysis:
    - DeBERTa-v3-base + LoRA with max_length=512:
      ~10.2GB VRAM for batch_size=8, grad_accum=4
      → Leaves only ~5.8GB for overhead, data, etc.
    - DeBERTa-v3-base + LoRA with max_length=256:
      ~5.8GB VRAM for batch_size=16, grad_accum=2
      → Leaves ~10.2GB for overhead → MUCH more stable
    - 512 tokens is inefficient because:
      a) Attention complexity is O(n²) → 512 is 4× more compute than 256
      b) Most Reddit mental health posts are 50-200 tokens
      c) The 99th percentile token length in this dataset is ~240 tokens
      d) Only ~2% of samples exceed 256 tokens (truncate those)
    - Recommend: max_length=256 with batch_size=16
    """

    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # ---------- LoRA Configuration ----------
    use_lora: bool = True
    lora_r: int = 8       # Low rank. 8 is good for 7 classes. 16 for more capacity.
    lora_alpha: int = 32  # Scaling = alpha / r = 4
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "query_proj", "value_proj", "key_proj", "output_proj",
        # DeBERTa uses 'query_proj', 'key_proj', 'value_proj', 'output_proj'
        # For XLMRoberta: ["query", "value", "key", "output.dense"]
    ])
    lora_bias: str = "none"    # Don't train bias for efficiency
    lora_task_type: str = "SEQ_CLS"

    # ---------- Training Hyperparameters ----------
    num_epochs: int = 20        # With early stopping, usually stops at 8-12
    batch_size: int = 16        # RTX 4060 16GB: 16 with DeBERTa-v3 + LoRA + 256 tokens
    eval_batch_size: int = 32
    learning_rate: float = 3e-5  # Slightly higher for DeBERTa-v3 (2e-5 for RoBERTa)
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2  # Effective batch = 32

    # ---------- Scheduler ----------
    scheduler: str = "cosine"  # Cosine with warmup restarts
    warmup_ratio: float = 0.1  # 10% of total steps for warmup
    warmup_steps: int = 0      # If 0, computed from warmup_ratio * total_steps

    # ---------- Loss ----------
    # Options:
    #   "ce"          — plain CrossEntropy
    #   "weighted_ce" — CrossEntropy with class weights (handle imbalance)
    #   "focal"       — Focal Loss (γ=2.0 focus on hard examples)
    #   "label_smooth_ce" — Label smoothing CE
    #   "confusion_focal" — Focal loss + confusion penalty for hard pairs
    loss_type: str = "confusion_focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # None = auto-compute from class counts
    label_smoothing: float = 0.1  # Used when loss_type="label_smooth_ce"
    
    # Per-class gamma for FocalLoss (higher = more focus on that class)
    # Labels: Normal, Depression, Anxiety, Bipolar, Stress, Suicidal, Personality_disorder
    # Higher gamma on Depression & Suicidal to fix the confusion matrix
    per_class_gamma: Optional[List[float]] = field(default_factory=lambda: [
        2.0,    # Normal
        3.5,    # Depression — most confused, highest focus
        2.5,    # Anxiety — moderately confused
        2.0,    # Bipolar
        2.0,    # Stress
        3.0,    # Suicidal — very confused with Depression
        2.0,    # Personality_disorder
    ])
    
    # Confusion penalty: weight for the auxiliary confusion-avoidance loss
    # Helps penalize dangerous misclassifications (e.g., Suicidal → Depression)
    confusion_penalty_weight: float = 0.3
    
    # Pairs to penalize: (true_class, wrong_class) pairs to discourage
    # Critical: Suicidal misclassified as anything else is very dangerous
    confusion_penalty_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (5, 1),  # Suicidal → Depression (50% of errors!)
        (5, 2),  # Suicidal → Anxiety
        (5, 0),  # Suicidal → Normal
        (1, 5),  # Depression → Suicidal
        (2, 1),  # Anxiety → Depression
        (1, 3),  # Depression → Bipolar
    ])

    # ---------- Optimization ----------
    mixed_precision: str = "fp16"  # "fp16", "bf16", "no"
    gradient_checkpointing: bool = True  # Saves ~30% VRAM at 15% speed cost
    use_compile: bool = False      # torch.compile (PyTorch 2.0+) - saves 5-10%

    # ---------- Early Stopping ----------
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    early_stopping_metric: str = "macro_f1"  # Monitor macro F1 on validation

    # ---------- Evaluation ----------
    eval_strategy: str = "epoch"        # "epoch", "steps"
    eval_steps: int = 50
    logging_steps: int = 10
    save_strategy: str = "epoch"        # "epoch", "steps", "best"
    save_total_limit: int = 3           # Keep only 3 best checkpoints
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True

    # ---------- Data Augmentation ----------
    use_augmentation: bool = True
    aug_synonym_prob: float = 0.2       # 20% chance of synonym replacement
    aug_random_swap: int = 2            # Max swaps per sample
    aug_random_delete_prob: float = 0.05  # 5% deletion noise

    # ---------- Class Imbalance Handling ----------
    use_balanced_sampling: bool = True
    balancing_strategy: str = "oversample"  # "oversample", "weights", "both"

    # ---------- Threshold Optimization ----------
    optimize_thresholds: bool = True
    threshold_optimization_metric: str = "macro_f1"
    threshold_n_trials: int = 50

    # ---------- Reproducibility ----------
    seed: int = 42

    # ---------- Hardware ----------
    num_workers: int = 4  # Adjust based on CPU cores (4 is safe for most)
    pin_memory: bool = True  # Faster GPU transfer

    # ---------- Paths (computed) ----------

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_output_name(self) -> str:
        return self.model_name.replace("/", "_").replace("-", "_") + "_mental_health"

    def get_checkpoint_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, "mental_health_model")

    def get_output_model_dir(self) -> str:
        return os.path.join(self.output_dir, "mental_health_model")

    def get_log_dir(self) -> str:
        return os.path.join(self.log_dir, "mental_health_model")

    def get_data_file_path(self) -> str:
        return os.path.join(self.data_path, self.dataset_filename)

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
class MHConfig:
    """Dataset-specific configuration for mental health data processing."""

    num_classes: int = NUM_MH_CLASSES  # 7

    label_column: str = "status"
    text_column: str = "statement"

    # Preprocessing
    min_text_length: int = 5           # Remove samples shorter than 5 chars
    max_text_length: int = 512         # Cap text length in characters
    remove_duplicates: bool = True
    remove_leakage: bool = True        # Remove texts that contain their label name

    # Label names that could leak (diagnosis words in text)
    leakage_words: List[str] = field(default_factory=lambda: [
        "depression", "anxiety", "bipolar", "suicidal",
        "personality disorder", "stress", "stressed",
        # But NOT "normal" — too common
    ])

    # Train/Val/Test split
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    stratify: bool = True  # Stratified split for class balance

    # Augmentation
    use_synonym_replacement: bool = True
    use_eda: bool = True               # Easy Data Augmentation
    eda_alpha: float = 0.1            # % of words to augment
    num_augmented_samples: int = 0    # 0 = auto (fill to match majority class)

    # Tokenization
    max_length: int = 256
    truncation: bool = True
    padding: str = "max_length"       # "max_length" for fixed-size batches

    # Threshold optimization
    default_threshold: float = 0.5

    # Experiment tracking
    experiment_name: str = "mental_health_classifier"
    tracking_uri: Optional[str] = None  # MLflow URI
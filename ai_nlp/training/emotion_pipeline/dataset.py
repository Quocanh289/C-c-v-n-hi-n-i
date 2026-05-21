"""
Dataset Module
===============
Multi-label GoEmotions dataset loader with:
- True multi-label support: preserves all 28 original labels per sample
- 9-coarse aggregation for backward compatibility
- Reddit/internet text preprocessing
- Data augmentation (EDA)
- Class weight computation for multi-label imbalance
- Balanced sampling strategies for multi-label data

GoEmotions CSV format (Kaggle): 28 one-hot columns + text column
Each sample can have multiple positive labels (multi-label).
"""

import os
import csv
import json
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Callable, Union, Set
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import (
    GOEMOTIONS_28,
    GOEMOTIONS_28_TO_IDX,
    GOEMOTIONS_IDX_TO_28,
    COARSE_EMOTIONS,
    COARSE_TO_IDX,
    GOEMOTIONS_CSV_COLUMNS,
    EMOTION_27_TO_9_MAP,
    AGGREGATION_MATRIX,
    TrainingConfig,
    GoEmotionsConfig,
)
from .preprocessing import RedditTextNormalizer, MultiLabelPreprocessor

logger = logging.getLogger(__name__)


# ============================================================
# Multi-Label GoEmotions Dataset
# ============================================================

class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions with true multi-label support.
    PRE-TOKENIZED: Tokenization happens once in __init__, NOT per-sample.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        config: TrainingConfig,
        goemotions_config: Optional[GoEmotionsConfig] = None,
        split: str = "train",
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.config = config
        self.go_config = goemotions_config or GoEmotionsConfig()
        self.split = split
        self.augment = augment and split == "train"
        self.task_type = config.task_type

        # Load CSV
        self.df = pd.read_csv(file_path)

        # Preprocessor
        self.preprocessor = MultiLabelPreprocessor(
            normalize_unicode=config.normalize_unicode,
            handle_emojis=config.handle_emojis,
            handle_repeated_chars=config.handle_repeated_chars,
            handle_urls=config.handle_urls,
        )

        # Process data: extract multi-label vectors
        texts, self.labels_28 = self._process_dataframe(self.df)

        # Compute coarse labels from fine labels
        self.labels_9 = self._aggregate_to_coarse(self.labels_28)

        # Limit samples if specified
        if max_samples and len(texts) > max_samples:
            indices = list(range(len(texts)))
            random.shuffle(indices)
            indices = indices[:max_samples]
            texts = [texts[i] for i in indices]
            self.labels_28 = [self.labels_28[i] for i in indices]
            self.labels_9 = [self.labels_9[i] for i in indices]

        # --- PRE-TOKENIZE everything at init (huge speedup) ---
        logger.info(f"Tokenizing {len(texts)} samples (max_length={config.max_seq_length})...")
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,  # dynamic padding across whole dataset
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        del texts  # free memory

        # Convert labels to tensors once
        for i in range(len(self.labels_28)):
            if isinstance(self.labels_28[i], list):
                self.labels_28[i] = torch.tensor(self.labels_28[i], dtype=torch.float32)
            if isinstance(self.labels_9[i], list):
                self.labels_9[i] = torch.tensor(self.labels_9[i], dtype=torch.float32)

        # Augmentation - simpler, just string-based at init
        self.augmenter = None
        if augment and config.use_augmentation:
            self.augmenter = TextAugmenter(
                synonym_prob=config.aug_synonym_prob,
                random_swap=config.aug_random_swap,
                random_delete_prob=config.aug_random_delete_prob,
            )
            # Pre-augment all texts (safer for tokenized data)
            # We store original texts for augmentation
            self.texts_for_aug = self.df["text"].tolist()

        # Compute class weights for multi-label
        self.class_weights_28 = None
        self.class_weights_9 = None
        if config.class_weights:
            self.class_weights_28 = self._compute_multi_label_class_weights(self.labels_28)
            self.class_weights_9 = self._compute_multi_label_class_weights(self.labels_9)

        logger.info(
            f"Loaded {split} set: {len(self.input_ids)} samples, "
            f"task={self.task_type}"
        )
        self._log_label_statistics()
    
    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Process DataFrame into texts and multi-hot label tensors.
        
        Supports multiple CSV formats:
        1. One-hot format: 28 label columns with 0/1 values
        2. Single "labels" column: comma-separated fine-grained indices
        3. Single "label" column: single fine-grained index
        """
        texts = []
        labels = []
        
        is_one_hot = any(col in df.columns for col in GOEMOTIONS_CSV_COLUMNS)
        
        for idx, row in df.iterrows():
            text = row.get("text", row.get(self.go_config.text_column, ""))
            if not text or not isinstance(text, str):
                continue
            
            # Normalize text
            text = self.preprocessor.process_text(text)
            
            # Extract multi-label vector
            if is_one_hot:
                multi_hot = self._parse_one_hot_row(row)
            elif "labels" in df.columns:
                multi_hot = self._parse_labels_column(row.get("labels", ""))
            elif "label" in df.columns:
                multi_hot = self._parse_single_label(row["label"])
            else:
                # Try to infer from any column that looks like a label
                multi_hot = self._infer_labels(row, df.columns)
            
            # Apply label frequency filter
            if self.go_config.min_label_frequency > 0:
                # Handled in _compute_label_frequencies
                pass
            
            texts.append(text)
            labels.append(multi_hot)
        
        return texts, labels
    
    def _parse_one_hot_row(self, row: pd.Series) -> List[int]:
        """Parse one-hot encoded row into 28-element multi-hot vector."""
        multi_hot = [0] * 28
        for col_idx, label_name in enumerate(GOEMOTIONS_CSV_COLUMNS):
            if col_idx < len(GOEMOTIONS_CSV_COLUMNS) and label_name in row:
                val = row[label_name]
                if val == 1 or (isinstance(val, str) and val.strip() == "1"):
                    multi_hot[col_idx] = 1
            elif label_name in GOEMOTIONS_28_TO_IDX:
                # Neutral column check
                if label_name == "neutral" and label_name in row:
                    if row[label_name] == 1:
                        multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        return multi_hot
    
    def _parse_labels_column(self, labels_str: str) -> List[int]:
        """Parse comma-separated label indices."""
        multi_hot = [0] * 28
        if not labels_str or labels_str == "nan":
            multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
            return multi_hot
        
        try:
            indices = [int(i.strip()) for i in labels_str.split(",") if i.strip()]
            for idx in indices:
                if 0 <= idx < 28:
                    multi_hot[idx] = 1
        except (ValueError, IndexError):
            multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        
        # If no labels, mark as neutral
        if sum(multi_hot) == 0:
            multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        
        return multi_hot
    
    def _parse_single_label(self, label_val) -> List[int]:
        """Parse a single label value."""
        multi_hot = [0] * 28
        try:
            idx = int(label_val)
            if 0 <= idx < 28:
                multi_hot[idx] = 1
            else:
                multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        except (ValueError, TypeError):
            multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        return multi_hot
    
    def _infer_labels(self, row: pd.Series, columns: List[str]) -> List[int]:
        """Infer label format from available columns."""
        multi_hot = [0] * 28
        
        # Check if any of our label columns exist
        found = False
        for col in columns:
            col_lower = col.lower().strip()
            if col_lower in GOEMOTIONS_28_TO_IDX:
                try:
                    val = float(row[col])
                    if val > 0.5:
                        multi_hot[GOEMOTIONS_28_TO_IDX[col_lower]] = 1
                        found = True
                except (ValueError, TypeError):
                    pass
        
        if not found:
            multi_hot[GOEMOTIONS_28_TO_IDX["neutral"]] = 1
        
        return multi_hot
    
    @staticmethod
    def _aggregate_to_coarse(labels_28: List[torch.Tensor]) -> List[torch.Tensor]:
        """Aggregate 28 fine labels to 9 coarse labels."""
        labels_9 = []
        for fine_vec in labels_28:
            if isinstance(fine_vec, list):
                fine_vec = torch.tensor(fine_vec, dtype=torch.float32)
            coarse = (AGGREGATION_MATRIX @ fine_vec.float()).clamp(0, 1)
            labels_9.append(coarse)
        return labels_9
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Labels already pre-converted to tensors
        label_28 = self.labels_28[idx]
        label_9 = self.labels_9[idx]
        
        # Tokenized data already pre-computed - just return it
        result = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        
        # Set labels based on task type
        if self.task_type == "multi_label":
            result["labels"] = label_28.float()  # [28] multi-hot
        elif self.task_type == "multi_class":
            result["labels"] = self._multi_label_to_single(label_28)
        elif self.task_type == "dual_head":
            result["labels_28"] = label_28.float()  # [28]
            result["labels_9"] = label_9.float()    # [9]
            result["labels"] = label_28.float()     # Primary for loss
        else:
            result["labels"] = label_28.float()
        
        return result
    
    def _multi_label_to_single(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """Convert multi-hot to single index for backward compatibility."""
        from .config import PRIORITY_ORDER, COARSE_TO_IDX
        
        active_labels = []
        for i, val in enumerate(multi_hot):
            if val > 0.5:
                label_name = GOEMOTIONS_IDX_TO_28.get(i, "neutral")
                active_labels.append(label_name)
        
        # Map to coarse priority
        for priority in PRIORITY_ORDER:
            if priority in active_labels or (
                priority == "neutral" and "neutral" in active_labels
            ):
                return torch.tensor(COARSE_TO_IDX[priority], dtype=torch.long)
        
        # Fine-to-coarse mapping fallback
        coarse_active = set()
        for label in active_labels:
            if label in EMOTION_27_TO_9_MAP:
                coarse_active.add(EMOTION_27_TO_9_MAP[label])
        
        for priority in PRIORITY_ORDER:
            if priority in coarse_active:
                return torch.tensor(COARSE_TO_IDX[priority], dtype=torch.long)
        
        return torch.tensor(COARSE_TO_IDX["neutral"], dtype=torch.long)
    
    def _compute_multi_label_class_weights(self, labels: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute per-label class weights for multi-label classification.
        
        Uses inverse frequency weighting with sqrt scaling
        to handle multi-label imbalance without over-penalizing.
        
        For multi-label, each label is treated independently:
        weight_i = sqrt(total_positive / positive_i)
        """
        # Detect dimension from actual data (handles both list and tensor)
        n_labels = len(GOEMOTIONS_28)  # default 28
        if len(labels) > 0:
            first = labels[0]
            if isinstance(first, list):
                n_labels = len(first)
            elif isinstance(first, torch.Tensor):
                n_labels = first.size(0)
            # Also store the expected n_labels for 9-class case
            # labels_9 tensors have size 9
        elif hasattr(self, 'labels_9') and labels is self.labels_9:
            n_labels = 9
        
        # Count positive occurrences per label
        positive_counts = torch.zeros(n_labels)
        for label_vec in labels:
            if isinstance(label_vec, list):
                vec = torch.tensor(label_vec)
            else:
                vec = label_vec
            positive_counts += vec.float()
        
        # Compute weights
        total_positives = positive_counts.sum()
        weights = torch.zeros(n_labels)
        
        for i in range(n_labels):
            count = positive_counts[i].item()
            if count > 0:
                # Inverse frequency with sqrt smoothing
                weight = total_positives / (count * n_labels)
                weights[i] = np.sqrt(weight)
            else:
                weights[i] = 1.0
        
        # Clamp extreme values
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        return weights
    
    def _log_label_statistics(self):
        """Log label distribution statistics."""
        n = len(self.input_ids)
        label_counts_28 = Counter()
        label_counts_9 = Counter()
        
        for i in range(n):
            for j, val in enumerate(self.labels_28[i]):
                if val > 0.5:
                    label_counts_28[GOEMOTIONS_IDX_TO_28[j]] += 1
            for j, val in enumerate(self.labels_9[i]):
                if val > 0.5:
                    label_counts_9[COARSE_EMOTIONS[j]] += 1
        
        logger.info(f"  Label distribution (28-class):")
        for label in GOEMOTIONS_28:
            count = label_counts_28.get(label, 0)
            pct = 100 * count / n if n > 0 else 0
            logger.info(f"    {label:20s}: {count:5d} ({pct:.1f}%)")
        
        logger.info(f"  Label distribution (9-class):")
        for label in COARSE_EMOTIONS:
            count = label_counts_9.get(label, 0)
            pct = 100 * count / n if n > 0 else 0
            logger.info(f"    {label:20s}: {count:5d} ({pct:.1f}%)")
    
    def get_class_distribution_28(self) -> Dict[str, int]:
        """Get 28-class label distribution."""
        label_counts = Counter()
        for label_vec in self.labels_28:
            for j, val in enumerate(label_vec):
                if val > 0.5:
                    label_counts[GOEMOTIONS_IDX_TO_28[j]] += 1
        return dict(sorted(label_counts.items()))
    
    def get_class_distribution_9(self) -> Dict[str, int]:
        """Get 9-class label distribution."""
        label_counts = Counter()
        for label_vec in self.labels_9:
            for j, val in enumerate(label_vec):
                if val > 0.5:
                    label_counts[COARSE_EMOTIONS[j]] += 1
        return dict(sorted(label_counts.items()))
    
    def get_label_frequencies_28(self) -> torch.Tensor:
        """Get per-label frequency (for loss weighting)."""
        freqs = torch.zeros(28)
        for label_vec in self.labels_28:
            freqs += torch.tensor(label_vec, dtype=torch.float32)
        return freqs / len(self.input_ids)


# ============================================================
# Data Collator for Dynamic Padding
# ============================================================

class MultiLabelDataCollator:
    """
    Ultra-fast data collator for pre-tokenized multi-label GoEmotions.
    Since data is pre-tokenized with fixed padding, just stack tensors.
    """
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        # All inputs are already padded identically - just stack them
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        
        # Labels
        if "labels_28" in features[0]:
            batch["labels_28"] = torch.stack([f["labels_28"] for f in features])
            batch["labels_9"] = torch.stack([f["labels_9"] for f in features])
            batch["labels"] = batch["labels_28"]
        elif "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        
        return batch


# ============================================================
# Data Loader Factory
# ============================================================

def create_multi_label_loaders(
    config: TrainingConfig,
    tokenizer,
    goemotions_config: Optional[GoEmotionsConfig] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[torch.Tensor]]:
    """
    Create train, validation, and test data loaders for multi-label GoEmotions.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    go_config = goemotions_config or GoEmotionsConfig()
    
    # Download or locate dataset
    data_paths = get_kaggle_goemotions(config)
    
    # Create datasets
    train_dataset = GoEmotionsDataset(
        file_path=data_paths["train"],
        tokenizer=tokenizer,
        config=config,
        goemotions_config=go_config,
        split="train",
        augment=config.use_augmentation,
        max_samples=go_config.max_train_samples,
    )
    
    val_dataset = GoEmotionsDataset(
        file_path=data_paths["val"],
        tokenizer=tokenizer,
        config=config,
        goemotions_config=go_config,
        split="val",
        augment=False,
        max_samples=go_config.max_val_samples,
    )
    
    test_dataset = None
    if os.path.exists(data_paths.get("test", "")):
        test_dataset = GoEmotionsDataset(
            file_path=data_paths["test"],
            tokenizer=tokenizer,
            config=config,
            goemotions_config=go_config,
            split="test",
            augment=False,
            max_samples=go_config.max_test_samples,
        )
    
    # Get class weights
    class_weights = train_dataset.class_weights_28
    
    # Create data collator
    collator = MultiLabelDataCollator()
    
    # Create train loader with optional balanced sampling
    if config.use_balanced_sampling and config.balancing_strategy == "samples":
        # Multi-label balanced sampling: weight by inverse of total positive labels
        sampler = _create_multi_label_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("Using multi-label balanced sampling for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    logger.info(
        f"Train: {len(train_dataset)} samples | "
        f"Val: {len(val_dataset)} samples"
    )
    if test_dataset:
        logger.info(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, class_weights


def _create_multi_label_sampler(dataset: GoEmotionsDataset) -> WeightedRandomSampler:
    """
    Create weighted sampler for multi-label data.
    Samples with fewer positive labels get higher weight
    (they are harder and rarer in multi-label setting).
    """
    weights = []
    for label_vec in dataset.labels_28:
        if isinstance(label_vec, list):
            vec = torch.tensor(label_vec)
        else:
            vec = label_vec
        n_pos = vec.sum().item()
        # Fewer positive labels = higher weight
        if n_pos > 0:
            weights.append(1.0 / n_pos)
        else:
            weights.append(1.0)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ============================================================
# Kaggle Dataset Downloader
# ============================================================

def get_kaggle_goemotions(config: TrainingConfig) -> Dict[str, str]:
    """
    Download the GoEmotions dataset from Kaggle if not already present.
    Falls back to generating a synthetic multi-label dataset for testing.
    Returns paths to train/val/test CSV files.
    """
    data_dir = os.path.join(config.data_dir, "goemotions")
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    # Check if files already exist
    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        logger.info(f"GoEmotions dataset already exists at {data_dir}")
        return {"train": train_path, "val": val_path, "test": test_path}
    
    # Try downloading from Kaggle
    if config.use_kaggle:
        dataset_downloaded = _try_download_via_kagglehub(config, data_dir)
        if not dataset_downloaded:
            dataset_downloaded = _try_download_via_opendatasets(data_dir)
        
        if dataset_downloaded:
            if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                return {"train": train_path, "val": val_path, "test": test_path}
    
    # Try finding pre-downloaded files
    alt_paths = [
        os.path.join("data", "goemotions"),
        os.path.join("data"),
        os.path.join("ai_nlp", "data"),
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            for fname in os.listdir(alt):
                if fname.endswith(".csv"):
                    alt_file = os.path.join(alt, fname)
                    logger.info(f"Found CSV at {alt_file}")
                    if fname in ["train.csv", "val.csv", "test.csv"]:
                        import shutil
                        shutil.copy2(alt_file, os.path.join(data_dir, fname))
                    else:
                        return _split_single_csv(alt_file, data_dir)
                    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                        return {"train": train_path, "val": val_path, "test": test_path}
    
    # Generate synthetic data with GoEmotions-realistic size
    logger.warning("Generating synthetic multi-label GoEmotions data (≈58k total samples)...")
    from .config import GoEmotionsConfig as GEC
    go_cfg = GEC()
    syn_train = go_cfg.max_train_samples or 44000
    syn_val = go_cfg.max_val_samples or 5500
    syn_test = go_cfg.max_test_samples or 5500
    _generate_synthetic_multilabel(
        data_dir,
        num_train=syn_train,
        num_val=syn_val,
        num_test=syn_test,
    )
    logger.info(f"Synthetic multi-label dataset generated at {data_dir}")
    
    return {"train": train_path, "val": val_path, "test": test_path}


def _try_download_via_kagglehub(config: TrainingConfig, data_dir: str) -> bool:
    """Try to download dataset via kagglehub."""
    try:
        logger.info(f"Downloading GoEmotions from Kaggle: {config.kaggle_dataset}")
        import kagglehub
        dataset_path = kagglehub.dataset_download(config.kaggle_dataset)
        logger.info(f"Dataset downloaded to {dataset_path}")
        
        import shutil
        for fname in ["train.csv", "test.csv", "val.csv"]:
            src = os.path.join(dataset_path, fname)
            dst = os.path.join(data_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logger.info(f"Copied {fname} to {data_dir}")
        return True
    except Exception as e:
        logger.warning(f"kagglehub download failed: {e}")
        return False


def _try_download_via_opendatasets(data_dir: str) -> bool:
    """Try to download dataset via opendatasets."""
    try:
        logger.info("Attempting download via opendatasets...")
        import opendatasets as od
        od.download(
            "https://www.kaggle.com/datasets/debarshichanda/goemotions",
            data_dir,
        )
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".csv"):
                    import shutil
                    shutil.copy2(os.path.join(root, f), os.path.join(data_dir, f))
        return True
    except Exception as e:
        logger.warning(f"opendatasets download failed: {e}")
        return False


def _generate_synthetic_multilabel(
    output_dir: str,
    num_train: int = 44000,
    num_val: int = 5500,
    num_test: int = 5500,
):
    """
    Generate synthetic multi-label GoEmotions-like data for testing.
    Each sample can have multiple emotion labels (multi-label).
    """
    emotion_texts = _get_synthetic_emotion_texts()
    
    # Multi-label co-occurrence probabilities
    # Certain emotions tend to co-occur
    co_occurrence_pairs = [
        ("admiration", "gratitude", 0.3),
        ("anger", "annoyance", 0.4),
        ("joy", "excitement", 0.35),
        ("sadness", "grief", 0.25),
        ("surprise", "realization", 0.3),
        ("fear", "nervousness", 0.35),
        ("love", "caring", 0.3),
        ("disappointment", "sadness", 0.3),
        ("approval", "admiration", 0.25),
    ]
    
    # Approximate distribution weights for 28 labels
    dist_weights = {
        "admiration": 0.10, "amusement": 0.06, "anger": 0.05, "annoyance": 0.04,
        "approval": 0.08, "caring": 0.03, "confusion": 0.03, "curiosity": 0.03,
        "desire": 0.02, "disappointment": 0.03, "disapproval": 0.03, "disgust": 0.02,
        "embarrassment": 0.02, "excitement": 0.04, "fear": 0.03,
        "gratitude": 0.06, "grief": 0.02, "joy": 0.08, "love": 0.04,
        "nervousness": 0.02, "optimism": 0.03, "pride": 0.02,
        "realization": 0.03, "relief": 0.02, "remorse": 0.02,
        "sadness": 0.04, "surprise": 0.03, "neutral": 0.20,
    }
    
    label_columns = GOEMOTIONS_CSV_COLUMNS
    emotions_pool = list(emotion_texts.keys())
    weights_list = [dist_weights.get(e, 0.02) for e in emotions_pool]
    total_w = sum(weights_list)
    weights_list = [w / total_w for w in weights_list]
    
    def create_rows(count):
        rows = []
        for _ in range(count):
            # Primary emotion
            primary = random.choices(emotions_pool, weights=weights_list, k=1)[0]
            text = random.choice(emotion_texts[primary])
            
            # Multi-hot vector
            row = {col: 0 for col in label_columns}
            row["text"] = text
            row[primary] = 1
            
            # Add co-occurring emotion
            for emo1, emo2, prob in co_occurrence_pairs:
                if primary == emo1 and random.random() < prob:
                    row[emo2] = 1
                elif primary == emo2 and random.random() < prob:
                    row[emo1] = 1
            
            # Random second label noise
            if random.random() < 0.08:
                other = random.choice(emotions_pool)
                if other != primary:
                    row[other] = 1
            
            rows.append(row)
        return rows
    
    train_rows = create_rows(num_train)
    val_rows = create_rows(num_val)
    test_rows = create_rows(num_test)
    
    columns = ["text"] + label_columns
    pd.DataFrame(train_rows, columns=columns).to_csv(
        os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(val_rows, columns=columns).to_csv(
        os.path.join(output_dir, "val.csv"), index=False)
    pd.DataFrame(test_rows, columns=columns).to_csv(
        os.path.join(output_dir, "test.csv"), index=False)
    
    logger.info(f"Generated synthetic multi-label data: train={num_train}, val={num_val}, test={num_test}")


def _get_synthetic_emotion_texts() -> Dict[str, List[str]]:
    """Get synthetic English texts for each emotion class."""
    return {
        "admiration": [
            "That's absolutely incredible work!",
            "I really admire your dedication to this.",
            "You're such an inspiration to everyone.",
            "What a brilliant achievement!",
            "I respect your commitment to this cause.",
            "Your efforts are truly commendable.",
            "That was a masterpiece of work.",
            "You set a great example for everyone.",
        ],
        "amusement": [
            "That was hilarious!",
            "I can't stop laughing at this!",
            "This is too funny!",
            "You always crack me up!",
            "That meme is gold!",
            "I'm dying laughing!",
            "This is comedy gold!",
            "That joke was perfect!",
        ],
        "anger": [
            "This is absolutely infuriating!",
            "I can't believe how stupid this is.",
            "This makes me so angry!",
            "I'm furious about what happened.",
            "This is completely unacceptable.",
            "I've had it with this nonsense.",
            "This is outrageous!",
            "This makes my blood boil!",
        ],
        "annoyance": [
            "This is so annoying.",
            "Why does this always happen?",
            "Ugh, not this again.",
            "This is really irritating.",
            "I'm so tired of this.",
            "Can you not?",
            "This is getting on my nerves.",
            "Why are you like this?",
        ],
        "approval": [
            "That sounds good to me.",
            "I agree with this completely.",
            "Well said, I second that.",
            "That's a great point.",
            "I'm on board with this.",
            "Finally, someone said it!",
            "This is the right take.",
            "That's exactly what I was thinking.",
        ],
        "caring": [
            "Take care of yourself, okay?",
            "I'm here if you need anything.",
            "Please be safe out there.",
            "Let me know if I can help.",
            "I hope you feel better soon.",
            "You're not alone in this.",
            "Sending you positive thoughts.",
            "I care about you.",
        ],
        "confusion": [
            "I don't understand what's happening.",
            "This doesn't make any sense.",
            "I'm so confused right now.",
            "What does this even mean?",
            "Can someone explain this?",
            "I'm lost, what's going on?",
            "This is so confusing.",
            "Wait, what?",
        ],
        "curiosity": [
            "I wonder why that happened.",
            "What do you think about this?",
            "I'm curious to know more.",
            "How does that work exactly?",
            "Tell me more about this.",
            "That's interesting, why is that?",
            "I'd love to learn more.",
            "What's the story behind this?",
        ],
        "desire": [
            "I really wish I could do that.",
            "If only I could have that.",
            "I want this so badly.",
            "I'm longing for a change.",
            "I hope one day I can experience that.",
            "That's everything I've ever wanted.",
            "I dream of being able to do that.",
            "I'm wishing so hard for this.",
        ],
        "disappointment": [
            "I'm really disappointed by this.",
            "That didn't live up to expectations.",
            "I expected so much more.",
            "What a letdown.",
            "I'm sad that it turned out this way.",
            "This is such a disappointment.",
            "I was really hoping for better.",
            "That was underwhelming.",
        ],
        "disapproval": [
            "I don't approve of this at all.",
            "This is not okay with me.",
            "I strongly disagree with this.",
            "That's not right.",
            "I can't support this decision.",
            "This is concerning.",
            "I don't think that's a good idea.",
            "That's not acceptable.",
        ],
        "disgust": [
            "That's absolutely disgusting!",
            "I'm repulsed by this.",
            "This is revolting.",
            "How could anyone think this is okay?",
            "That's gross.",
            "I'm sickened by this.",
            "This is vile.",
            "That's despicable.",
        ],
        "embarrassment": [
            "I'm so embarrassed right now.",
            "I can't believe I did that.",
            "This is so awkward.",
            "I want to disappear.",
            "That was so cringe.",
            "I'm mortified.",
            "Please forget that happened.",
            "This is so humiliating.",
        ],
        "excitement": [
            "I'm so excited for this!",
            "This is going to be amazing!",
            "I can't wait!",
            "This is so exciting!",
            "I'm hyped!",
            "Let's go!",
            "This is going to be epic!",
            "I'm counting down the days!",
        ],
        "fear": [
            "I'm terrified of what might happen.",
            "That was absolutely horrifying.",
            "I'm scared to death.",
            "This is genuinely frightening.",
            "I'm so afraid right now.",
            "That sent chills down my spine.",
            "I've never been so frightened.",
            "The fear is overwhelming.",
        ],
        "gratitude": [
            "Thank you so much!",
            "I really appreciate this.",
            "I'm so grateful for your help.",
            "Thank you from the bottom of my heart.",
            "I can't thank you enough.",
            "This means so much to me.",
            "I'm truly thankful.",
            "Thanks a million!",
        ],
        "grief": [
            "I'm devastated by this loss.",
            "My heart is breaking.",
            "I'm overwhelmed with grief.",
            "The pain is unbearable.",
            "I've lost all hope.",
            "I'm drowning in sorrow.",
            "Nothing matters anymore.",
            "I feel so empty inside.",
        ],
        "joy": [
            "I'm so happy!",
            "This is the best news ever!",
            "I'm absolutely delighted!",
            "What a wonderful day!",
            "I'm over the moon!",
            "I'm on top of the world!",
            "This is pure bliss!",
            "Best day ever!",
        ],
        "love": [
            "I absolutely adore you.",
            "You mean the world to me.",
            "My heart is full of love.",
            "I cherish every moment.",
            "You're my everything.",
            "I love you so much.",
            "You're one of a kind.",
            "I'm so lucky to have you.",
        ],
        "nervousness": [
            "I'm really nervous about this.",
            "My anxiety is through the roof.",
            "I can't stop worrying.",
            "I feel so uneasy.",
            "My heart is racing.",
            "I'm dreading what comes next.",
            "I'm on edge all the time.",
            "I can't seem to calm down.",
        ],
        "optimism": [
            "I'm hopeful things will get better.",
            "Things are looking up!",
            "I believe it will work out.",
            "There's light at the end of the tunnel.",
            "I'm optimistic about the future.",
            "Everything will be fine.",
            "I see a bright future ahead.",
            "Better days are coming.",
        ],
        "pride": [
            "I'm so proud of myself!",
            "That was all my hard work paying off.",
            "I did it!",
            "I'm proud of what I accomplished.",
            "I can't believe I pulled this off.",
            "I'm really proud of this achievement.",
            "All that work was worth it.",
            "I'm proud of who I've become.",
        ],
        "realization": [
            "Oh, now I get it!",
            "It all makes sense now.",
            "I finally understand.",
            "That explains everything.",
            "Oh wow, I just realized.",
            "It clicked for me.",
            "Now I see what you mean.",
            "That's what was happening.",
        ],
        "relief": [
            "What a relief!",
            "I'm so glad that's over.",
            "Thank goodness!",
            "I can finally breathe.",
            "That was a close call.",
            "I'm relieved it worked out.",
            "A weight has been lifted.",
            "Finally, some good news.",
        ],
        "remorse": [
            "I'm really sorry for what I did.",
            "I regret my actions deeply.",
            "I wish I could take it back.",
            "I feel terrible about this.",
            "Please forgive me.",
            "I know I messed up.",
            "I'm full of regret.",
            "I should have known better.",
        ],
        "sadness": [
            "I'm so sad about this.",
            "This breaks my heart.",
            "I feel so depressed.",
            "I can't stop crying.",
            "I feel so empty.",
            "My heart aches.",
            "I'm overwhelmed with sadness.",
            "The sadness won't go away.",
        ],
        "surprise": [
            "Wow, I did not expect that!",
            "That's absolutely shocking!",
            "I'm completely astonished!",
            "This came out of nowhere!",
            "Unbelievable!",
            "I never saw that coming!",
            "I'm blown away!",
            "That's genuinely surprising!",
        ],
        "neutral": [
            "I went to the store today.",
            "The weather is nice outside.",
            "I need to finish my work.",
            "The meeting is at 3pm.",
            "I'll have coffee please.",
            "The book is on the table.",
            "Let me know if you need help.",
            "I'll see you tomorrow.",
        ],
    }


def _split_single_csv(filepath: str, output_dir: str) -> Dict[str, str]:
    """Split a single CSV file into train/val/test splits."""
    logger.info(f"Splitting {filepath} into train/val/test...")
    df = pd.read_csv(filepath)
    
    np.random.seed(42)
    mask = np.random.rand(len(df))
    train_mask = mask < 0.8
    val_mask = (mask >= 0.8) & (mask < 0.9)
    test_mask = mask >= 0.9
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    df[train_mask].to_csv(train_path, index=False)
    df[val_mask].to_csv(val_path, index=False)
    df[test_mask].to_csv(test_path, index=False)
    
    logger.info(f"Split complete: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    return {"train": train_path, "val": val_path, "test": test_path}


# ============================================================
# Data Augmentation (EDA)
# ============================================================

class TextAugmenter:
    """
    Easy Data Augmentation (EDA) techniques for English text.
    Synonym replacement, random swap, random deletion.
    Used for training data augmentation.
    """
    
    def __init__(
        self,
        synonym_prob: float = 0.3,
        random_swap: int = 2,
        random_delete_prob: float = 0.1,
    ):
        self.synonym_prob = synonym_prob
        self.random_swap = random_swap
        self.random_delete_prob = random_delete_prob
        self._synonyms = self._get_default_synonyms()
    
    def augment(self, text: str) -> str:
        """Apply random augmentation to text."""
        words = text.split()
        if len(words) <= 3:
            return text
        
        aug_type = random.choice(["synonym", "swap", "delete", "none"])
        
        if aug_type == "synonym" and self.synonym_prob > 0:
            words = self._synonym_replacement(words)
        elif aug_type == "swap" and self.random_swap > 0:
            words = self._random_swap(words)
        elif aug_type == "delete" and self.random_delete_prob > 0:
            words = self._random_deletion(words)
        
        return " ".join(words)
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        new_words = words.copy()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self._synonyms and random.random() < self.synonym_prob:
                synonym = random.choice(self._synonyms[word_lower])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words[i] = synonym
        return new_words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        new_words = words.copy()
        for _ in range(min(self.random_swap, len(words) // 2)):
            i, j = random.sample(range(len(words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        new_words = [
            word for word in words
            if random.random() > self.random_delete_prob
        ]
        return new_words if new_words else [random.choice(words)]
    
    def _get_default_synonyms(self) -> Dict[str, List[str]]:
        return {
            "good": ["great", "excellent", "fine", "nice", "wonderful"],
            "bad": ["terrible", "awful", "horrible", "poor", "lousy"],
            "happy": ["glad", "joyful", "cheerful", "delighted", "pleased"],
            "sad": ["unhappy", "sorrowful", "gloomy", "depressed", "down"],
            "angry": ["furious", "irate", "enraged", "mad", "livid"],
            "love": ["adore", "cherish", "treasure", "worship"],
            "hate": ["detest", "despise", "loathe", "abhor"],
            "big": ["large", "huge", "enormous", "massive", "immense"],
            "small": ["tiny", "little", "miniature", "compact"],
            "very": ["extremely", "incredibly", "remarkably", "highly"],
            "pretty": ["quite", "rather", "fairly", "somewhat"],
            "nice": ["pleasant", "lovely", "delightful", "agreeable"],
            "great": ["fantastic", "superb", "outstanding", "remarkable"],
            "think": ["believe", "suppose", "reckon", "consider"],
            "feel": ["sense", "experience", "perceive", "detect"],
            "look": ["appear", "seem", "resemble", "suggest"],
            "want": ["desire", "wish", "long", "yearn"],
            "need": ["require", "necessitate", "demand"],
            "like": ["enjoy", "appreciate", "admire", "favor"],
            "cool": ["awesome", "amazing", "impressive", "neat"],
            "weird": ["strange", "odd", "bizarre", "peculiar"],
            "fun": ["entertaining", "enjoyable", "amusing", "lively"],
            "scared": ["afraid", "frightened", "terrified", "alarmed"],
            "tired": ["exhausted", "weary", "fatigued", "drained"],
            "interesting": ["intriguing", "fascinating", "engaging", "captivating"],
            "boring": ["tedious", "monotonous", "dull", "uninteresting"],
            "easy": ["simple", "effortless", "straightforward"],
            "hard": ["difficult", "challenging", "tough", "arduous"],
            "thank": ["grateful", "appreciative", "thankful"],
            "sorry": ["apologetic", "regretful", "remorseful"],
            "wow": ["amazing", "astonishing", "stunning", "incredible"],
            "amazing": ["incredible", "fantastic", "remarkable", "extraordinary"],
            "beautiful": ["gorgeous", "stunning", "lovely", "breathtaking"],
            "terrible": ["dreadful", "horrible", "awful", "atrocious"],
            "exciting": ["thrilling", "inspiring", "stimulating", "electrifying"],
            "peaceful": ["calm", "serene", "tranquil", "relaxed"],
            "confused": ["perplexed", "bewildered", "puzzled", "baffled"],
            "annoyed": ["irritated", "vexed", "aggravated", "frustrated"],
            "surprised": ["astonished", "astounded", "flabbergasted", "dumbfounded"],
            "grateful": ["thankful", "appreciative", "indebted", "obliged"],
            "worried": ["concerned", "troubled", "distressed", "anxious"],
            "confident": ["assured", "certain", "positive", "optimistic"],
            "lonely": ["isolated", "solitary", "alone", "forsaken"],
            "hopeful": ["optimistic", "encouraged", "promising", "positive"],
            "hopeless": ["desperate", "despairing", "forlorn", "downhearted"],
        }
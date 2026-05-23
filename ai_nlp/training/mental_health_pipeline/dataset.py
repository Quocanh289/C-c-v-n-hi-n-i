"""
Dataset Module
===============
PyTorch Dataset and DataLoader creation for mental health classification.

Handles:
- Loading the Kaggle Reddit mental health CSV
- Cleaning and preprocessing
- Train/validation/test split (stratified)
- Tokenization with dynamic padding
- Data augmentation (EDA: Easy Data Augmentation)
- Balanced sampling via WeightedRandomSampler
- Optimized DataLoader with pinned memory + multiprocessing
"""

import os
import math
import random
import logging
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import (
    MENTAL_HEALTH_LABELS,
    MENTAL_HEALTH_LABELS_TO_IDX,
    IDX_TO_MH_LABELS,
    MH_CLASS_COUNTS,
    TrainingConfig,
    MHConfig,
)
from .preprocessor import RedditTextPreprocessor

logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Easy Data Augmentation (EDA) for mental health text.
    
    Operations:
    1. Synonym replacement — replace words with WordNet synonyms
    2. Random insertion — insert synonyms of random words
    3. Random swap — swap two words in the sentence
    4. Random deletion — delete words with probability p
    
    For mental health texts, we apply MUCH lower augmentation rates
    to avoid altering clinical meaning.
    """
    
    def __init__(
        self,
        synonym_prob: float = 0.2,
        swap_max: int = 2,
        delete_prob: float = 0.05,
        seed: int = 42,
    ):
        self.synonym_prob = synonym_prob
        self.swap_max = swap_max
        self.delete_prob = delete_prob
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Try to load WordNet (for synonym replacement)
        self.stop_words = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "i", "me", "my", "we", "our", "you", "your", "he", "she",
            "it", "they", "them", "this", "that", "these", "those",
            "am", "not", "no", "nor", "so", "if", "or", "and", "but",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "about", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "only", "own", "same",
            # Mental health words we don't want to replace
            "depressed", "anxious", "panic", "suicide", "hopeless",
        ])
        
        self._wordnet_available = False
        try:
            import nltk
            nltk.data.find("corpora/wordnet")
            self._wordnet_available = True
            from nltk.corpus import wordnet
            self.wordnet = wordnet
        except (LookupError, ImportError):
            logger.warning("WordNet not available. Synonym replacement disabled. "
                          "Run: python -c 'import nltk; nltk.download(\"wordnet\")'")
    
    def synonym_replacement(self, words: List[str], n: int) -> List[str]:
        """Replace n words with WordNet synonyms."""
        if not self._wordnet_available or n <= 0:
            return words
            
        new_words = list(words)
        candidate_indices = [
            i for i, w in enumerate(new_words)
            if w.lower() not in self.stop_words and w.isalpha()
        ]
        
        self.rng.shuffle(candidate_indices)
        num_replaced = 0
        
        for idx in candidate_indices:
            if num_replaced >= n:
                break
            synonyms = self._get_synonyms(new_words[idx])
            if synonyms:
                synonym = self.rng.choice(synonyms)
                new_words[idx] = synonym
                num_replaced += 1
                
        return new_words
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get WordNet synonyms for a word."""
        synonyms = set()
        for syn in self.wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ").replace("-", " ")
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)
    
    def random_swap(self, words: List[str], n: int) -> List[str]:
        """Swap n pairs of words randomly."""
        if n <= 0 or len(words) < 2:
            return words
            
        new_words = list(words)
        for _ in range(min(n, len(words) // 2)):
            idx1, idx2 = self.rng.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words
    
    def random_deletion(self, words: List[str], p: float) -> List[str]:
        """Delete each word with probability p."""
        if p <= 0 or len(words) <= 3:
            return words
            
        new_words = []
        for word in words:
            if self.rng.random() > p:
                new_words.append(word)
                
        # Don't delete everything
        if len(new_words) == 0:
            return words[:1]
        return new_words
    
    # Clinical keywords that MUST NOT be augmented or changed
    # Augmenting these could alter clinical meaning or mask distress signals
    CLINICAL_PROTECTED_KEYWORDS = [
        "kill myself", "end my life", "want to die", "better off dead",
        "suicide", "no reason to live", "can't go on", "end it all",
        "wish i was dead", "take my own life", "suicidal", "ending it",
        "don't want to live", "i give up", "nothing matters",
        "ready to die", "just want peace", "i want out",
        "depressed", "depression", "hopeless", "worthless",
        "panic attack", "mania", "hypomania", "psychotic",
        "self harm", "self-harm", "cutting", "overdose",
    ]
    
    def _has_protected_content(self, text: str) -> bool:
        """Check if text contains clinical keywords that should not be augmented."""
        text_lower = text.lower()
        for keyword in self.CLINICAL_PROTECTED_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    def augment(self, text: str) -> str:
        """
        Apply EDA to a single text.
        Returns augmented text (or original if no operation selected).
        
        Safety: texts containing critical clinical keywords (suicidal ideation,
        self-harm, crisis statements) are NOT augmented — we must preserve
        their semantic meaning exactly.
        """
        # PROTECTED: Do not augment any text with critical clinical keywords
        if self._has_protected_content(text):
            return text
        
        words = text.split()
        if len(words) <= 3:
            return text
        
        # Randomly choose one augmentation operation
        op = self.rng.random()
        
        if op < 0.33 and self._wordnet_available:
            # Synonym replacement
            n = max(1, int(len(words) * self.synonym_prob))
            words = self.synonym_replacement(words, n)
        elif op < 0.66:
            # Random swap
            n = max(1, self.swap_max)
            words = self.random_swap(words, n)
        else:
            # Random deletion
            words = self.random_deletion(words, self.delete_prob)
            
        return " ".join(words)


class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for mental health classification.
    
    Features:
    - Pre-tokenized for fast DataLoader throughput
    - Optional on-the-fly augmentation
    - Returns input_ids, attention_mask, label
    - Works with HuggingFace tokenizers
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        is_training: bool = False,
        augmenter: Optional[DataAugmenter] = None,
    ):
        assert len(texts) == len(labels), "Texts and labels must have same length"
        
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.augmenter = augmenter
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Apply augmentation for training
        if self.is_training and self.augmenter is not None:
            text = self.augmenter.augment(text)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Fixed-size padding for batch consistency
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


def load_and_split_data(
    config: TrainingConfig,
    mh_config: MHConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the mental health CSV, clean, and split into train/val/test.
    
    Args:
        config: TrainingConfig with paths
        mh_config: MHConfig with preprocessing params
        
    Returns:
        (train_df, val_df, test_df)
    """
    data_path = config.get_data_file_path()
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please download from Kaggle: "
            f"https://www.kaggle.com/datasets/maazkareem/sentiment-and-mental-health-dataset-reddit-based"
        )
    
    # Load CSV
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
    
    # Map label column name
    text_col = config.text_column
    label_col = config.label_column
    
    # Verify columns
    if text_col not in df.columns:
        # Try to find the text column
        potential_text_cols = [c for c in df.columns if c.lower() in ("statement", "text", "sentence", "content")]
        if potential_text_cols:
            text_col = potential_text_cols[0]
            logger.info(f"Using '{text_col}' as text column")
        else:
            raise ValueError(f"Text column '{text_col}' not found. Available: {list(df.columns)}")
    
    if label_col not in df.columns:
        potential_label_cols = [c for c in df.columns if c.lower() in ("status", "label", "class", "category", "target")]
        if potential_label_cols:
            label_col = potential_label_cols[0]
            logger.info(f"Using '{label_col}' as label column")
        else:
            raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    
    # Check label values
    available_labels = df[label_col].unique()
    valid_labels = set(MENTAL_HEALTH_LABELS)
    missing_labels = set(available_labels) - valid_labels
    if missing_labels:
        logger.warning(f"Unknown labels found: {missing_labels}. Mapping them.")
        # Try to normalize
        label_norm_map = {}
        for lbl in available_labels:
            lbl_str = str(lbl).strip()
            if lbl_str.lower() in [l.lower() for l in MENTAL_HEALTH_LABELS]:
                # Find the correctly cased version
                for valid in MENTAL_HEALTH_LABELS:
                    if lbl_str.lower() == valid.lower():
                        label_norm_map[lbl] = valid
                        break
            else:
                label_norm_map[lbl] = lbl_str
        
        df[label_col] = df[label_col].map(label_norm_map)
        # Drop rows with labels not in our set
        df = df[df[label_col].isin(MENTAL_HEALTH_LABELS)]
        logger.info(f"After normalization: {len(df)} samples")
    
    # Clean text
    preprocessor = RedditTextPreprocessor(
        min_text_length=mh_config.min_text_length,
        remove_leakage=mh_config.remove_leakage,
        leakage_words=mh_config.leakage_words,
    )
    
    df = preprocessor.process_dataframe(
        df,
        text_column=text_col,
        label_column=label_col,
        remove_leakage=mh_config.remove_leakage,
    )
    
    # Log final label distribution
    logger.info("Final label distribution:")
    for label, count in df[label_col].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    # First split: train vs temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(mh_config.val_ratio + mh_config.test_ratio),
        random_state=config.random_seed,
        stratify=df[label_col] if mh_config.stratify else None,
    )
    
    # Second split: val vs test
    val_ratio_of_temp = mh_config.val_ratio / (mh_config.val_ratio + mh_config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_of_temp),
        random_state=config.random_seed,
        stratify=temp_df[label_col] if mh_config.stratify else None,
    )
    
    logger.info(
        f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


def create_dataloaders(
    config: TrainingConfig,
    mh_config: MHConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Optimizations for RTX 4060 16GB:
    - num_workers=4 for parallel data loading
    - pin_memory=True for faster GPU transfer
    - WeightedRandomSampler for balanced training
    - Batch size = 16 for training, 32 for evaluation
    """
    # Load and split data
    train_df, val_df, test_df = load_and_split_data(config, mh_config)
    
    # Map labels to indices
    label_to_idx = MENTAL_HEALTH_LABELS_TO_IDX
    
    # Create augmenter
    augmenter = DataAugmenter(
        synonym_prob=config.aug_synonym_prob,
        swap_max=config.aug_random_swap,
        delete_prob=config.aug_random_delete_prob,
        seed=config.seed,
    ) if config.use_augmentation else None
    
    # Create datasets
    train_dataset = MentalHealthDataset(
        texts=train_df["cleaned_text"].tolist(),
        labels=[label_to_idx[l] for l in train_df[config.label_column]],
        tokenizer=tokenizer,
        max_length=mh_config.max_length,
        is_training=True,
        augmenter=augmenter,
    )
    
    val_dataset = MentalHealthDataset(
        texts=val_df["cleaned_text"].tolist(),
        labels=[label_to_idx[l] for l in val_df[config.label_column]],
        tokenizer=tokenizer,
        max_length=mh_config.max_length,
        is_training=False,
    )
    
    test_dataset = MentalHealthDataset(
        texts=test_df["cleaned_text"].tolist(),
        labels=[label_to_idx[l] for l in test_df[config.label_column]],
        tokenizer=tokenizer,
        max_length=mh_config.max_length,
        is_training=False,
    )
    
    # WeightedRandomSampler for balanced training
    train_sampler = None
    if config.use_balanced_sampling:
        # Compute weights inversely proportional to class frequencies
        train_labels = train_df[config.label_column].tolist()
        class_counts = np.bincount([label_to_idx[l] for l in train_labels], minlength=config.num_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label_to_idx[l]] for l in train_labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info("Using WeightedRandomSampler for balanced training")
    
    # Create DataLoaders with optimal settings for RTX 4060
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,  # Drop incomplete batch for consistent gradient accumulation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    logger.info(
        f"Dataloaders created: "
        f"Train={len(train_loader)} batches x {config.batch_size}, "
        f"Val={len(val_loader)} batches x {config.eval_batch_size}, "
        f"Test={len(test_loader)} batches x {config.eval_batch_size}"
    )
    
    return train_loader, val_loader, test_loader


def get_tokenizer(config: TrainingConfig) -> PreTrainedTokenizer:
    """
    Load tokenizer for the specified model.
    
    For DeBERTa-v3: uses AutoTokenizer
    For XLM-RoBERTa: uses XLMRobertaTokenizer
    
    Optimization tips:
    - Add padding side = 'right' for left-to-right models
    - Set truncation_side = 'right' to truncate from the end
    - Cache tokenizer to avoid re-downloading
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,  # Use Rust tokenizer for 5x speed
        add_prefix_space=True,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    
    logger.info(
        f"Tokenizer loaded: {config.model_name} "
        f"(vocab_size={tokenizer.vocab_size}, "
        f"max_length={config.max_seq_length})"
    )
    
    return tokenizer
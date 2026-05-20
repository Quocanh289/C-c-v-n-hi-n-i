"""
Dataset Module
===============
Loads, preprocesses, and augments the GoEmotions dataset from Kaggle.
Maps 27 fine-grained emotion labels to 9 coarse emotions.
"""

import os
import csv
import json
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Callable
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import (
    EMOTION_27_TO_9_MAP,
    COARSE_EMOTIONS,
    COARSE_TO_IDX,
    GOEMOTIONS_27,
    TrainingConfig,
    GoEmotionsConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# Emotion Label Mapper
# ============================================================

class EmotionLabelMapper:
    """
    Maps GoEmotions 27 fine-grained labels to 9 coarse labels.
    Handles multi-label to single-label conversion by taking
    the highest-priority emotion when multiple are present.
    """

    # Priority order: when multiple emotions are present, use the first match
    PRIORITY_ORDER = [
        "anger", "sadness", "fear", "anxiety", "surprise",
        "joy", "love", "admiration", "neutral"
    ]

    def __init__(self, coarse_labels: List[str] = COARSE_EMOTIONS):
        self.coarse_labels = coarse_labels
        self.fine_to_coarse = EMOTION_27_TO_9_MAP

    def map_labels(self, fine_labels: List[str]) -> int:
        """
        Convert list of fine-grained label names to a single coarse index.
        If no emotions present, returns 'neutral'.
        If multiple emotions, takes highest priority coarse.
        """
        if not fine_labels or all(l == "neutral" for l in fine_labels):
            return COARSE_TO_IDX.get("neutral", 8)

        coarse_set = set()
        for label in fine_labels:
            if label in self.fine_to_coarse:
                coarse_set.add(self.fine_to_coarse[label])

        if not coarse_set:
            return COARSE_TO_IDX.get("neutral", 8)

        # Return highest priority coarse emotion
        for priority in self.PRIORITY_ORDER:
            if priority in coarse_set:
                return COARSE_TO_IDX.get(priority, 8)

        return COARSE_TO_IDX["neutral"]

    def map_vector(self, one_hot_vector: np.ndarray, label_names: List[str]) -> int:
        """Convert one-hot encoded vector to coarse label index."""
        active_labels = [label_names[i] for i, val in enumerate(one_hot_vector) if val == 1]
        return self.map_labels(active_labels)


# ============================================================
# Kaggle GoEmotions Downloader
# ============================================================

def get_kaggle_goemotions(config: TrainingConfig) -> Dict[str, str]:
    """
    Download the GoEmotions dataset from Kaggle if not already present.
    Falls back to generating a synthetic dataset for testing.
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

    # Try downloading from Kaggle via opendatasets
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

    # If dataset is still not found, generate synthetic GoEmotions-like data
    logger.warning("GoEmotions dataset not found via Kaggle. Generating synthetic data for testing...")
    # Use reasonable defaults for synthetic data
    from .config import GoEmotionsConfig
    go_cfg = GoEmotionsConfig()
    syn_train = go_cfg.max_train_samples or 5000
    syn_val = go_cfg.max_val_samples or 500
    _generate_synthetic_goemotions(data_dir, num_train=syn_train, num_val=syn_val, num_test=500)
    logger.info(f"Synthetic GoEmotions dataset generated at {data_dir}")

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
    """Try to download dataset via opendatasets (alternative method)."""
    try:
        logger.info("Attempting download via opendatasets...")
        import opendatasets as od
        download_path = os.path.join(data_dir, "..")
        od.download("https://www.kaggle.com/datasets/debarshichanda/goemotions", data_dir)
        
        # Check if files were downloaded (they might be in a subdirectory)
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".csv"):
                    import shutil
                    shutil.copy2(os.path.join(root, f), os.path.join(data_dir, f))
        return True
    except Exception as e:
        logger.warning(f"opendatasets download failed: {e}")
        return False


def _generate_synthetic_goemotions(
    output_dir: str,
    num_train: int = 5000,
    num_val: int = 500,
    num_test: int = 500,
):
    """Generate synthetic GoEmotions-like data for testing the pipeline."""
    import random
    
    # Sample texts for each coarse emotion
    emotion_texts = {
        "admiration": [
            "That's absolutely incredible work!", "I really admire your dedication.",
            "You're such an inspiration to all of us.", "What a brilliant achievement!",
            "I respect your commitment to this cause.", "Amazing job on the presentation!",
            "You have my deepest respect.", "That was truly impressive to watch.",
        ],
        "anger": [
            "This is absolutely infuriating!", "I can't believe how stupid this is.",
            "This makes me so angry!", "I'm furious about what happened.",
            "How dare they say that!", "This is completely unacceptable.",
            "I'm so fed up with this nonsense.", "That's the last straw!",
        ],
        "anxiety": [
            "I'm really worried about the results.", "This is making me so nervous.",
            "I can't stop stressing about this.", "I feel so uneasy right now.",
            "My anxiety is through the roof.", "I'm frightened about what might happen.",
            "This situation has me panicked.", "I feel so overwhelmed and anxious.",
        ],
        "fear": [
            "I'm terrified of what comes next.", "That was absolutely horrifying!",
            "I'm scared to death.", "This is like a nightmare come true.",
            "I dread the thought of that happening.", "That's genuinely frightening.",
            "I'm so afraid right now.", "This is pure terror.",
        ],
        "joy": [
            "I'm so happy right now!", "This is the best news ever!",
            "I'm absolutely delighted!", "What a wonderful day!",
            "I couldn't be more thrilled!", "This brings me so much joy.",
            "I'm over the moon!", "This is amazing, I'm so excited!",
        ],
        "love": [
            "I absolutely adore you!", "You're so precious to me.",
            "I love this so much!", "This is the most beautiful thing ever.",
            "You're so sweet and caring.", "I have so much love for this.",
            "This is absolutely lovely.", "You're such a wonderful person.",
        ],
        "sadness": [
            "I'm so sad about this.", "This breaks my heart.",
            "I feel so depressed and alone.", "This is truly disappointing.",
            "I can't stop crying.", "I feel so empty inside.",
            "This is the worst feeling ever.", "I'm heartbroken.",
        ],
        "surprise": [
            "Wow, I did not expect that!", "That's absolutely shocking!",
            "I'm completely astonished!", "This came out of nowhere!",
            "Unbelievable! I'm speechless.", "What a surprise!",
            "I never saw that coming!", "That's genuinely surprising!",
        ],
        "neutral": [
            "I went to the store today.", "The weather is nice outside.",
            "I need to finish my work.", "The meeting is at 3pm.",
            "I'll have coffee please.", "The book is on the table.",
            "Let me know if you need help.", "I'll see you tomorrow.",
        ],
    }
    
    label_columns = GOEMOTIONS_27
    
    def create_rows(count):
        rows = []
        for _ in range(count):
            coarse_emotion = random.choice(list(emotion_texts.keys()))
            text = random.choice(emotion_texts[coarse_emotion])
            
            # Create one-hot row
            row = {col: 0 for col in label_columns}
            row["text"] = text
            
            # Map coarse to one or more fine labels
            fine_labels = [k for k, v in EMOTION_27_TO_9_MAP.items() if v == coarse_emotion]
            if fine_labels:
                chosen = random.choice(fine_labels)
                row[chosen] = 1
                # Sometimes add a second fine label
                if random.random() < 0.15:
                    others = [l for l in fine_labels if l != chosen]
                    if others:
                        row[random.choice(others)] = 1
            
            rows.append(row)
        return rows
    
    train_rows = create_rows(num_train)
    val_rows = create_rows(num_val)
    test_rows = create_rows(num_test)
    
    # Save to CSV
    columns = ["text"] + label_columns
    pd.DataFrame(train_rows, columns=columns).to_csv(
        os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(val_rows, columns=columns).to_csv(
        os.path.join(output_dir, "val.csv"), index=False)
    pd.DataFrame(test_rows, columns=columns).to_csv(
        os.path.join(output_dir, "test.csv"), index=False)
    
    logger.info(f"Generated synthetic dataset: train={num_train}, val={num_val}, test={num_test}")


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
# Data Augmentation
# ============================================================

class TextAugmenter:
    """
    Easy Data Augmentation (EDA) techniques for text:
    - Synonym replacement
    - Random insertion (deactivated by default)
    - Random swap
    - Random deletion
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

        # Simple synonym dictionary
        self._synonyms = _get_default_synonyms()

    def augment(self, text: str) -> str:
        """Apply random augmentations to text."""
        words = text.split()
        if len(words) <= 3:
            return text  # Skip very short texts

        aug_type = random.choice(["synonym", "swap", "delete", "none"])

        if aug_type == "synonym" and self.synonym_prob > 0:
            words = self._synonym_replacement(words)
        elif aug_type == "swap" and self.random_swap > 0:
            words = self._random_swap(words)
        elif aug_type == "delete" and self.random_delete_prob > 0:
            words = self._random_deletion(words)

        return " ".join(words)

    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace words with synonyms."""
        new_words = words.copy()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self._synonyms and random.random() < self.synonym_prob:
                synonym = random.choice(self._synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words[i] = synonym
        return new_words

    def _random_swap(self, words: List[str]) -> List[str]:
        """Randomly swap two words."""
        new_words = words.copy()
        for _ in range(min(self.random_swap, len(words) // 2)):
            i, j = random.sample(range(len(words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words

    def _random_deletion(self, words: List[str]) -> List[str]:
        """Randomly delete words with given probability."""
        new_words = [
            word for word in words
            if random.random() > self.random_delete_prob
        ]
        return new_words if new_words else [random.choice(words)]  # Keep at least one word


def _get_default_synonyms() -> Dict[str, List[str]]:
    """Get a basic English synonym dictionary for augmentation."""
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
        "dislike": ["disfavor", "object", "oppose", "resent"],
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
        "please": ["delight", "satisfy", "gratify"],
        "wow": ["amazing", "astonishing", "stunning", "incredible"],
        "no": ["not", "never", "none", "nothing"],
        "yes": ["yeah", "yep", "sure", "absolutely"],
        "maybe": ["perhaps", "possibly", "potentially"],
        "always": ["constantly", "continually", "perpetually", "forever"],
        "never": ["rarely", "seldom", "hardly"],
    }


# ============================================================
# GoEmotions Dataset (PyTorch)
# ============================================================

class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions with:
    - 27→9 emotion label mapping
    - Optional data augmentation
    - Multi-label to single-label conversion
    - Class weight computation
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
        self.tokenizer = tokenizer
        self.config = config
        self.go_config = goemotions_config or GoEmotionsConfig()
        self.split = split
        self.augment = augment and split == "train"

        # Load CSV
        self.df = pd.read_csv(file_path)

        # Map labels
        self.mapper = EmotionLabelMapper()
        self.texts, self.labels = self._process_dataframe(self.df)

        # Limit samples if specified
        if max_samples and len(self.texts) > max_samples:
            indices = list(range(len(self.texts)))
            random.shuffle(indices)
            indices = indices[:max_samples]
            self.texts = [self.texts[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # Augmentation
        self.augmenter = TextAugmenter(
            synonym_prob=config.aug_synonym_prob,
            random_swap=config.aug_random_swap,
            random_delete_prob=config.aug_random_delete_prob,
        ) if augment else None

        # Compute class weights if needed
        self.class_weights = None
        if config.class_weights:
            self.class_weights = self._compute_class_weights()

        logger.info(
            f"Loaded {split} set: {len(self.texts)} samples, "
            f"{len(set(self.labels))} classes"
        )

    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Process DataFrame to extract texts and mapped labels."""
        texts = []
        labels = []

        # Check DataFrame format
        is_one_hot = any(col in df.columns for col in self.go_config.label_columns)
        is_single_col = "label" in df.columns

        for idx, row in df.iterrows():
            text = row.get("text", row.get(self.go_config.text_column, ""))
            if not text or not isinstance(text, str):
                continue

            if is_one_hot:
                # One-hot encoded format: multiple label columns with 0/1
                active_fine_labels = [
                    col for col in self.go_config.label_columns
                    if col in row and row[col] == 1
                ]
                coarse_idx = self.mapper.map_labels(active_fine_labels)
            elif is_single_col:
                # Single label column
                fine_label = str(row["label"])
                coarse_idx = self.mapper.map_labels([fine_label])
            else:
                # Try parsing from 'labels' column (space-separated indices)
                labels_str = str(row.get("labels", ""))
                if labels_str and labels_str != "nan":
                    fine_indices = [int(i) for i in labels_str.strip().split(",") if i.strip()]
                    active_fine_labels = [
                        GOEMOTIONS_27[i] for i in fine_indices
                        if 0 <= i < len(GOEMOTIONS_27)
                    ]
                    coarse_idx = self.mapper.map_labels(active_fine_labels)
                else:
                    coarse_idx = COARSE_TO_IDX.get("neutral", 8)

            # Apply neutral filter
            if self.go_config.filter_neutral and coarse_idx == COARSE_TO_IDX.get("neutral", 8):
                continue

            texts.append(text)
            labels.append(coarse_idx)

        return texts, labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Apply augmentation for training
        if self.augment and self.augmenter:
            text = self.augmenter.augment(text)

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
        }

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        label_counts = Counter(self.labels)
        total = len(self.labels)
        n_classes = self.config.num_labels

        weights = torch.zeros(n_classes)
        for i in range(n_classes):
            count = label_counts.get(i, 0)
            if count > 0:
                weights[i] = total / (count * n_classes)
            else:
                weights[i] = 1.0

        # Avoid extreme weights
        weights = torch.clamp(weights, min=0.1, max=10.0)
        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for analysis."""
        dist = Counter(self.labels)
        return {COARSE_EMOTIONS[i]: dist[i] for i in sorted(dist.keys())}

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """Create a weighted sampler for imbalanced classes."""
        weights = torch.zeros(len(self.labels))
        label_weights = self._compute_class_weights()
        for i, label in enumerate(self.labels):
            weights[i] = label_weights[label]
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
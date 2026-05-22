"""
Reddit Text Preprocessor
=========================
Advanced text cleaning pipeline specifically designed for Reddit mental health data.

Key considerations:
  - Reddit text is noisy (URLs, subreddit mentions, usernames)
  - Mental health posts contain emotional indicators (!!!, ???, repeated chars)
  - We must NOT strip emotional information
  - We must remove label leakage (text containing the diagnosis word itself)
  
Preprocessing pipeline order:
  1. Unicode normalization
  2. URL removal
  3. Subreddit r/... removal
  4. Username u/... removal
  5. HTML entity decoding
  6. Repeated character normalization (but preserve emotional punctuation)
  7. Whitespace normalization
  8. Label leakage detection
  9. Empty/short text filtering
"""

import re
import html
import unicodedata
import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RedditTextPreprocessor:
    """
    Advanced text preprocessor for Reddit mental health data.
    
    Preserves emotional information while cleaning noisy social media artifacts.
    Supports configurable pipeline stages and batch processing.
    """

    def __init__(
        self,
        min_text_length: int = 5,
        remove_leakage: bool = True,
        leakage_words: Optional[List[str]] = None,
        normalize_repeated_chars: bool = True,
    ):
        self.min_text_length = min_text_length
        self.remove_leakage = remove_leakage
        self.leakage_words = leakage_words or [
            "depression", "anxiety", "bipolar", "suicidal",
            "personality disorder", "stress", "stressed",
            # NOT including "normal" — it's far too common in Reddit text
        ]
        self.normalize_repeated_chars = normalize_repeated_chars

        # Stats tracking
        self.stats = {
            "total_processed": 0,
            "urls_removed": 0,
            "subreddits_removed": 0,
            "usernames_removed": 0,
            "repeated_chars_normalized": 0,
            "leakage_removed": 0,
            "too_short_removed": 0,
            "empty_removed": 0,
        }

    def clean_text(self, text: str) -> str:
        """
        Full cleaning pipeline for a single text.
        Order matters — each step builds on the previous.
        
        Args:
            text: Raw Reddit text
            
        Returns:
            Cleaned text (empty string if filtered)
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        original = text

        # Step 1: Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Decode HTML entities
        text = html.unescape(text)

        # Step 3: Remove URLs (preserve the text around them)
        url_pattern = r'https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+'
        urls_found = re.findall(url_pattern, text)
        if urls_found:
            self.stats["urls_removed"] += len(urls_found)
        text = re.sub(url_pattern, ' ', text)

        # Step 4: Remove subreddit mentions (r/subreddit)
        subreddit_pattern = r'/?(?:r|R)/(\w+)'
        subreddits_found = re.findall(subreddit_pattern, text)
        if subreddits_found:
            self.stats["subreddits_removed"] += len(subreddits_found)
        text = re.sub(subreddit_pattern, ' ', text)

        # Step 5: Remove usernames (u/username)
        username_pattern = r'/?(?:u|U)/(\w+)'
        usernames_found = re.findall(username_pattern, text)
        if usernames_found:
            self.stats["usernames_removed"] += len(usernames_found)
        text = re.sub(username_pattern, ' ', text)

        # Step 6: Remove markdown formatting links [text](url)
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)

        # Step 7: Remove Reddit quote markers (> text)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Step 8: Normalize repeated characters (but PRESERVE emotional punctuation)
        # Emotional punctuation to preserve: !!! ??? ... LOL LMAO etc.
        if self.normalize_repeated_chars:
            text = self._normalize_repeated_characters(text)

        # Step 9: Remove extra whitespace (but preserve single spaces)
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 10: Filter by minimum length
        if len(text) < self.min_text_length:
            self.stats["too_short_removed"] += 1
            return ""

        self.stats["total_processed"] += 1
        return text

    def _normalize_repeated_characters(self, text: str) -> str:
        """
        Normalize repeated characters like "sooooo" → "sooo", but:
        - PRESERVE repeated punctuation like !!! ??? ... as emotional signals
        - PRESERVE internet slang like "LOL", "LMAO", "ROFL"
        - Only normalize alphabetic characters repeated 4+ times to 3 repeats
        
        This is crucial for mental health detection because:
        - "I'm sooooo tired" (repeated vowels = emotional intensity)
        - "Help me !!!!" (multiple exclamation marks = distress)
        - "What???" (multiple question marks = confusion/panic)
        """
        # Preserve emotional punctuation patterns first by replacing with placeholders
        # We use special tokens to mark them
        preserved_patterns = []

        # Pattern 1: Multiple exclamation marks (!!!, !!1!, etc.)
        def preserve_exclamation(m):
            preserved_patterns.append(("!!", m.group(0)))
            return " !!__PRESERVE__!! "

        text = re.sub(r'!{2,}', preserve_exclamation, text)

        # Pattern 2: Multiple question marks
        def preserve_question(m):
            preserved_patterns.append(("??", m.group(0)))
            return " ??__PRESERVE__?? "

        text = re.sub(r'\?{2,}', preserve_question, text)

        # Pattern 3: Mixed !? sequences (emotional intensity)
        def preserve_mixed(m):
            preserved_patterns.append(("!?", m.group(0)))
            return " !?__PRESERVE__!? "

        text = re.sub(r'[!?]{2,}', preserve_mixed, text)

        # Pattern 4: Elipsis (..., ...., ......)
        def preserve_elipsis(m):
            preserved_patterns.append(("..", m.group(0)))
            return " ..__PRESERVE__.. "

        text = re.sub(r'\.{3,}', preserve_elipsis, text)

        # Now normalize alphabetic repeated characters
        # Replace characters repeated 4+ times with 3 repeats
        # e.g., "sooooo" → "sooo", "noooooo" → "nooo"
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

        # Restore preserved patterns
        for i, (_, original) in enumerate(preserved_patterns):
            placeholder = f"!!__PRESERVE__!!" if "!" in original and "?" not in original else \
                         f"??__PRESERVE__??" if "?" in original and "!" not in original else \
                         f"!?__PRESERVE__!?" if "!" in original and "?" in original else \
                         f"..__PRESERVE__.."
            # Find and replace the placeholder (there might be multiple instances)
            text = text.replace(placeholder, original, 1)

        if len(preserved_patterns) > 0:
            self.stats["repeated_chars_normalized"] += 1

        return text

    def detect_label_leakage(self, text: str, label: str) -> bool:
        """
        Detect if text contains the label as a diagnosis word.
        This prevents the model from simply learning to recognize the word
        "depression" to predict Depression — we want it to understand
        the LANGUAGE and SYMPTOMS of depression, not just keyword matching.
        
        Args:
            text: Cleaned text
            label: Ground truth label
            
        Returns:
            True if leakage detected, False otherwise
        """
        if not self.remove_leakage:
            return False

        text_lower = text.lower()
        label_lower = label.lower().replace("_", " ").replace("-", " ")

        # For "Normal" and "Stress", leakage is harder to define
        # "normal" is a very common word, "stress" is also common
        if label_lower in ("normal", "stress"):
            return False

        # For multi-word labels, check if the label words appear
        # e.g., "personality disorder" in text
        if label_lower in text_lower:
            self.stats["leakage_removed"] += 1
            return True

        return False

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "statement",
        label_column: str = "status",
        remove_leakage: bool = True,
    ) -> pd.DataFrame:
        """
        Process entire dataframe with cleaning and filtering.
        
        Args:
            df: Input dataframe
            text_column: Column name containing text
            label_column: Column name containing labels
            remove_leakage: Whether to remove samples with label leakage
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Processing dataframe: {len(df)} rows")

        # Reset stats
        self.stats = {k: 0 for k in self.stats}

        # Make a copy
        df = df.copy()

        # Track original rows
        original_count = len(df)

        # Remove rows with missing text or labels
        before = len(df)
        df = df.dropna(subset=[text_column, label_column])
        logger.info(f"Dropped {before - len(df)} rows with missing values")

        # Clean text
        logger.info("Cleaning text...")
        df["cleaned_text"] = df[text_column].apply(self.clean_text)

        # Remove empty texts after cleaning
        before_empty = len(df)
        df = df[df["cleaned_text"].str.len() >= self.min_text_length]
        self.stats["empty_removed"] = before_empty - len(df)
        logger.info(f"Removed {before_empty - len(df)} empty/short texts")

        # Remove label leakage if enabled
        if remove_leakage:
            before_leak = len(df)
            df["has_leakage"] = df.apply(
                lambda row: self.detect_label_leakage(row["cleaned_text"], row[label_column]),
                axis=1,
            )
            df = df[~df["has_leakage"]]
            df = df.drop(columns=["has_leakage"])
            logger.info(f"Removed {before_leak - len(df)} leaky samples")
        else:
            df = df.drop(columns=["has_leakage"], errors="ignore")

        # Remove duplicate texts
        if self.remove_leakage:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=["cleaned_text"])
            logger.info(f"Removed {before_dedup - len(df)} duplicate texts")

        # Final size
        logger.info(
            f"Processing complete: {original_count} → {len(df)} "
            f"({original_count - len(df)} removed, "
            f"{(1 - len(df)/original_count)*100:.1f}% reduction)"
        )

        # Log stats
        for k, v in self.stats.items():
            if v > 0:
                logger.info(f"  {k}: {v}")

        return df

    def get_length_statistics(self, df: pd.DataFrame, text_column: str = "cleaned_text") -> Dict:
        """
        Compute text length statistics for tokenization decisions.
        
        Returns:
            Dict with token length stats (both character and approximate token counts)
        """
        char_lengths = df[text_column].str.len()

        # Approximate token count (word-level, not subword)
        # English rough estimate: tokens ≈ chars / 4
        word_lengths = df[text_column].str.split().str.len()

        # Subword token estimate (more accurate for BPE/WordPiece)
        # Rough: whitespace tokens * 1.3 for English subword expansion
        subword_estimates = word_lengths * 1.3

        return {
            "char_length": {
                "mean": float(char_lengths.mean()),
                "median": float(char_lengths.median()),
                "std": float(char_lengths.std()),
                "min": int(char_lengths.min()),
                "max": int(char_lengths.max()),
                "p95": int(char_lengths.quantile(0.95)),
                "p99": int(char_lengths.quantile(0.99)),
            },
            "word_length": {
                "mean": float(word_lengths.mean()),
                "median": float(word_lengths.median()),
                "std": float(word_lengths.std()),
                "min": int(word_lengths.min()),
                "max": int(word_lengths.max()),
                "p95": int(word_lengths.quantile(0.95)),
                "p99": int(word_lengths.quantile(0.99)),
            },
            "subword_estimate": {
                "mean": float(subword_estimates.mean()),
                "median": float(subword_estimates.median()),
                "p95": int(subword_estimates.quantile(0.95)),
                "p99": int(subword_estimates.quantile(0.99)),
            },
            "recommended_max_length": min(512, int(subword_estimates.quantile(0.99) * 1.1)),
        }

    @staticmethod
    def get_label_distribution(df: pd.DataFrame, label_column: str = "status") -> pd.Series:
        """Get label distribution for EDA."""
        return df[label_column].value_counts()

    @staticmethod
    def compute_class_weights(df: pd.DataFrame, label_column: str = "status") -> Dict[str, float]:
        """
        Compute balanced class weights from label distribution.
        weight = total_samples / (n_classes * class_count)
        """
        counts = df[label_column].value_counts()
        n_classes = len(counts)
        total = len(df)
        return {
            label: total / (n_classes * count)
            for label, count in counts.items()
        }

    def get_cleaning_stats(self) -> Dict:
        """Return accumulated cleaning statistics."""
        return dict(self.stats)


def analyze_text_quality(df: pd.DataFrame, text_column: str = "statement") -> Dict:
    """
    Exploratory analysis of raw text quality before cleaning.
    
    Returns dict with diagnostics including:
    - URL counts
    - Subreddit mention counts
    - Empty/bad row counts
    - Character length stats
    """
    if text_column not in df.columns:
        return {"error": f"Column '{text_column}' not found"}

    texts = df[text_column].dropna()

    return {
        "total_samples": len(df),
        "non_null_texts": len(texts),
        "null_texts": int(df[text_column].isna().sum()),
        "urls_found": int(texts.str.count(r'https?://\S+|www\.\S+').sum()),
        "subreddits_found": int(texts.str.count(r'/?(?:r|R)/(\w+)').sum()),
        "usernames_found": int(texts.str.count(r'/?(?:u|U)/(\w+)').sum()),
        "avg_char_length": float(texts.str.len().mean()),
        "median_char_length": float(texts.str.len().median()),
        "min_char_length": int(texts.str.len().min()),
        "max_char_length": int(texts.str.len().max()),
        "empty_texts": int((texts.str.strip() == "").sum()),
    }
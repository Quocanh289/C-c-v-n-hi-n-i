"""Lightweight emotion feature inference used by the API and meta-classifier.

The project UI expects GoEmotions-shaped results, but a deployable checkpoint is
not bundled in this repository. This module provides stable rule-based scores
that can also feed the CSV-trained mental-health meta-classifier.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

GOEMOTIONS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

COARSE_EMOTIONS = [
    "admiration", "anger", "anxiety", "fear", "joy",
    "love", "sadness", "surprise", "neutral",
]

_FINE_TO_COARSE = {
    "admiration": "admiration", "approval": "admiration", "gratitude": "admiration",
    "pride": "admiration", "anger": "anger", "annoyance": "anger",
    "disapproval": "anger", "disgust": "anger", "nervousness": "anxiety",
    "confusion": "anxiety", "fear": "fear", "amusement": "joy",
    "excitement": "joy", "joy": "joy", "optimism": "joy", "relief": "joy",
    "caring": "love", "desire": "love", "love": "love",
    "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness",
    "remorse": "sadness", "sadness": "sadness", "surprise": "surprise",
    "curiosity": "surprise", "realization": "surprise", "neutral": "neutral",
}

_KEYWORDS = {
    "admiration": {"tuyet voi": 0.6, "xuat sac": 0.7, "amazing": 0.6, "admire": 0.65},
    "anger": {"tuc gian": 0.75, "cau gat": 0.55, "angry": 0.75, "furious": 0.85},
    "anxiety": {"lo lang": 0.75, "bat an": 0.6, "anxious": 0.75, "worried": 0.7, "stress": 0.45},
    "fear": {"so hai": 0.75, "hoang so": 0.8, "fear": 0.7, "afraid": 0.7, "panic": 0.7},
    "joy": {"vui": 0.65, "hanh phuc": 0.75, "happy": 0.7, "joy": 0.75, "excited": 0.65},
    "love": {"yeu": 0.7, "thuong": 0.55, "love": 0.75, "caring": 0.55},
    "sadness": {"buon": 0.7, "tuyet vong": 0.8, "tram cam": 0.8, "sad": 0.7, "depressed": 0.8, "hopeless": 0.8},
    "surprise": {"bat ngo": 0.65, "ngac nhien": 0.65, "surprise": 0.65, "shocked": 0.7},
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _language(text: str) -> str:
    return "vi" if re.search(r"[ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]", text.lower()) else "en"


class GoEmotionsInference:
    """Stable rule inference matching the response shape expected by routes."""

    is_loaded = True

    def classify(self, text: str, output_mode: Optional[str] = None) -> Dict:
        value = _normalize(text)
        coarse = {label: 0.0 for label in COARSE_EMOTIONS}
        for label, phrases in _KEYWORDS.items():
            coarse[label] = min(0.98, sum(weight for phrase, weight in phrases.items() if phrase in value))

        strongest = max((score for label, score in coarse.items() if label != "neutral"), default=0.0)
        coarse["neutral"] = round(max(0.02, 0.82 - strongest), 4)
        primary = max(coarse, key=coarse.get)
        confidence = coarse[primary]

        fine = {label: 0.0 for label in GOEMOTIONS_28}
        for label, score in coarse.items():
            matching = [fine_label for fine_label, group in _FINE_TO_COARSE.items() if group == label]
            if matching:
                fine[matching[0]] = score

        language = _language(text)
        label_type = "coarse" if output_mode == "coarse" or (output_mode is None and language == "vi") else "fine"
        return {
            "primary_emotion": primary,
            "confidence": confidence,
            "label_type": label_type,
            "language": language,
            "scores_28": fine,
            "scores_9": coarse,
            "toxicity_score": 0.0,
            "toxicity_binary": False,
            "sarcasm_score": 0.0,
            "sarcasm_binary": False,
            "source": "rule_emotion_features",
            "model": "rule_emotion_features",
            "num_labels": 9 if label_type == "coarse" else 28,
        }


_instance: Optional[GoEmotionsInference] = None


def get_inference() -> GoEmotionsInference:
    global _instance
    if _instance is None:
        _instance = GoEmotionsInference()
    return _instance

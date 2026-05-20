"""
Emotion Lens - Advanced Emotion Classification Module
=====================================================
Multi-class emotion classifier using XLM-RoBERTa fine-tuned on GoEmotions.
Supports 9 coarse emotion classes: admiration, anger, anxiety, fear,
joy, love, sadness, surprise, neutral.

This module provides:
- Model inference (predict emotion from text)
- Model loading from training pipeline checkpoints
- Fallback to lightweight rule-based baseline
- Integration with the training pipeline
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================
# Constants (aligned with training pipeline)
# ============================================================

COARSE_EMOTIONS = [
    "admiration", "anger", "anxiety", "fear", "joy",
    "love", "sadness", "surprise", "neutral"
]

EMOTION_TO_IDX = {label: idx for idx, label in enumerate(COARSE_EMOTIONS)}
IDX_TO_EMOTION = {idx: label for idx, label in enumerate(COARSE_EMOTIONS)}

NUM_EMOTIONS = len(COARSE_EMOTIONS)  # 9

# Default model path (from training pipeline output)
DEFAULT_MODEL_PATH = os.path.join("ai_nlp", "training", "models", "emotion_model")

# Fallback keyword-based emotion detection (for when model is unavailable)
FALLBACK_KEYWORDS: Dict[str, List[str]] = {
    "admiration": ["tuyệt vời", "xuất sắc", "awesome", "amazing", "great", "wonderful", "brilliant", "impressive", "respect", "admire"],
    "anger": ["angry", "furious", "mad", "tức giận", "phẫn nộ", "khó chịu", "annoying", "hate", "damn", "stupid", "idiot"],
    "anxiety": ["lo lắng", "bồn chồn", "worried", "anxious", "nervous", "scared", "frightened", "panicked", "uneasy", "stressed"],
    "fear": ["fear", "terrified", "horrified", "sợ hãi", "kinh hoàng", "đáng sợ", "nightmare", "terror", "dread"],
    "joy": ["happy", "joy", "excited", "delighted", "glad", "vui", "hạnh phúc", "tuyệt", "tuyệt vời", "awesome", "amazing", "wonderful"],
    "love": ["love", "adorable", "precious", "yêu", "thương", "quý", "đáng yêu", "beautiful", "cute", "sweet", "dear"],
    "sadness": ["sad", "unhappy", "depressed", "buồn", "thất vọng", "cô đơn", "alone", "lonely", "heartbroken", "cry", "tear"],
    "surprise": ["surprise", "shock", "wow", "unexpected", "ngạc nhiên", "bất ngờ", "shocked", "astonishing", "incredible", "unbelievable"],
    "neutral": [],
}

# Asian language character ranges
VIETNAMESE_CHARS = r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]"


# ============================================================
# Emotion Classifier
# ============================================================

class EmotionClassifier:
    """
    Emotion classifier with model inference and rule-based fallback.
    
    Uses a trained GoEmotions model for primary inference,
    with keyword-based fallback for robustness.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_fallback: bool = True,
        confidence_threshold: float = 0.3,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fallback = use_fallback
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Try to load the trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from pipeline output."""
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                logger.info(f"Model path {self.model_path} not found. Using fallback only.")
                return
            
            # Check for transformer model
            model_file = os.path.join(self.model_path, "pytorch_model.bin")
            config_file = os.path.join(self.model_path, "config.json")
            
            if not os.path.exists(model_file) and not os.path.exists(config_file):
                # Try training output directory
                training_dir = os.path.join("ai_nlp", "training", "checkpoints", "emotion_model", "best_model")
                if os.path.exists(training_dir):
                    self.model_path = training_dir
            
            # Load using HuggingFace transformers
            from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
            
            # Load config first to check labels
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    model_config = json.load(f)
                num_labels = model_config.get("num_labels", NUM_EMOTIONS)
            else:
                num_labels = NUM_EMOTIONS
            
            # Load model with proper number of labels
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_path)
            self.model = XLMRobertaForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model loaded from {self.model_path} on {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using fallback only.")
            self.is_loaded = False
    
    def classify(self, text: str) -> Dict:
        """
        Classify emotion from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with primary_emotion, confidence, all_scores
        """
        # Try model inference first
        if self.is_loaded and self.model is not None:
            try:
                return self._model_inference(text)
            except Exception as e:
                logger.warning(f"Model inference failed: {e}. Falling back.")
        
        # Fallback to keyword-based
        if self.use_fallback:
            return self._keyword_fallback(text)
        
        return {
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "all_scores": {emotion: 0.0 for emotion in COARSE_EMOTIONS},
            "source": "none",
        }
    
    @torch.no_grad()
    def _model_inference(self, text: str) -> Dict:
        """Run transformer model inference."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Build scores dictionary
        n_labels = min(len(probs), NUM_EMOTIONS)
        emotion_scores = {}
        for i in range(n_labels):
            emotion = IDX_TO_EMOTION.get(i, COARSE_EMOTIONS[i] if i < len(COARSE_EMOTIONS) else f"class_{i}")
            emotion_scores[emotion] = float(probs[i])
        
        # For any missing emotions (if model has fewer classes), add with 0
        for emotion in COARSE_EMOTIONS:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
        
        # Get primary emotion (highest non-neutral probability)
        max_score = 0.0
        primary = "neutral"
        
        for emotion, score in emotion_scores.items():
            if emotion != "neutral" and score > max_score:
                max_score = score
                primary = emotion
        
        # If all non-neutral scores are below threshold, use neutral
        if max_score < self.confidence_threshold:
            primary = "neutral"
            max_score = emotion_scores.get("neutral", 0.0)
        
        return {
            "primary_emotion": primary,
            "confidence": max_score,
            "all_scores": emotion_scores,
            "source": "model",
        }
    
    def _keyword_fallback(self, text: str) -> Dict:
        """Rule-based keyword matching as fallback."""
        text_lower = text.lower()
        scores = {emotion: 0.0 for emotion in COARSE_EMOTIONS}
        
        # Count keyword matches for each emotion
        for emotion, keywords in FALLBACK_KEYWORDS.items():
            if not keywords:
                continue
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > 0:
                scores[emotion] = min(1.0, match_count / 5.0)  # Normalize to [0, 1]
        
        # Determine primary emotion
        max_score = max(scores.values())
        if max_score > 0:
            primary = max(scores, key=scores.get)
        else:
            primary = "neutral"
        
        # Detect language
        has_vietnamese = bool(re.search(VIETNAMESE_CHARS, text_lower))
        
        return {
            "primary_emotion": primary,
            "confidence": max_score,
            "all_scores": scores,
            "source": "fallback_rules",
            "language": "vi" if has_vietnamese else "en",
        }


# ============================================================
# Module-level functions (backward compatibility)
# ============================================================

_classifier_instance: Optional[EmotionClassifier] = None


def get_classifier() -> EmotionClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier()
    return _classifier_instance


def classify_emotions(tokens: List[str]) -> List[str]:
    """
    Classify emotions from tokenized text.
    (Backward-compatible function)
    
    Args:
        tokens: List of token strings
        
    Returns:
        List of detected emotion labels
    """
    text = " ".join(tokens)
    classifier = get_classifier()
    result = classifier.classify(text)
    return [result["primary_emotion"]]


def predict_emotion(text: str) -> Dict:
    """
    Predict emotion from raw text.
    
    Args:
        text: Input text string
        
    Returns:
        Dict with emotion prediction results
    """
    classifier = get_classifier()
    result = classifier.classify(text)
    return result


def predict_batch(texts: List[str]) -> List[Dict]:
    """
    Predict emotions for multiple texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of prediction result dicts
    """
    results = []
    for text in texts:
        result = predict_emotion(text)
        results.append(result)
    return results
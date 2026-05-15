# ====================================================
# Multi-Task Emotion Model (XLM-RoBERTa)
# Base transformer with separate heads for:
# - Emotion classification (8 categories)
# - Toxicity detection (binary + score)
# - Sarcasm detection (binary + score)
# - Intent classification
# Supports LoRA fine-tuning for continuous learning
# ====================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

logger = logging.getLogger(__name__)

# ====================================================
# Constants
# ====================================================

EMOTION_LABELS = [
    "joy", "anger", "sadness", "anxiety",
    "fear", "surprise", "neutral", "toxic", "sarcastic"
]

EMOTION_LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
EMOTION_IDX_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

NUM_EMOTIONS = len(EMOTION_LABELS)  # 9

DEFAULT_MODEL_NAME = "xlm-roberta-base"  # Multilingual: supports EN + VI


# ====================================================
# Model Architecture
# ====================================================

class MultiTaskEmotionModel(nn.Module):
    """
    Multi-task learning architecture with:
    - Shared XLM-RoBERTa encoder
    - Emotion classification head (9-class)
    - Toxicity head (binary + score regression)
    - Sarcasm head (binary + score regression)
    - Intent classification head
    
    Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_emotions: int = NUM_EMOTIONS,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.use_lora = use_lora
        
        # Shared encoder
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        self.encoder = XLMRobertaModel.from_pretrained(model_name, config=self.config)
        
        # Apply LoRA if specified
        if use_lora:
            self._apply_lora(lora_r, lora_alpha)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific heads
        # 1. Emotion classification head (9 classes)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )
        
        # 2. Toxicity head (binary + score)
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # [binary_logit, score]
        )
        
        # 3. Sarcasm head (binary + score)
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # [binary_logit, score]
        )
        
        # 4. Intent classification head
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8),  # 8 intent types
        )
        
        # Loss functions
        self.emotion_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.toxicity_loss = nn.BCEWithLogitsLoss()
        self.sarcasm_loss = nn.BCEWithLogitsLoss()
        self.intent_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        logger.info(f"MultiTaskEmotionModel initialized with {model_name} "
                    f"(LoRA: {use_lora}, hidden_dim: {hidden_dim})")
    
    def _apply_lora(self, r: int = 8, alpha: int = 32):
        """
        Apply LoRA to query and value projection matrices.
        This enables efficient fine-tuning with minimal parameters.
        """
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Add LoRA layers to attention projections
        # In practice, we'd use a LoRA wrapper class
        # This is a simplified implementation
        lora_params = []
        
        for name, module in self.encoder.named_modules():
            if 'attention.self.query' in name or 'attention.self.value' in name:
                if isinstance(module, nn.Linear):
                    # Add LoRA: W' = W + BA where B in R^(d×r), A in R^(r×d')
                    lora_a = nn.Linear(module.in_features, r, bias=False)
                    lora_b = nn.Linear(r, module.out_features, bias=False)
                    
                    # Initialize A with Kaiming, B with zeros
                    nn.init.kaiming_uniform_(lora_a.weight, a=5 ** 0.5)
                    nn.init.zeros_(lora_b.weight)
                    
                    # Scale by alpha/r
                    scale = alpha / r
                    lora_b.weight.data *= scale
                    
                    # Store LoRA layers
                    setattr(module, 'lora_a', lora_a)
                    setattr(module, 'lora_b', lora_b)
                    lora_params.extend([lora_a.weight, lora_b.weight])
        
        # Only train LoRA parameters
        self.lora_params = lora_params
        logger.info(f"Applied LoRA with r={r}, alpha={alpha}. "
                    f"Trainable parameters: {sum(p.numel() for p in lora_params):,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional) [batch_size, seq_len]
            
        Returns:
            Dict with keys: emotion_logits, toxicity_logits, 
                            sarcasm_logits, intent_logits, pooled_output
        """
        # Shared encoder forward
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token representation (pooled output)
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        pooled = self.dropout(pooled)
        
        # Task-specific heads
        emotion_logits = self.emotion_head(pooled)   # [batch_size, 9]
        toxicity_logits = self.toxicity_head(pooled)  # [batch_size, 2]
        sarcasm_logits = self.sarcasm_head(pooled)    # [batch_size, 2]
        intent_logits = self.intent_head(pooled)      # [batch_size, 8]
        
        return {
            "emotion_logits": emotion_logits,
            "toxicity_logits": toxicity_logits,
            "sarcasm_logits": sarcasm_logits,
            "intent_logits": intent_logits,
            "pooled_output": pooled,
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        emotion_labels: Optional[torch.Tensor] = None,
        toxicity_labels: Optional[torch.Tensor] = None,
        sarcasm_labels: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.
        
        Args:
            outputs: Model outputs from forward()
            emotion_labels: [batch_size] with class indices
            toxicity_labels: [batch_size, 2] for [binary, score]
            sarcasm_labels: [batch_size, 2] for [binary, score]
            intent_labels: [batch_size] with class indices
            task_weights: Dict with weights for each task loss
            
        Returns:
            Dict with individual losses and total loss
        """
        if task_weights is None:
            task_weights = {
                "emotion": 1.0,
                "toxicity": 0.5,
                "sarcasm": 0.5,
                "intent": 0.3,
            }
        
        losses = {}
        total_loss = 0.0
        
        if emotion_labels is not None:
            loss = self.emotion_loss(outputs["emotion_logits"], emotion_labels)
            losses["emotion_loss"] = loss * task_weights["emotion"]
            total_loss += losses["emotion_loss"]
        
        if toxicity_labels is not None:
            loss = self.toxicity_loss(outputs["toxicity_logits"], toxicity_labels)
            losses["toxicity_loss"] = loss * task_weights["toxicity"]
            total_loss += losses["toxicity_loss"]
        
        if sarcasm_labels is not None:
            loss = self.sarcasm_loss(outputs["sarcasm_logits"], sarcasm_labels)
            losses["sarcasm_loss"] = loss * task_weights["sarcasm"]
            total_loss += losses["sarcasm_loss"]
        
        if intent_labels is not None:
            loss = self.intent_loss(outputs["intent_logits"], intent_labels)
            losses["intent_loss"] = loss * task_weights["intent"]
            total_loss += losses["intent_loss"]
        
        losses["total_loss"] = total_loss
        return losses
    
    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return probabilities.
        """
        outputs = self.forward(input_ids, attention_mask)
        
        return {
            "emotion_probs": F.softmax(outputs["emotion_logits"], dim=-1),
            "toxicity_probs": torch.sigmoid(outputs["toxicity_logits"]),
            "sarcasm_probs": torch.sigmoid(outputs["sarcasm_logits"]),
            "intent_probs": F.softmax(outputs["intent_logits"], dim=-1),
            "emotion_logits": outputs["emotion_logits"],
        }
    
    def save_pretrained(self, save_path: str):
        """Save model weights and config."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_emotions": self.num_emotions,
            "hidden_dim": self.hidden_dim,
            "use_lora": self.use_lora,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> "MultiTaskEmotionModel":
        """Load model from saved weights."""
        # Load config
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        
        # Initialize model
        model = cls(
            model_name=config.get("model_name", DEFAULT_MODEL_NAME),
            num_emotions=config.get("num_emotions", NUM_EMOTIONS),
            hidden_dim=config.get("hidden_dim", 768),
            use_lora=config.get("use_lora", True),
        )
        
        # Load weights
        weights_path = os.path.join(load_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model loaded from {load_path}")
        
        return model


# ====================================================
# Model Manager
# ====================================================

@dataclass
class InferenceResult:
    """Result from model inference."""
    emotions: Dict[str, float] = field(default_factory=dict)
    primary_emotion: str = "neutral"
    toxicity_score: float = 0.0
    toxicity_binary: bool = False
    sarcasm_score: float = 0.0
    sarcasm_binary: bool = False
    intent: str = "unknown"
    confidence: float = 0.0
    language: str = "en"
    processing_time_ms: float = 0.0


class EmotionModelManager:
    """
    Manages the emotion model lifecycle: loading, inference, 
    caching, and fine-tuning.
    """
    
    def __init__(
        self,
        model_path: str = "models/emotion_model",
        device: str = "cpu",
        use_cache: bool = True,
        cache_size: int = 1000,
    ):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        self.cache: Dict[str, InferenceResult] = {}
        self.cache_size = cache_size
        
        self.model: Optional[MultiTaskEmotionModel] = None
        self.tokenizer: Optional[XLMRobertaTokenizer] = None
        self.is_loaded = False
        
        # Language detection helper
        self.vi_char_pattern = r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]"
    
    def load_model(self):
        """Load model and tokenizer from disk or HuggingFace."""
        if self.is_loaded:
            return
        
        # Try loading from local path first
        local_path = self.model_path
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "pytorch_model.bin")):
            self.model = MultiTaskEmotionModel.from_pretrained(local_path)
            tokenizer_name = self.model.model_name if self.model else DEFAULT_MODEL_NAME
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        else:
            # Load pretrained from HuggingFace
            logger.info(f"Loading model from HuggingFace: {DEFAULT_MODEL_NAME}")
            self.model = MultiTaskEmotionModel(model_name=DEFAULT_MODEL_NAME, use_lora=False)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        logger.info(f"Model loaded on {self.device}")
    
    def unload_model(self):
        """Free up GPU memory."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Vietnamese, English, or mixed."""
        import re
        vi_matches = len(re.findall(self.vi_char_pattern, text.lower()))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return "en"
        
        vi_ratio = vi_matches / total_chars
        
        if vi_ratio > 0.15:
            # Check for significant English words
            en_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
            if en_words > 0 and en_words / max(len(text.split()), 1) > 0.3:
                return "mixed"
            return "vi"
        
        return "en"
    
    @torch.no_grad()
    def analyze(
        self,
        text: str,
        return_all_probs: bool = False,
    ) -> InferenceResult:
        """
        Analyze a single text for emotions.
        Uses caching for performance.
        """
        import time
        start_time = time.time()
        
        # Check cache
        if self.use_cache and text in self.cache:
            result = self.cache[text]
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Run inference
        predictions = self.model.predict(input_ids, attention_mask)
        
        # Process emotion probabilities
        emotion_probs = predictions["emotion_probs"][0].cpu().numpy()
        emotion_scores = {
            EMOTION_IDX_TO_LABEL[i]: float(prob)
            for i, prob in enumerate(emotion_probs)
        }
        
        # Get primary emotion (highest probability, excluding neutral weighting)
        primary_idx = emotion_probs[1:].argmax() + 1  # Skip neutral for primary
        primary_confidence = float(emotion_probs[primary_idx])
        
        # But check neutral if confidence is low
        if primary_confidence < 0.3:
            neutral_prob = float(emotion_probs[EMOTION_LABEL_TO_IDX["neutral"]])
            if neutral_prob > primary_confidence:
                primary_idx = EMOTION_LABEL_TO_IDX["neutral"]
                primary_confidence = neutral_prob
        
        primary_emotion = EMOTION_IDX_TO_LABEL[primary_idx]
        
        # Process toxicity
        toxicity_probs = predictions["toxicity_probs"][0].cpu().numpy()
        toxicity_score = float(toxicity_probs[1])  # score output
        toxicity_binary = bool(toxicity_probs[0] > 0.5)  # binary output
        
        # Process sarcasm
        sarcasm_probs = predictions["sarcasm_probs"][0].cpu().numpy()
        sarcasm_score = float(sarcasm_probs[1])
        sarcasm_binary = bool(sarcasm_probs[0] > 0.5)
        
        # Process intent
        intent_probs = predictions["intent_probs"][0].cpu().numpy()
        intent_idx = int(intent_probs.argmax())
        intent_labels = [
            "statement", "question", "request", "opinion",
            "complaint", "praise", "joke", "sarcasm",
        ]
        intent = intent_labels[intent_idx] if intent_idx < len(intent_labels) else "unknown"
        
        # Detect language
        language = self.detect_language(text)
        
        # Build result
        result = InferenceResult(
            emotions=emotion_scores,
            primary_emotion=primary_emotion,
            toxicity_score=toxicity_score,
            toxicity_binary=toxicity_binary,
            sarcasm_score=sarcasm_score,
            sarcasm_binary=sarcasm_binary,
            intent=intent,
            confidence=primary_confidence,
            language=language,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Cache result
        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[text] = result
        
        return result
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[InferenceResult]:
        """
        Analyze multiple texts in batch for efficiency.
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache first
            uncached = []
            cached_results = []
            
            for text in batch_texts:
                if self.use_cache and text in self.cache:
                    cached_results.append(self.cache[text])
                else:
                    uncached.append(text)
            
            results.extend(cached_results)
            
            # Process uncached texts
            if uncached:
                for text in uncached:
                    result = self.analyze(text)
                    results.append(result)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_usage_pct": (len(self.cache) / self.cache_size) * 100 if self.cache_size > 0 else 0,
        }
    
    def clear_cache(self):
        """Clear inference cache."""
        self.cache.clear()
        logger.info("Inference cache cleared")
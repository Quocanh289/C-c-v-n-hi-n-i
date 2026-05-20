"""
Model Module
=============
Multi-label emotion classification model supporting:
- XLM-RoBERTa-base and RoBERTa-base
- Full fine-tuning and LoRA fine-tuning
- Single-head (28-label) and dual-head (28+9) architectures
- Multi-label sigmoid output (independent binary per label)
- Gradient checkpointing for memory efficiency
- Model comparison utilities

Architecture:
    Base encoder → dropout → linear head(s) → sigmoid multi-label outputs
    
    Single-head: [CLS] → linear(768, 28) → sigmoid → 28 emotion probabilities
    Dual-head:   [CLS] → linear(768, 37) → split → 28 fine + 9 coarse probabilities
"""

import os
import json
import copy
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from .config import (
    TrainingConfig,
    GOEMOTIONS_28,
    COARSE_EMOTIONS,
    GOEMOTIONS_28_TO_IDX,
    COARSE_TO_IDX,
)

logger = logging.getLogger(__name__)


# ============================================================
# Multi-Label Emotion Classification Model
# ============================================================

class GoEmotionsModel(nn.Module):
    """
    Transformer-based multi-label emotion classifier for GoEmotions.
    
    Supports:
    - Multi-label output (28 independent sigmoids)
    - Single-label output (9-class softmax, backward compat)
    - Dual-head output (28 fine + 9 coarse, both sigmoid)
    - LoRA fine-tuning
    - Gradient checkpointing
    - Both XLM-RoBERTa and RoBERTa base models
    
    Output configurations:
        "multi_label": output_dim = 28 (sigmoid, multi-hot)
        "multi_class": output_dim = 9  (softmax, single class)
        "dual_head":   output_dim = 37 (28 sigmoid + 9 sigmoid)
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model_name: Optional[str] = None,
        num_labels: Optional[int] = None,
        num_coarse_labels: Optional[int] = None,
    ):
        super().__init__()
        
        self.cfg = config
        self.model_name = model_name or config.model_name
        self.num_labels = num_labels or config.num_labels  # 28
        self.num_coarse = num_coarse_labels or config.num_coarse_labels  # 9
        self.task_type = config.task_type
        
        # Total output dimension
        if self.task_type == "dual_head":
            self.output_dim = self.num_labels + self.num_coarse  # 37
        else:
            self.output_dim = self.num_labels  # 28 (or 9 for multi_class)
        
        # Determine model type
        base_model_name = self.model_name
        logger.info(f"Loading base model: {base_model_name}")
        
        # Load base model using AutoModel for flexibility
        # Support both RobertaModel and XLMRobertaModel
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        
        # Get hidden dimension from the model config
        self.hidden_dim = self.base_model.config.hidden_size
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply LoRA if configured
        if config.use_lora:
            self._apply_lora()
        else:
            self.is_lora = False
        
        # Classification head(s)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Separate coarse head for dual_head mode
        self.coarse_classifier = None
        if self.task_type == "dual_head":
            self.coarse_classifier = nn.Linear(self.hidden_dim, self.num_coarse)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Model initialized: {self.model_name}")
        logger.info(f"  Task type: {self.task_type}")
        logger.info(f"  Output dim: {self.output_dim}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        
        if config.use_lora:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logger.info(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def _init_weights(self):
        """Initialize classifier weights with normal distribution."""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        if self.coarse_classifier is not None:
            self.coarse_classifier.weight.data.normal_(mean=0.0, std=0.02)
            if self.coarse_classifier.bias is not None:
                self.coarse_classifier.bias.data.zero_()
    
    def _apply_lora(self):
        """Apply LoRA using PEFT library."""
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self._get_lora_targets(),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()
        self.is_lora = True
        self.lora_config = lora_config
    
    def _get_lora_targets(self) -> List[str]:
        """
        Get LoRA target modules based on model type.
        RoBERTa and XLM-RoBERTa have different module names.
        """
        # Default targets that work for both
        targets = ["query", "value"]
        
        # Detect model type from config
        model_type = getattr(self.base_model.config, "model_type", "").lower()
        
        if "roberta" in model_type:
            targets = ["query", "key", "value", "output.dense"]
        elif "xlm" in model_type or "xlm-roberta" in model_type:
            targets = ["query", "key", "value", "output.dense"]
        elif "deberta" in model_type:
            targets = ["query_proj", "key_proj", "value_proj", "output.dense"]
        else:
            # Fallback to attention modules
            targets = ["query", "value"]
        
        return self.cfg.lora_target_modules or targets
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Dict with keys depending on task_type:
            - 'logits': Main output logits
            - 'probs': Sigmoid/softmax probabilities
            - 'coarse_logits': Coarse logits (dual_head only)
            - 'coarse_probs': Coarse probabilities (dual_head only)
            - 'pooled': Pooled features
        """
        # Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use pooled [CLS] output
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        pooled = self.dropout(pooled)
        
        # Main classification head
        logits = self.classifier(pooled)  # [batch_size, output_dim]
        
        result = {"logits": logits, "pooled": pooled}
        
        # Apply appropriate activation
        if self.task_type == "multi_class":
            result["probs"] = F.softmax(logits, dim=-1)
        else:
            # Multi-label: sigmoid for independent binary classification
            result["probs"] = torch.sigmoid(logits)
            
            # Split for dual head
            if self.task_type == "dual_head":
                fine_logits = logits[:, :self.num_labels]
                coarse_logits = logits[:, self.num_labels:]
                
                # Also compute from separate coarse head if available
                if self.coarse_classifier is not None:
                    coarse_logits = self.coarse_classifier(pooled)
                
                result["fine_logits"] = fine_logits
                result["coarse_logits"] = coarse_logits
                result["fine_probs"] = torch.sigmoid(fine_logits)
                result["coarse_probs"] = torch.sigmoid(coarse_logits)
        
        return result
    
    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thresholds: Optional[Union[float, List[float]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return predictions.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            thresholds: Per-label thresholds for multi-label (default: 0.5)
        
        Returns:
            Dict with predictions, probabilities, logits
        """
        outputs = self.forward(input_ids, attention_mask)
        
        if self.task_type == "multi_class":
            predictions = torch.argmax(outputs["probs"], dim=-1)
        else:
            # Multi-label: threshold sigmoid probabilities
            if thresholds is not None:
                thresh = torch.tensor(thresholds, device=input_ids.device)
            else:
                thresh = 0.5
            predictions = (outputs["probs"] > thresh).float()
        
        return {
            "logits": outputs["logits"],
            "probs": outputs["probs"],
            "predictions": predictions,
        }
    
    def save_pretrained(self, save_path: str):
        """Save model, tokenizer, and config."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model (handles LoRA weights too)
        self.base_model.save_pretrained(save_path)
        
        # Save classifier heads
        torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
        if self.coarse_classifier is not None:
            torch.save(
                self.coarse_classifier.state_dict(),
                os.path.join(save_path, "coarse_classifier.pt"),
            )
        
        # Save model config
        config_data = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "num_coarse": self.num_coarse,
            "output_dim": self.output_dim,
            "task_type": self.task_type,
            "use_lora": self.cfg.use_lora,
            "labels_28": GOEMOTIONS_28,
            "labels_9": COARSE_EMOTIONS,
            "label2id_28": GOEMOTIONS_28_TO_IDX,
            "label2id_9": COARSE_TO_IDX,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, config: TrainingConfig) -> "GoEmotionsModel":
        """Load model from saved weights."""
        from .config import GOEMOTIONS_28, COARSE_EMOTIONS
        
        # Load saved config
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                saved_config = json.load(f)
            # Override config with saved values
            config.num_labels = saved_config.get("num_labels", config.num_labels)
            config.num_coarse_labels = saved_config.get("num_coarse", config.num_coarse_labels)
            config.task_type = saved_config.get("task_type", config.task_type)
        
        # Initialize model
        model = cls(config=config)
        
        # Load classifier weights
        classifier_path = os.path.join(load_path, "classifier.pt")
        if os.path.exists(classifier_path):
            state_dict = torch.load(classifier_path, map_location="cpu")
            if state_dict["weight"].size(0) != model.classifier.weight.size(0):
                logger.warning(f"Classifier weight mismatch: saved={state_dict['weight'].size(0)}, model={model.classifier.weight.size(0)}. Reinitializing.")
            else:
                model.classifier.load_state_dict(state_dict)
                logger.info(f"Classifier loaded from {classifier_path}")
        
        # Load coarse classifier
        coarse_path = os.path.join(load_path, "coarse_classifier.pt")
        if os.path.exists(coarse_path) and model.coarse_classifier is not None:
            model.coarse_classifier.load_state_dict(
                torch.load(coarse_path, map_location="cpu")
            )
            logger.info(f"Coarse classifier loaded from {coarse_path}")
        
        return model


# ============================================================
# Model & Tokenizer Factory
# ============================================================

def get_model_and_tokenizer(
    config: TrainingConfig,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    load_path: Optional[str] = None,
) -> Tuple[GoEmotionsModel, AutoTokenizer]:
    """
    Create or load model and tokenizer.
    
    Args:
        config: Training configuration
        model_name: HuggingFace model name
        num_labels: Number of emotion labels
        load_path: Path to load existing model from
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or config.model_name
    num_labels = num_labels or config.num_labels
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing (common for GPT2-style tokenizers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    
    # Create or load model
    if load_path and os.path.exists(load_path):
        logger.info(f"Loading model from {load_path}")
        model = GoEmotionsModel.from_pretrained(load_path, config)
    else:
        logger.info(f"Creating new model: {model_name}")
        model = GoEmotionsModel(
            config=config,
            model_name=model_name,
            num_labels=num_labels,
        )
    
    return model, tokenizer


# ============================================================
# Model Comparison Utilities
# ============================================================

class ModelComparisonResult:
    """Container for model comparison results."""
    
    def __init__(
        self,
        model_a_name: str,
        model_b_name: str,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
    ):
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.metrics_a = metrics_a
        self.metrics_b = metrics_b
    
    @property
    def winner(self) -> str:
        """Determine the better model based on macro F1."""
        score_a = self.metrics_a.get("macro_f1_micro_avg", 0)
        score_b = self.metrics_b.get("macro_f1_micro_avg", 0)
        return self.model_a_name if score_a >= score_b else self.model_b_name
    
    def print_summary(self):
        """Print comparison summary."""
        print(f"\n{'='*60}")
        print(f"Model Comparison: {self.model_a_name} vs {self.model_b_name}")
        print(f"{'='*60}")
        
        all_metrics = set(list(self.metrics_a.keys()) + list(self.metrics_b.keys()))
        
        for metric in sorted(all_metrics):
            val_a = self.metrics_a.get(metric, float('nan'))
            val_b = self.metrics_b.get(metric, float('nan'))
            diff = val_a - val_b
            marker = "✓" if diff > 0 else ("✗" if diff < 0 else "=")
            print(f"  {metric:30s}: {val_a:.4f} vs {val_b:.4f} ({diff:+.4f}) {marker}")
        
        print(f"\n  Winner: {self.winner}")
        print(f"{'='*60}\n")
    
    def to_dict(self) -> Dict:
        return {
            "model_a": {"name": self.model_a_name, "metrics": self.metrics_a},
            "model_b": {"name": self.model_b_name, "metrics": self.metrics_b},
            "winner": self.winner,
        }
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compare_models(
    config: TrainingConfig,
    metric_fn,
    test_loader,
    device: torch.device,
) -> ModelComparisonResult:
    """
    Compare RoBERTa-base vs XLM-RoBERTa-base on test data.
    
    Args:
        config: Base training config
        metric_fn: Function that takes (model, loader) and returns metrics dict
        test_loader: Test data loader
        device: Torch device
    
    Returns:
        ModelComparisonResult with comparison data
    """
    models_to_test = [
        ("roberta-base", "RoBERTa-base (English-only)"),
        ("FacebookAI/xlm-roberta-base", "XLM-RoBERTa-base (Multilingual)"),
    ]
    
    results = {}
    for model_name, display_name in models_to_test:
        logger.info(f"\nTesting model: {display_name} ({model_name})")
        
        # Override model name
        cmp_config = copy.deepcopy(config)
        cmp_config.model_name = model_name
        
        # Create model
        model, _ = get_model_and_tokenizer(cmp_config, model_name=model_name)
        model.to(device)
        
        # Evaluate
        metrics = metric_fn(model, test_loader)
        results[display_name] = metrics
        
        logger.info(f"  Results for {display_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.4f}")
    
    model_names = list(results.keys())
    return ModelComparisonResult(
        model_a_name=model_names[0],
        model_b_name=model_names[1],
        metrics_a=results[model_names[0]],
        metrics_b=results[model_names[1]],
    )
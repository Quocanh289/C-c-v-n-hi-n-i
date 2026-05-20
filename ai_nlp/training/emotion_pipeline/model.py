"""
Model Module
=============
XLM-RoBERTa with LoRA fine-tuning for GoEmotions emotion classification.
Provides model creation, tokenizer loading, and inference utilities.
"""

import os
import json
import logging
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from .config import TrainingConfig, COARSE_EMOTIONS, COARSE_TO_IDX

logger = logging.getLogger(__name__)


# ============================================================
# GoEmotions Classification Model
# ============================================================

class GoEmotionsModel(nn.Module):
    """
    XLM-RoBERTa-based emotion classifier for GoEmotions (9 classes).
    Supports optional LoRA fine-tuning and gradient checkpointing.
    
    Architecture:
        - Shared XLM-RoBERTa encoder
        - Pooled [CLS] representation
        - Classification head (hidden_dim → 9 emotions)
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model_name: Optional[str] = None,
        num_labels: Optional[int] = None,
    ):
        super().__init__()
        
        self.cfg = config
        self.model_name = model_name or config.model_name
        self.num_labels = num_labels or config.num_labels
        
        # Load base model
        base_model_name = self.model_name
        logger.info(f"Loading base model: {base_model_name}")
        
        # Use HuggingFace XLMRobertaModel directly
        self.base_model = XLMRobertaModel.from_pretrained(
            base_model_name,
            num_labels=self.num_labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        
        # Get hidden dimension
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
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        
        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        logger.info(f"Model initialized: {self.model_name}, {self.num_labels} classes")
        if config.use_lora:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def _apply_lora(self):
        """Apply LoRA using PEFT library."""
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()
        self.is_lora = True
        
        # Store config
        self.lora_config = lora_config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: Optional [batch_size, seq_len]
        
        Returns:
            Dict with 'logits' and optionally 'hidden_states'
        """
        # Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use pooled [CLS] output
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)  # [batch_size, num_labels]
        
        return {"logits": logits, "pooled": pooled}
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference and return probabilities.
        """
        outputs = self.forward(input_ids, attention_mask)
        probs = F.softmax(outputs["logits"], dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        
        return {
            "logits": outputs["logits"],
            "probs": probs,
            "predictions": predictions,
        }
    
    def save_pretrained(self, save_path: str):
        """Save model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model (handles LoRA weights too if using PEFT)
        self.base_model.save_pretrained(save_path)
        
        # Save classifier separately
        torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
        
        # Save config
        config_path = os.path.join(save_path, "config.json")
        # Save minimal config for reloading
        config_data = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "use_lora": self.cfg.use_lora,
            "labels": COARSE_EMOTIONS,
            "label2id": COARSE_TO_IDX,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, config: TrainingConfig) -> "GoEmotionsModel":
        """Load model from saved weights."""
        # Load config
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_config = json.load(f)
        
        # Initialize model
        model = cls(config=config)
        
        # Load classifier weights
        classifier_path = os.path.join(load_path, "classifier.pt")
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
            logger.info(f"Classifier loaded from {classifier_path}")
        
        return model


# ============================================================
# Model & Tokenizer Factory
# ============================================================

def get_model_and_tokenizer(
    config: TrainingConfig,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    load_path: Optional[str] = None,
) -> Tuple[GoEmotionsModel, XLMRobertaTokenizer]:
    """
    Create or load model and tokenizer.
    
    Args:
        config: Training configuration
        model_name: HuggingFace model name
        num_labels: Number of emotion classes
        load_path: Path to load existing model from
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or config.model_name
    num_labels = num_labels or config.num_labels
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
"""
Model Module
=============
Mental Health Classifier using DeBERTa-v3-base + LoRA.

Architecture:
  - Base: DeBERTa-v3-base (184M params) or XLM-RoBERTa-base (279M params)
  - PEFT: LoRA (Low-Rank Adaptation) — only trains ~0.5M params
  - Head: Dropout → Linear(768, 7) → Softmax

Rationale for DeBERTa-v3-base over XLM-RoBERTa-base:
  1. DeBERTa-v3 uses disentangled attention → better at understanding nuanced
     mental health language (e.g., "I'm fine" vs actual distress signals)
  2. DeBERTa-v3 has enhanced masked decoder → better at context understanding
     of long Reddit posts
  3. Smaller model (184M vs 279M) → faster training, less VRAM
  4. Consistently outperforms RoBERTa on GLUE/SuperGLUE
  5. For English-only mental health detection, multilingual XLM is unnecessary
  6. RTX 4060 16GB: DeBERTa-v3 + LoRA fits comfortably at batch 16, 256 tokens

For multilingual (e.g., Vietnamese adaptation later):
  - Use XLM-RoBERTa-base: "FacebookAI/xlm-roberta-base"
  - Or PhoBERT: "vinai/phobert-base" (trained separately on Vietnamese data)
"""

import os
import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    DebertaV2ForSequenceClassification,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)

from .config import MENTAL_HEALTH_LABELS, NUM_MH_CLASSES, TrainingConfig

logger = logging.getLogger(__name__)


class MentalHealthClassifier(nn.Module):
    """
    Mental Health Classifier with DeBERTa-v3-base + LoRA.
    
    Features:
    - PEFT LoRA fine-tuning (only ~0.5M trainable params)
    - Custom classification head with configurable dropout
    - Gradient checkpointing for VRAM efficiency
    - Supports both DeBERTa-v3 and XLM-RoBERTa backbones
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # Build the model
        self._build_model()
        
        # Count parameters
        self._log_parameter_count()
    
    def _build_model(self):
        """Build the base model with LoRA adapter."""
        logger.info(f"Building model: {self.config.model_name}")
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        
        # Load base model in fp32 ALWAYS.
        # fp16 is handled by autocast + GradScaler in the trainer's train_epoch().
        # DO NOT set torch_dtype=float16 here — it causes "Attempting to unscale FP16 gradients"
        # because the model parameters are already fp16, making GradScaler's unscaling invalid.
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=model_config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )
        
        # Check if it's DeBERTa-v3 (has different layer naming)
        is_deberta = "deberta" in self.config.model_name.lower()
        is_xlmr = "xlm-roberta" in self.config.model_name.lower()
        
        logger.info(f"Model type: {'DeBERTa-v3' if is_deberta else 'XLM-RoBERTa' if is_xlmr else 'Other'}")
        
        # Apply gradient checkpointing to save VRAM
        if self.config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply LoRA
        if self.config.use_lora:
            self._apply_lora(is_deberta=is_deberta)
        
        # Store model on correct device
        self.device = torch.device(self.config.device)
        self.to(self.device)
    
    def _apply_lora(self, is_deberta: bool = False):
        """
        Apply LoRA configuration to the base model.
        
        For DeBERTa-v3: target_modules=["query_proj", "value_proj", "key_proj", "output_proj"]
        For RoBERTa/XLM-R: target_modules=["query", "value", "key", "output.dense"]
        """
        # Determine target modules based on model architecture
        if is_deberta:
            target_modules = ["query_proj", "value_proj", "key_proj", "output_proj"]
        else:
            # XLM-RoBERTa/RoBERTa
            target_modules = ["query", "value", "key", "output.dense"]
        
        # Use custom target modules if provided
        if self.config.lora_target_modules:
            target_modules = self.config.lora_target_modules
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.SEQ_CLS,
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        logger.info(f"LoRA applied: r={self.config.lora_r}, alpha={self.config.lora_alpha}, targets={target_modules}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Ground truth labels [batch] (optional)
            
        Returns:
            Dict with logits, loss (if labels provided), probs
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probs": probs,
        }
        
        if labels is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get raw logits for inference."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    
    def get_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get probabilities for inference."""
        logits = self.get_logits(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and confidences.
        
        Returns:
            (predictions, confidences) where predictions are class indices
        """
        probs = self.get_probs(input_ids, attention_mask)
        predictions = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values
        return predictions, confidences
    
    def _log_parameter_count(self):
        """Log parameter counts for debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(
            f"Parameters: Total={total_params:,} "
            f"({total_params/1e6:.1f}M), "
            f"Trainable={trainable_params:,} "
            f"({trainable_params/1e6:.1f}M, "
            f"{100*trainable_params/total_params:.2f}%)"
        )
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def save_pretrained(self, save_dir: str):
        """
        Save the model (base + LoRA adapter + classifier head).
        
        Args:
            save_dir: Directory to save model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save LoRA adapter
        if self.config.use_lora:
            self.base_model.save_pretrained(save_dir)
            logger.info(f"LoRA adapter saved to {save_dir}")
        else:
            self.base_model.save_pretrained(save_dir)
            logger.info(f"Full model saved to {save_dir}")
        
        # Save label mapping
        import json
        label_mapping = {
            "id2label": {str(i): l for i, l in enumerate(MENTAL_HEALTH_LABELS)},
            "label2id": {l: i for i, l in enumerate(MENTAL_HEALTH_LABELS)},
            "num_labels": self.num_labels,
        }
        with open(os.path.join(save_dir, "label_mapping.json"), "w") as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info(f"Label mapping saved to {save_dir}")


def build_model(config: TrainingConfig) -> MentalHealthClassifier:
    """
    Convenience function to build the mental health classifier.
    
    Args:
        config: TrainingConfig with model settings
        
    Returns:
        Initialized MentalHealthClassifier
    """
    return MentalHealthClassifier(config)


def load_trained_model(
    model_path: str,
    config: Optional[TrainingConfig] = None,
) -> MentalHealthClassifier:
    """
    Load a trained model from checkpoint for inference.
    
    Args:
        model_path: Path to saved model (LoRA adapter + classifier head)
        config: Optional TrainingConfig (will load from model_path if available)
        
    Returns:
        Loaded MentalHealthClassifier in eval mode
    """
    if config is None:
        config = TrainingConfig()
    
    # Build model architecture
    model = MentalHealthClassifier(config)
    
    # Load LoRA adapter weights
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {model_path}")
        # PeftModel requires loading from the base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        model.base_model = PeftModel.from_pretrained(base_model, model_path)
    else:
        logger.warning(f"No adapter found at {model_path}, using untrained model")
    
    model.eval()
    return model
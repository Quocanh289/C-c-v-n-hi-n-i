"""
Model Module
=============
Mental Health Classifier using DeBERTa-v3-base + LoRA.
Plus Ensemble version combining DeBERTa-v3, XLM-RoBERTa, and PhoBERT.

Architecture:
  - Base: DeBERTa-v3-base (184M params) or XLM-RoBERTa-base (279M params)
    or PhoBERT-base (135M params)
  - PEFT: LoRA (Low-Rank Adaptation) — only trains ~0.5M params
  - Head: Dropout -> Linear(768, 7) -> Softmax

Ensemble:
  - Trains 3 models independently (sequentially to fit RTX 4060 16GB)
  - Inference: weighted average of all 3 softmax outputs
  - Weights: DeBERTa=0.4, XLM-R=0.35, PhoBERT=0.25
  - Each model catches different linguistic patterns, ensemble reduces variance
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from .config import (
    MENTAL_HEALTH_LABELS,
    MENTAL_HEALTH_LABELS_TO_IDX,
    IDX_TO_MH_LABELS,
    NUM_MH_CLASSES,
    TrainingConfig,
    ENSEMBLE_MODELS,
)

logger = logging.getLogger(__name__)


class MentalHealthClassifier(nn.Module):
    """
    Mental Health Classifier with Transformer + LoRA.
    
    Supports:
    - DeBERTa-v3-base
    - XLM-RoBERTa-base
    - PhoBERT-base
    - Any AutoModelForSequenceClassification compatible model
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.model_name = config.model_name
        
        # Build the model
        self._build_model()
        
        # Count parameters
        self._log_parameter_count()
    
    def _build_model(self):
        """Build the base model with LoRA adapter."""
        logger.info(f"Building model: {self.model_name}")
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        
        # Load base model in fp32 ALWAYS.
        # fp16 is handled by autocast + GradScaler in the trainer's train_epoch().
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=model_config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )
        
        # Check model architecture
        is_deberta = "deberta" in self.model_name.lower()
        is_xlmr = "xlm-roberta" in self.model_name.lower()
        is_phobert = "phobert" in self.model_name.lower()
        
        logger.info(f"Model type: {'DeBERTa-v3' if is_deberta else 'XLM-RoBERTa' if is_xlmr else 'PhoBERT' if is_phobert else 'Other'}")
        
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
        For RoBERTa/XLM-R/PhoBERT: target_modules=["query", "value", "key", "output.dense"]
        """
        # Determine target modules based on model architecture
        if is_deberta:
            target_modules = ["query_proj", "value_proj", "key_proj", "output_proj"]
        else:
            # XLM-RoBERTa/RoBERTa/PhoBERT
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
        
        # Save label mapping and model info
        model_info = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "id2label": {str(i): l for i, l in enumerate(MENTAL_HEALTH_LABELS)},
            "label2id": {l: i for i, l in enumerate(MENTAL_HEALTH_LABELS)},
        }
        with open(os.path.join(save_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {save_dir}")


class EnsembleMentalHealthClassifier(nn.Module):
    """
    Ensemble of 3 mental health classifiers with weighted voting.
    
    Models:
    1. DeBERTa-v3-base (weight 0.40) — best English understanding
    2. XLM-RoBERTa-base (weight 0.35) — multilingual, different attention
    3. PhoBERT-base (weight 0.25) — syllable-level BPE, different tokenization
    
    These models have fundamentally different architectures and tokenization,
    so their errors are decorrelated → ensemble reduces variance significantly.
    
    Strategy:
    - Each model is trained independently (sequentially to fit RTX 4060 16GB)
    - At inference, all 3 are loaded and their probabilities are averaged
    - Weighted average: final_prob = Σ(weight_i * prob_i)
    - Only loads one model at a time to save VRAM during inference
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.num_labels = config.num_labels
        
        # Ensemble configuration
        self.model_keys = config.ensemble_model_keys
        self.weights = self._get_weights(config)
        
        # Models (lazy loaded)
        self.models: Dict[str, MentalHealthClassifier] = {}
        self.tokenizers: Dict[str, PreTrainedTokenizer] = {}
        
        logger.info(f"Ensemble initialized with {len(self.model_keys)} models")
        for key in self.model_keys:
            info = ENSEMBLE_MODELS[key]
            logger.info(f"  {key}: {info['name']} (weight={info['weight']})")
    
    def _get_weights(self, config: TrainingConfig) -> Dict[str, float]:
        """Get ensemble weights from config or defaults."""
        if config.ensemble_weights is not None:
            return config.ensemble_weights
        return {
            key: ENSEMBLE_MODELS[key]["weight"]
            for key in config.ensemble_model_keys
        }
    
    def load_model(self, model_key: str, checkpoint_dir: str):
        """
        Load a single ensemble model from its checkpoint.
        
        Each model is loaded separately to manage VRAM.
        
        Args:
            model_key: One of "deberta", "xlmr", "phobert"
            checkpoint_dir: Path to the model's checkpoint directory
        """
        if model_key in self.models:
            logger.info(f"Model {model_key} already loaded")
            return
        
        model_info = ENSEMBLE_MODELS[model_key]
        model_name = model_info["name"]
        
        logger.info(f"Loading ensemble model '{model_key}' ({model_name}) from {checkpoint_dir}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<pad>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        
        # Create a config specific to this model
        model_config = TrainingConfig()
        model_config.model_name = model_name
        model_config.num_labels = self.num_labels
        model_config.use_lora = True
        model_config.lora_target_modules = model_info["lora_target_modules"]
        
        # Build model architecture
        model = MentalHealthClassifier(model_config)
        
        # Load LoRA adapter weights
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_labels,
            )
            model.base_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        
        model.to(self.device)
        model.eval()
        
        self.models[model_key] = model
        self.tokenizers[model_key] = tokenizer
        
        logger.info(f"Loaded {model_key} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    
    def unload_model(self, model_key: str):
        """Unload a model from GPU to free VRAM."""
        if model_key in self.models:
            self.models[model_key].to("cpu")
            del self.models[model_key]
            del self.tokenizers[model_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_key} from GPU")
    
    def unload_all(self):
        """Unload all models to free VRAM."""
        for key in list(self.models.keys()):
            self.unload_model(key)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ensemble forward pass with weighted averaging.
        
        Requires all models to be loaded first via load_model().
        
        Args:
            input_ids: Token IDs (using DeBERTa's tokenizer)
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dict with:
              - "logits": aggregated logits
              - "probs": weighted average probabilities
              - "individual_probs": dict of per-model probabilities
              - "ensemble_weights": dict of per-model weights
              - "loss": optional loss
        """
        if not self.models:
            raise RuntimeError("No models loaded. Call load_model() for each model first.")
        
        # Get per-model probabilities
        individual_probs = {}
        all_probs_weighted = None
        total_weight = 0.0
        
        for key in self.model_keys:
            if key not in self.models:
                logger.warning(f"Model {key} not loaded, skipping")
                continue
            
            model = self.models[key]
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            probs = outputs["probs"]
            weight = self.weights[key]
            
            individual_probs[key] = probs
            
            # Weighted sum
            if all_probs_weighted is None:
                all_probs_weighted = probs * weight
            else:
                all_probs_weighted += probs * weight
            total_weight += weight
        
        # Normalize by total weight
        avg_probs = all_probs_weighted / total_weight
        
        # Compute aggregated logits (inverse softmax approximation)
        avg_logits = torch.log(avg_probs + 1e-10)
        
        result = {
            "logits": avg_logits,
            "probs": avg_probs,
            "individual_probs": individual_probs,
            "ensemble_weights": dict(self.weights),
        }
        
        if labels is not None:
            # Use CE loss on the averaged probabilities for evaluation
            loss = F.cross_entropy(avg_logits, labels)
            result["loss"] = loss
        
        return result
    
    def predict(self, text: str, tokenizer_key: str = "deberta") -> Dict:
        """
        Predict mental health condition for a single text using ensemble.
        
        Args:
            text: Input text
            tokenizer_key: Which model's tokenizer to use for encoding
            
        Returns:
            Dict with ensemble predictions
        """
        if not self.models:
            raise RuntimeError("No models loaded. Call load_model() for each model first.")
        
        # Use the specified model's tokenizer
        tokenizer = self.tokenizers.get(tokenizer_key)
        if tokenizer is None:
            tokenizer = list(self.tokenizers.values())[0]
        
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Ensemble forward
        outputs = self.forward(input_ids, attention_mask)
        avg_probs = outputs["probs"].cpu().numpy()[0]
        individual_probs = {
            k: v.cpu().numpy()[0] for k, v in outputs["individual_probs"].items()
        }
        
        prediction = int(torch.argmax(outputs["probs"], dim=-1)[0])
        confidence = float(torch.max(outputs["probs"], dim=-1).values[0])
        
        # Get top 3 from ensemble
        top3_indices = torch.argsort(outputs["probs"][0], descending=True)[:3].tolist()
        top3 = [
            {"label": MENTAL_HEALTH_LABELS[i], "confidence": float(avg_probs[i])}
            for i in top3_indices
        ]
        
        result = {
            "ensemble_prediction": MENTAL_HEALTH_LABELS[prediction],
            "ensemble_confidence": confidence,
            "all_scores_ensemble": {
                label: float(avg_probs[i])
                for i, label in enumerate(MENTAL_HEALTH_LABELS)
            },
            "per_model_predictions": {
                key: {
                    "label": MENTAL_HEALTH_LABELS[int(np.argmax(individual_probs[key]))],
                    "confidence": float(np.max(individual_probs[key])),
                    "scores": {
                        label: float(individual_probs[key][i])
                        for i, label in enumerate(MENTAL_HEALTH_LABELS)
                    }
                }
                for key in individual_probs
            },
            "top_predictions": top3,
            "weights": self.weights,
            "num_models_used": len(self.models),
        }
        
        return result


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
        config: Optional TrainingConfig
        
    Returns:
        Loaded MentalHealthClassifier in eval mode
    """
    if config is None:
        config = TrainingConfig()
    
    # Determine model name from saved model_info if available
    model_info_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            model_info = json.load(f)
        config.model_name = model_info.get("model_name", config.model_name)
        config.num_labels = model_info.get("num_labels", config.num_labels)
    
    # Build model architecture
    model = MentalHealthClassifier(config)
    
    # Load LoRA adapter weights
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {model_path}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        model.base_model = PeftModel.from_pretrained(base_model, model_path)
    else:
        logger.warning(f"No adapter found at {model_path}, using untrained model")
    
    model.eval()
    return model
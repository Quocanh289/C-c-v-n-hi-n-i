# ====================================================
# Training Pipeline for Continuous Learning
# Implements LoRA fine-tuning for incremental model
# updates using collected feedback data
# ====================================================

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from app.models.emotion_model import (
    MultiTaskEmotionModel,
    EmotionModelManager,
    EMOTION_LABELS,
    EMOTION_LABEL_TO_IDX,
)

logger = logging.getLogger(__name__)


# ====================================================
# Feedback Dataset
# ====================================================

class FeedbackDataset(Dataset):
    """
    Dataset for training on human-verified feedback.
    Loads feedback from JSONL files and prepares
    training batches for the multi-task model.
    """
    
    def __init__(
        self,
        feedback_files: List[str],
        tokenizer,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for filepath in feedback_files:
            self._load_feedback_file(filepath)
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            import random
            self.samples = random.sample(self.samples, max_samples)
        
        logger.info(f"Loaded {len(self.samples)} training samples from {len(feedback_files)} files")
    
    def _load_feedback_file(self, filepath: str):
        """Load feedback items from a JSONL file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    
                    text = item.get("text", "")
                    if not text:
                        continue
                    
                    # Get emotion label
                    emotion = item.get("corrected_emotion", item.get("predicted_emotion", "neutral"))
                    emotion_idx = EMOTION_LABEL_TO_IDX.get(emotion, EMOTION_LABEL_TO_IDX["neutral"])
                    
                    # Get toxicity labels
                    toxicity = item.get("corrected_toxicity", item.get("predicted_toxicity", 0.0))
                    toxicity_binary = 1.0 if toxicity and toxicity > 0.5 else 0.0
                    
                    # Get sarcasm labels
                    sarcasm = item.get("corrected_sarcasm", item.get("predicted_sarcasm", 0.0))
                    sarcasm_binary = 1.0 if sarcasm and sarcasm > 0.5 else 0.0
                    
                    self.samples.append({
                        "text": text,
                        "emotion_label": emotion_idx,
                        "toxicity_label": [toxicity_binary, float(toxicity or 0.0)],
                        "sarcasm_label": [sarcasm_binary, float(sarcasm or 0.0)],
                    })
        except Exception as e:
            logger.error(f"Failed to load feedback file {filepath}: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        encoded = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(sample["emotion_label"], dtype=torch.long),
            "toxicity_label": torch.tensor(sample["toxicity_label"], dtype=torch.float),
            "sarcasm_label": torch.tensor(sample["sarcasm_label"], dtype=torch.float),
        }


# ====================================================
# Training Functions
# ====================================================

def train_emotion_model(
    feedback_dir: str,
    model_save_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    validation_split: float = 0.1,
    use_lora: bool = True,
    warmup_steps: int = 100,
    max_samples: Optional[int] = None,
):
    """
    Train/fine-tune the emotion model using collected feedback.
    
    This function:
    1. Loads all feedback data from the feedback directory
    2. Splits into train/validation sets
    3. Fine-tunes using LoRA (efficient adapter training)
    4. Saves the updated model with version metadata
    5. Updates the model manager to use the new weights
    
    Args:
        feedback_dir: Directory containing feedback JSONL files
        model_save_dir: Directory to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        use_lora: Whether to use LoRA fine-tuning
        warmup_steps: LR scheduler warmup steps
        max_samples: Maximum training samples (for testing)
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting model retraining...")
        logger.info(f"Feedback dir: {feedback_dir}")
        logger.info(f"Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")
        
        # Collect feedback files
        feedback_files = []
        if os.path.exists(feedback_dir):
            for filename in sorted(os.listdir(feedback_dir)):
                if filename.endswith(".jsonl"):
                    feedback_files.append(os.path.join(feedback_dir, filename))
        
        if not feedback_files:
            logger.warning("No feedback files found. Skipping training.")
            return
        
        # Load base model with LoRA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = MultiTaskEmotionModel(
            model_name="xlm-roberta-base",
            use_lora=use_lora,
        )
        
        # Use the model's tokenizer
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        
        model.to(device)
        
        # Create dataset and split
        dataset = FeedbackDataset(
            feedback_files=feedback_files,
            tokenizer=tokenizer,
            max_samples=max_samples,
        )
        
        if len(dataset) < 10:
            logger.warning(f"Too few training samples ({len(dataset)}). Skipping.")
            return
        
        # Split into train/validation
        val_size = max(1, int(len(dataset) * validation_split))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Optimizer and scheduler
        if use_lora and hasattr(model, 'lora_params') and model.lora_params:
            optimizer = AdamW(model.lora_params, lr=learning_rate)
            logger.info(f"Training {len(model.lora_params)} LoRA parameters")
        else:
            optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(warmup_steps, total_steps // 10),
            num_training_steps=total_steps,
        )
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                emotion_labels = batch["emotion_label"].to(device)
                toxicity_labels = batch["toxicity_label"].to(device)
                sarcasm_labels = batch["sarcasm_label"].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Compute loss
                losses = model.compute_loss(
                    outputs=outputs,
                    emotion_labels=emotion_labels,
                    toxicity_labels=toxicity_labels,
                    sarcasm_labels=sarcasm_labels,
                )
                
                loss = losses["total_loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    emotion_labels = batch["emotion_label"].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    emotions = outputs["emotion_logits"]
                    
                    # Accuracy
                    _, predicted = torch.max(emotions, 1)
                    total += emotion_labels.size(0)
                    correct += (predicted == emotion_labels).sum().item()
                    
                    # Loss
                    losses = model.compute_loss(
                        outputs=outputs,
                        emotion_labels=emotion_labels,
                    )
                    val_loss += losses["total_loss"].item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total if total > 0 else 0
            
            logger.info(
                f"Epoch {epoch+1} complete | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Accuracy: {accuracy:.4f}"
            )
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Save checkpoint
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                checkpoint_dir = os.path.join(
                    model_save_dir,
                    f"emotion_model_v{timestamp}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                model.save_pretrained(checkpoint_dir)
                
                # Save metrics
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": accuracy,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "use_lora": use_lora,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "created_at": timestamp,
                    "status": "active",
                }
                
                metrics_path = os.path.join(checkpoint_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                
                # Update config
                config_path = os.path.join(checkpoint_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    config["created_at"] = timestamp
                    config["status"] = "active"
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                
                logger.info(f"Best model saved to {checkpoint_dir}")
            
            model.train()
        
        logger.info("=" * 60)
        logger.info("Retraining complete!")
        
        return {
            "status": "complete",
            "best_val_loss": best_val_loss,
            "epochs_trained": epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "checkpoint_dir": checkpoint_dir,
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
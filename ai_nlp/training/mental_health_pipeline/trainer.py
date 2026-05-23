"""
Custom PyTorch Trainer
=======================
High-performance training loop optimized for RTX 4060 16GB.

Features:
- Mixed precision training (fp16) via torch.cuda.amp
- Gradient accumulation for effective batch size control
- Cosine learning rate scheduler with warmup
- Gradient clipping for training stability
- Early stopping with patience
- Model checkpointing (best model only)
- Comprehensive logging and metric tracking
- Memory-efficient evaluation

Optimized for RTX 4060 16GB:
- Batch size 16, grad accum 2 → effective batch 32
- Gradient checkpointing enabled → saves ~30% VRAM
- fp16 mixed precision → ~40% faster throughput
- Clean cache between epochs → prevents fragmentation
- Pin memory + num_workers=4 → max GPU utilization
"""

import os
import gc
import math
import time
import json
import logging
from typing import Dict, Optional, Tuple, Callable, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

import numpy as np
from tqdm import tqdm

from .config import (
    MENTAL_HEALTH_LABELS,
    MENTAL_HEALTH_LABELS_TO_IDX,
    IDX_TO_MH_LABELS,
    TrainingConfig,
)
from .model import MentalHealthClassifier
from .losses import get_loss_fn
from .metrics import MentalHealthMetrics

logger = logging.getLogger(__name__)


class MentalHealthTrainer:
    """
    Custom PyTorch trainer for mental health classification.
    
    Uses a custom training loop (not HuggingFace Trainer) for maximum
    control over:
    - Gradient accumulation steps
    - Mixed precision scheduling
    - Memory management
    - Custom loss functions
    - Per-epoch evaluation
    """
    
    def __init__(
        self,
        model: MentalHealthClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        experiment_name: str = "mental_health",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device(config.device)
        
        # Loss function
        self.criterion = get_loss_fn(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Gradient scaler for fp16
        self.scaler = GradScaler("cuda", enabled=(config.mixed_precision == "fp16"))
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = self._load_best_metric_from_disk()
        self.best_epoch = -1
        self.patience_counter = 0
        self.is_early_stopped = False
        
        # Metrics tracking
        self.train_losses: list = []
        self.val_metrics: list = []
        self.val_losses: list = []
        
        # Create output directories
        self.checkpoint_dir = config.get_checkpoint_dir()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
        
        logger.info(
            f"Optimizer: AdamW, lr={self.config.learning_rate}, "
            f"weight_decay={self.config.weight_decay}"
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create cosine scheduler with linear warmup."""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if warmup_steps > 0:
            # Warmup phase: linear increase from 0 to lr
            # Warmup: start_factor must be > 0 per torch docs. Use 0.01 → 1.0
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            
            # Main phase: cosine annealing
            main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps - warmup_steps,
                T_mult=1,
                eta_min=self.config.learning_rate * 0.01,
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
            
            logger.info(
                f"Scheduler: linear warmup ({warmup_steps} steps) + "
                f"cosine annealing ({total_steps - warmup_steps} steps)"
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps,
                T_mult=1,
                eta_min=self.config.learning_rate * 0.01,
            )
            logger.info(f"Scheduler: cosine annealing ({total_steps} steps)")
        
        return scheduler
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs} [Train]",
            leave=False,
            ncols=100,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=(self.config.mixed_precision == "fp16")):
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs["logits"], labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation: only step after accum_steps batches
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                logger.debug(
                    f"Step {self.global_step}: "
                    f"loss={loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        
        # Clear GPU cache after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        
        Args:
            loader: DataLoader to evaluate on (defaults to val_loader)
            
        Returns:
            Dict with metrics
        """
        loader = loader or self.val_loader
        self.model.eval()
        
        metrics = MentalHealthMetrics()
        total_loss = 0.0
        
        for batch in tqdm(loader, desc="Evaluating", leave=False, ncols=100):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs["logits"], labels)
            
            # Get predictions
            predictions = torch.argmax(outputs["probs"], dim=-1)
            
            # Update metrics
            metrics.update(predictions, labels, outputs["probs"])
            total_loss += loss.item()
        
        # Compute all metrics
        results = metrics.compute()
        results["avg_loss"] = total_loss / len(loader)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop with early stopping and checkpointing.
        
        Returns:
            Dict with final metrics, best epoch, and history
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Early stopping patience: {self.config.early_stopping_patience}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Total trainable params: "
                    f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_results = self.evaluate()
            self.val_metrics.append(val_results)
            self.val_losses.append(val_results["avg_loss"])
            
            # Log epoch results
            macro_f1 = val_results.get("macro_f1", 0.0)
            weighted_f1 = val_results.get("weighted_f1", 0.0)
            accuracy = val_results.get("accuracy", 0.0)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_results['avg_loss']:.4f} | "
                f"Acc: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f} | "
                f"Weighted F1: {weighted_f1:.4f}"
            )
            
            # Check early stopping metric
            current_metric = val_results.get(
                self.config.early_stopping_metric, macro_f1
            )
            
            # Save best model
            if current_metric > self.best_metric + self.config.early_stopping_threshold:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(val_results, is_best=True)
                logger.info(f"[BEST] New best model! Macro F1: {macro_f1:.4f} (epoch {epoch + 1})")
            else:
                self.patience_counter += 1
                logger.info(
                    f"[NO IMPROVEMENT] Patience: "
                    f"{self.patience_counter}/{self.config.early_stopping_patience}"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best epoch: {self.best_epoch + 1} (Macro F1: {self.best_metric:.4f})"
                )
                self.is_early_stopped = True
                break
            
            # Periodic save (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(val_results, is_best=False, suffix=f"epoch_{epoch+1}")
        
        total_time = time.time() - start_time
        
        # Load best model
        self._load_best_checkpoint()
        
        # Final evaluation
        final_results = self.evaluate()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"Best epoch: {self.best_epoch + 1}")
        logger.info(f"Best macro F1: {self.best_metric:.4f}")
        logger.info(f"Final macro F1: {final_results.get('macro_f1', 0.0):.4f}")
        logger.info("=" * 60)
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "final_metrics": final_results,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "total_time_seconds": total_time,
            "early_stopped": self.is_early_stopped,
            "total_epochs_completed": self.current_epoch + 1,
        }
    
    def _save_checkpoint(self, val_results: Dict, is_best: bool = False, suffix: str = ""):
        """
        Save model checkpoint.
        
        Args:
            val_results: Validation metrics
            is_best: Whether this is the best model
            suffix: Optional suffix for filename
        """
        checkpoint_name = "best_model" if is_best else f"checkpoint_{suffix}" if suffix else f"checkpoint_epoch_{self.current_epoch}"
        save_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save optimizer and scheduler state (for resume)
        if not is_best:
            training_state = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "config": self.config,
            }
            torch.save(training_state, os.path.join(save_path, "training_state.pt"))
        
        # Save metrics
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {save_path}")
        
        # Clean up old checkpoints (keep only save_total_limit)
        self._cleanup_checkpoints()
    
    def _load_best_metric_from_disk(self) -> float:
        """
        Load the best metric value from an existing best_model checkpoint on disk.
        
        This prevents new training runs from overwriting a previously saved
        best model unless the new model actually performs better.
        
        Returns:
            The best metric value from disk, or -inf if no previous best model exists.
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model")
        metrics_path = os.path.join(best_path, "metrics.json")
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                metric_key = self.config.early_stopping_metric
                previous_best = metrics.get(metric_key, -float("inf"))
                
                if previous_best > -float("inf"):
                    logger.info(
                        f"Loaded previous best model from {best_path}: "
                        f"{metric_key}={previous_best:.6f}"
                    )
                    return previous_best
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Could not load previous best metric: {e}")
        
        logger.info("No previous best model found. Starting fresh.")
        return -float("inf")

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model")
        if os.path.exists(os.path.join(best_path, "adapter_model.safetensors")):
            logger.info(f"Loading best model from {best_path}")
            
            # For LoRA models, we need to reload the base + adapter
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification
            
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
            )
            self.model.base_model = PeftModel.from_pretrained(base_model, best_path)
            self.model.to(self.device)
            self.model.eval()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only save_total_limit."""
        limit = self.config.save_total_limit
        if limit <= 0:
            return
        
        checkpoint_dir = self.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            return
        
        # List all checkpoint directories
        checkpoints = []
        for entry in os.listdir(checkpoint_dir):
            entry_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(entry_path) and entry != "best_model":
                checkpoints.append(entry_path)
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove oldest checkpoints beyond limit
        while len(checkpoints) > limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate on test set using the best model.
        
        Args:
            test_loader: Test DataLoader
            
        Returns:
            Dict with test metrics
        """
        logger.info("Evaluating on test set...")
        
        # Load best model if available
        best_path = os.path.join(self.checkpoint_dir, "best_model")
        if os.path.exists(os.path.join(best_path, "adapter_model.safetensors")):
            self._load_best_checkpoint()
        
        results = self.evaluate(test_loader)
        
        logger.info("=" * 60)
        logger.info("TEST SET RESULTS")
        logger.info(f"Accuracy: {results.get('accuracy', 0.0):.4f}")
        logger.info(f"Macro F1: {results.get('macro_f1', 0.0):.4f}")
        logger.info(f"Weighted F1: {results.get('weighted_f1', 0.0):.4f}")
        logger.info(f"Macro Precision: {results.get('macro_precision', 0.0):.4f}")
        logger.info(f"Macro Recall: {results.get('macro_recall', 0.0):.4f}")
        logger.info(f"MCC: {results.get('mcc', 0.0):.4f}")
        logger.info("=" * 60)
        
        # Log per-class metrics
        logger.info("Per-class metrics:")
        per_class = results.get("per_class", {})
        for label in MENTAL_HEALTH_LABELS:
            if label in per_class:
                m = per_class[label]
                logger.info(
                    f"  {label:25s}: "
                    f"P={m['precision']:.4f} "
                    f"R={m['recall']:.4f} "
                    f"F1={m['f1']:.4f} "
                    f"(n={m['support']})"
                )
        
        # Log hardest classes
        logger.info("Hardest classes (lowest F1):")
        for hc in results.get("hardest_classes", []):
            logger.info(f"  {hc['label']:25s}: F1={hc['f1']:.4f}")
        
        # Log misclassifications
        logger.info("Top misclassifications:")
        for mc in results.get("misclassifications", [])[:5]:
            logger.info(
                f"  True={mc['true_label']:25s} → "
                f"Pred={mc['predicted_label']:25s}: "
                f"{mc['count']} ({mc['percentage']:.1f}%)"
            )
        
        return results
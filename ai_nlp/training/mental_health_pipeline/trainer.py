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

Best model saving strategy:
1. During training: epochs compete ONLY within the current session
   (track session_best, save to session-specific temp dir)
2. After training ends: compare session_best vs disk's best_model
3. Only overwrite disk's best_model if session_best is truly better
   This prevents a worse training run from destroying a previously good model
"""

import os
import gc
import math
import time
import json
import logging
from typing import Dict, Optional, Tuple, Callable, Any
from pathlib import Path
from datetime import datetime

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
        
        # Create output directories
        self.checkpoint_dir = config.get_checkpoint_dir()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state — local to this session only (no disk loading)
        self.current_epoch = 0
        self.global_step = 0
        self.session_best_metric = -float("inf")
        self.session_best_epoch = -1
        self.session_best_results = None
        self.patience_counter = 0
        self.is_early_stopped = False
        
        # Session identifier for temp storage
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_temp_dir = os.path.join(
            self.checkpoint_dir, f"session_best_{self.session_id}"
        )
        
        # Metrics tracking
        self.train_losses: list = []
        self.val_metrics: list = []
        self.val_losses: list = []
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info("Epoch comparisons are local to this training session only")
    
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
    def evaluate(self, loader: Optional[DataLoader] = None, log_confusion: bool = False) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        
        Args:
            loader: DataLoader to evaluate on (defaults to val_loader)
            log_confusion: Whether to log detailed confusion stats
            
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
        
        # Log confusion analysis if requested
        if log_confusion and "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            # Log the worst confusion pairs
            n_classes = len(cm)
            confusion_pairs = []
            for i in range(n_classes):
                total_i = sum(cm[i])
                if total_i == 0:
                    continue
                for j in range(n_classes):
                    if i != j and cm[i][j] > 0:
                        pct = cm[i][j] / total_i * 100
                        if pct >= 5.0:  # Only log pairs with >= 5% confusion
                            true_label = IDX_TO_MH_LABELS[i]
                            pred_label = IDX_TO_MH_LABELS[j]
                            confusion_pairs.append((true_label, pred_label, cm[i][j], pct))
            
            # Sort by count descending
            confusion_pairs.sort(key=lambda x: x[2], reverse=True)
            
            if confusion_pairs:
                logger.info("  Top confusion pairs (true → pred):")
                for true_l, pred_l, count, pct in confusion_pairs[:6]:
                    logger.info(f"    {true_l:25s} → {pred_l:25s}: {count} ({pct:.1f}%)")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        
        Best model saving strategy:
        - During training: epochs compete ONLY within this session
          (best is tracked via self.session_best_metric)
        - The session's best model is saved to a temp dir
        - After training: compare session_best vs disk's best_model
        - Only overwrite disk's best_model if session_best is better
        
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
        logger.info(f"Session ID: {self.session_id}")
        logger.info("Epoch comparisons: within this session only")
        logger.info("Post-training: session best vs disk best_model")
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
            
            # Check early stopping metric (local to this session)
            current_metric = val_results.get(
                self.config.early_stopping_metric, macro_f1
            )
            
            # Track session's best model (compares ONLY against other epochs in this session)
            if current_metric > self.session_best_metric + self.config.early_stopping_threshold:
                self.session_best_metric = current_metric
                self.session_best_epoch = epoch
                self.session_best_results = val_results
                self.patience_counter = 0
                # Save to session temp dir
                self._save_session_checkpoint(val_results)
                logger.info(
                    f"[SESSION BEST] New session best! "
                    f"Macro F1: {macro_f1:.4f} (epoch {epoch + 1})"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"[NO IMPROVEMENT] Session best: {self.session_best_metric:.4f} "
                    f"(epoch {self.session_best_epoch + 1}) | "
                    f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Session best: epoch {self.session_best_epoch + 1} "
                    f"(Macro F1: {self.session_best_metric:.4f})"
                )
                self.is_early_stopped = True
                break
            
            # Periodic save (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(val_results, is_best=False, suffix=f"epoch_{epoch+1}")
        
        total_time = time.time() - start_time
        
        # ----- POST-TRAINING: Compare session best vs disk best_model -----
        logger.info("=" * 60)
        logger.info("POST-TRAINING: Comparing session best vs disk best_model")
        logger.info("=" * 60)
        
        disk_best_metric = self._load_best_metric_from_disk()
        session_best_metric = self.session_best_metric
        
        logger.info(f"  Session best {self.config.early_stopping_metric}: {session_best_metric:.6f}")
        logger.info(f"  Disk best {self.config.early_stopping_metric}: {disk_best_metric:.6f}")
        
        if session_best_metric > disk_best_metric + self.config.early_stopping_threshold:
            logger.info(
                f"  ✓ Session model is BETTER! "
                f"Overwriting disk's best_model (gain: {session_best_metric - disk_best_metric:.6f})"
            )
            self._promote_session_to_best()
        else:
            logger.info(
                f"  ✗ Session model did NOT outperform disk's best model. "
                f"Disk's best_model preserved."
            )
            # Clean up session temp dir
            self._cleanup_session_checkpoint()
        
        # Load the (possibly updated) best model
        self._load_best_checkpoint()
        
        # Final evaluation
        final_results = self.evaluate()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"Session best epoch: {self.session_best_epoch + 1}")
        logger.info(f"Session best macro F1: {self.session_best_metric:.4f}")
        logger.info(f"Disk best macro F1: {final_results.get('macro_f1', 0.0):.4f}")
        logger.info("=" * 60)
        
        return {
            "best_epoch": self.session_best_epoch,
            "best_metric": session_best_metric,
            "disk_best_metric": disk_best_metric,
            "was_best_updated": session_best_metric > disk_best_metric + self.config.early_stopping_threshold,
            "final_metrics": final_results,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "total_time_seconds": total_time,
            "early_stopped": self.is_early_stopped,
            "total_epochs_completed": self.current_epoch + 1,
            "session_id": self.session_id,
        }
    
    def _save_session_checkpoint(self, val_results: Dict):
        """
        Save the session's best model to a temporary directory.
        
        This is only promoted to the official 'best_model' folder
        if it beats the existing disk best after training completes.
        """
        save_path = self.session_temp_dir
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save session info
        session_info = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "session_id": self.session_id,
            "session_best_metric": self.session_best_metric,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_path, "session_info.json"), "w") as f:
            json.dump(session_info, f, indent=2)
        
        # Save metrics
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        
        logger.debug(f"Session best saved to {save_path}")
    
    def _promote_session_to_best(self):
        """
        Promote the session's best model to the official best_model folder.
        
        This is called AFTER training, only if the session best beats disk best.
        """
        src = self.session_temp_dir
        dst = os.path.join(self.checkpoint_dir, "best_model")
        
        # Remove old best_model
        if os.path.exists(dst):
            import shutil
            shutil.rmtree(dst)
            logger.info(f"Removed old best_model from {dst}")
        
        # Copy session best to best_model
        import shutil
        shutil.copytree(src, dst)
        logger.info(f"Promoted session best to {dst}")
        
        # Mark with promotion info
        promotion_info = {
            "promoted_from_session": self.session_id,
            "promoted_at": datetime.now().isoformat(),
            "session_best_metric": self.session_best_metric,
            "session_best_epoch": self.session_best_epoch + 1,
        }
        with open(os.path.join(dst, "promotion_info.json"), "w") as f:
            json.dump(promotion_info, f, indent=2)
    
    def _cleanup_session_checkpoint(self):
        """Remove the session temp dir if it was not promoted."""
        if os.path.exists(self.session_temp_dir):
            import shutil
            shutil.rmtree(self.session_temp_dir)
            logger.info(f"Cleaned up session temp dir: {self.session_temp_dir}")
    
    def _save_checkpoint(self, val_results: Dict, is_best: bool = False, suffix: str = ""):
        """
        Save periodic checkpoint (not best model).
        
        Args:
            val_results: Validation metrics
            is_best: Whether this is the best model (always False here)
            suffix: Optional suffix for filename
        """
        checkpoint_name = f"checkpoint_{suffix}" if suffix else f"checkpoint_epoch_{self.current_epoch}"
        save_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save optimizer and scheduler state (for resume)
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "session_best_metric": self.session_best_metric,
            "session_best_epoch": self.session_best_epoch,
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
        Load the best metric value from the disk's best_model checkpoint.
        
        Used ONLY post-training to decide if session best should overwrite.
        Returns:
            The best metric value from disk, or -inf if no best model exists.
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
                        f"Disk best model: {metric_key}={previous_best:.6f}"
                    )
                    return previous_best
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Could not load disk best metric: {e}")
        
        logger.info("No disk best model found.")
        return -float("inf")

    def _load_best_checkpoint(self):
        """Load the best model checkpoint from disk's best_model folder."""
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
        
        # List all checkpoint directories (excluding best_model and session_best)
        checkpoints = []
        for entry in os.listdir(checkpoint_dir):
            entry_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(entry_path) and entry not in ("best_model",):
                # Skip session temp dir
                if "session_best" in entry:
                    continue
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
        
        # Load best model from disk's best_model folder
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
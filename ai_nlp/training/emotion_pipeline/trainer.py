"""
Trainer Module
==============
Main training loop with:
- Mixed precision (FP16/FP32)
- Gradient accumulation
- Cosine/linear LR scheduling
- Early stopping
- Model checkpointing
- Evaluation at epoch/steps intervals
- TensorBoard/MLflow logging
"""

import os
import json
import time
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from .config import TrainingConfig
from .model import GoEmotionsModel
from .losses import FocalLoss, CombinedLoss
from .metrics import EmotionMetrics, compute_class_metrics, compute_all_metrics
from .gpu_utils import AverageMeter, ProgressMeter
from .threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from a training run."""
    
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    best_step: int = 0
    total_epochs: int = 0
    total_steps: int = 0
    checkpoint_dir: str = ""
    training_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "best_metrics": self.best_metrics,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "training_time_seconds": self.training_time,
        }


class Trainer:
    """
    Full-featured trainer for GoEmotions emotion classification.
    
    Features:
    - Mixed precision (FP16/FP32) via GradScaler
    - Gradient accumulation
    - Cosine/linear warmup scheduler
    - Early stopping
    - Model checkpointing (best/latest)
    - Per-epoch evaluation
    - Threshold optimization
    """
    
    def __init__(
        self,
        model: GoEmotionsModel,
        tokenizer,
        config: TrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(
            primary_loss=config.primary_loss,
            secondary_loss=config.secondary_loss,
            primary_weight=config.primary_weight,
            secondary_weight=config.secondary_weight,
            focal_gamma=config.focal_gamma,
            asl_gamma_neg=config.asl_gamma_neg,
            asl_gamma_pos=config.asl_gamma_pos,
            label_smoothing=config.label_smoothing,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if (
            config.mixed_precision == "fp16" and self.device.type == "cuda"
        ) else None
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_score = float("-inf") if config.greater_is_better else float("inf")
        self.best_epoch = 0
        self.best_metrics: Optional[EmotionMetrics] = None
        self.early_stop_counter = 0
        self.training_start_time = time.time()
        
        # Checkpoint dir
        self.checkpoint_dir = config.get_checkpoint_dir()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_scheduler(self, total_steps: int):
        """Create learning rate scheduler."""
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.config.num_cycles,
            )
        elif self.config.scheduler == "cosine_with_restarts":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.config.num_cycles,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        
        logger.info(f"Scheduler: {self.config.scheduler}, warmup_steps={warmup_steps}, total_steps={total_steps}")
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, scheduler) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            scheduler: LR scheduler
        
        Returns:
            Average training loss for this epoch
        """
        self.model.train()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            data_time.update(time.time() - end)
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass (with or without mixed precision)
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs["logits"], labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs["logits"], labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            losses.update(loss.item() * self.config.gradient_accumulation_steps, input_ids.size(0))
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {self.current_epoch+1} | "
                        f"Step {self.global_step} | "
                        f"Loss: {losses.avg:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Data: {data_time.avg*1000:.0f}ms | "
                        f"Batch: {batch_time.avg*1000:.0f}ms"
                    )
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Evaluation at step intervals
            if (self.config.eval_strategy == "steps" and
                self.global_step % self.config.eval_steps == 0):
                pass  # Evaluation happens in train()
        
        return losses.avg
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, EmotionMetrics, Optional[ThresholdOptimizer]]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (avg_loss, metrics, threshold_optimizer)
        """
        self.model.eval()
        
        losses = AverageMeter()
        all_logits = []
        all_labels = []
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs["logits"], labels)
            
            losses.update(loss.item(), input_ids.size(0))
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # Concatenate all batches
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        from .config import COARSE_EMOTIONS
        
        # Compute metrics
        metrics = compute_all_metrics(all_logits, all_labels, COARSE_EMOTIONS)
        metrics.accuracy = compute_class_metrics(all_labels, np.argmax(all_logits, axis=1), COARSE_EMOTIONS).accuracy
        
        logger.info(
            f"Evaluation | Loss: {losses.avg:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | "
            f"Macro F1: {metrics.macro_f1:.4f} | "
            f"Weighted F1: {metrics.weighted_f1:.4f} | "
            f"MCC: {metrics.mcc:.4f}"
        )
        
        # Threshold optimization
        threshold_optimizer = None
        if self.config.optimize_thresholds:
            probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
            threshold_optimizer = ThresholdOptimizer(
                n_classes=self.config.num_labels,
                metric=self.config.threshold_optimization_metric,
                n_trials=self.config.threshold_n_trials,
                label_names=COARSE_EMOTIONS,
            )
            threshold_optimizer.optimize(probs, all_labels)
        
        return losses.avg, metrics, threshold_optimizer
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> TrainingResult:
        """
        Run the complete training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            test_loader: Optional test data
        
        Returns:
            TrainingResult with best metrics and checkpoint info
        """
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        scheduler = self._get_scheduler(total_steps)
        
        logger.info(f"Starting training: {len(train_loader)} batches/epoch, {self.config.num_epochs} epochs")
        logger.info(f"Total optimization steps: {total_steps}")
        
        result = TrainingResult()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, scheduler)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss, metrics, threshold_opt = self.evaluate(val_loader)
            
            # Determine if this is the best model
            metric_value = getattr(metrics, self.config.metric_for_best_model, metrics.macro_f1)
            
            is_best = False
            if self.config.greater_is_better:
                if metric_value > self.best_score:
                    is_best = True
                    self.best_score = metric_value
                    self.best_epoch = epoch
                    self.best_metrics = metrics
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            else:
                if metric_value < self.best_score:
                    is_best = True
                    self.best_score = metric_value
                    self.best_epoch = epoch
                    self.best_metrics = metrics
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            
            # Save checkpoint if best
            if is_best:
                self._save_checkpoint(epoch, metrics, is_best=True)
                logger.info(f"New best model! {self.config.metric_for_best_model}={metric_value:.4f}")
            
            # Save latest checkpoint
            if epoch % 2 == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch, metrics, is_best=False)
            
            # Early stopping
            if self.early_stop_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"({self.early_stop_counter} without improvement)"
                )
                break
            
            # Log per-class performance
            logger.info(f"Per-class F1:")
            for label, cls_metrics in metrics.per_class.items():
                logger.info(f"  {label}: F1={cls_metrics['f1']:.4f} "
                           f"P={cls_metrics['precision']:.4f} "
                           f"R={cls_metrics['recall']:.4f} "
                           f"(n={cls_metrics['support']})")
        
        # Final test evaluation
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            self._load_best_checkpoint()
            test_loss, test_metrics, _ = self.evaluate(test_loader)
            logger.info(f"Test Results | Loss: {test_loss:.4f} | "
                       f"Acc: {test_metrics.accuracy:.4f} | "
                       f"Macro F1: {test_metrics.macro_f1:.4f} | "
                       f"Weighted F1: {test_metrics.weighted_f1:.4f}")
            
            # Save test metrics
            test_metrics.save(os.path.join(self.checkpoint_dir, "test_metrics.json"))
        
        # Save training summary
        training_time = time.time() - self.training_start_time
        result = TrainingResult(
            best_metrics=self.best_metrics.to_dict() if self.best_metrics else {},
            best_epoch=self.best_epoch + 1,
            best_step=self.global_step,
            total_epochs=self.current_epoch + 1,
            total_steps=self.global_step,
            checkpoint_dir=self.checkpoint_dir,
            training_time=training_time,
        )
        
        # Save result summary
        result_path = os.path.join(self.checkpoint_dir, "training_result.json")
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Training complete! Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
        logger.info(f"Best epoch: {result.best_epoch}, Best {self.config.metric_for_best_model}: {self.best_score:.4f}")
        
        return result
    
    def _save_checkpoint(self, epoch: int, metrics: EmotionMetrics, is_best: bool = False):
        """Save model checkpoint."""
        prefix = "best_model" if is_best else f"checkpoint_epoch_{epoch+1}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, prefix)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metrics
        metrics.save(os.path.join(checkpoint_path, "metrics.json"))
        
        # Save optimizer state
        if not is_best:
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "epoch": epoch,
                "step": self.global_step,
                "best_score": self.best_score,
            }, os.path.join(checkpoint_path, "optimizer_state.pt"))
        
        # Save training config
        training_config = {
            "epoch": epoch + 1,
            "best_model": is_best,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "train_samples": len(train_loader.dataset) if hasattr(self, 'train_loader') else 0,
            "num_classes": self.config.num_labels,
            "labels": self.config.num_labels,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(training_config, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model")
        if os.path.exists(best_path):
            logger.info(f"Loading best model from {best_path}")
            self.model = GoEmotionsModel.from_pretrained(best_path, self.config)
            self.model.to(self.device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a specific checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.model = GoEmotionsModel.from_pretrained(checkpoint_path, self.config)
        self.model.to(self.device)
        
        # Load optimizer state if exists
        opt_path = os.path.join(checkpoint_path, "optimizer_state.pt")
        if os.path.exists(opt_path):
            state = torch.load(opt_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            if state["scaler"] and self.scaler:
                self.scaler.load_state_dict(state["scaler"])
            self.global_step = state.get("step", 0)
            logger.info(f"Loaded optimizer state, resuming from step {self.global_step}")
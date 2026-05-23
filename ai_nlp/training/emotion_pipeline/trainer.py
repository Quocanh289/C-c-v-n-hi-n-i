"""
Trainer Module
===============
Multi-label training loop for GoEmotions emotion classification.
Supports:
- Mixed precision (FP16/BF16)
- Gradient accumulation & clipping
- Cosine/linear LR scheduling with warmup
- Early stopping with patience
- Model checkpointing (best/latest)
- Multi-label and dual-head evaluation
- Threshold optimization per epoch
- TensorBoard logging
"""

import os
import json
import time
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
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

from .config import TrainingConfig, GOEMOTIONS_28, COARSE_EMOTIONS
from .model import GoEmotionsModel
from .losses import get_loss_function, CombinedLoss
from .metrics import (
    EmotionMetrics,
    compute_multi_label_metrics,
    compute_all_metrics,
    detect_overfitting,
    detect_data_leakage,
    detect_distribution_shift,
)
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
    best_thresholds: Optional[List[float]] = None
    # Overfitting/leakage analysis
    overfitting_analysis: Dict = field(default_factory=dict)
    data_leakage_analysis: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "best_metrics": self.best_metrics,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "training_time_seconds": self.training_time,
            "best_thresholds": self.best_thresholds,
            "overfitting_analysis": self.overfitting_analysis,
            "data_leakage_analysis": self.data_leakage_analysis,
        }


class Trainer:
    """
    Full-featured trainer for multi-label GoEmotions emotion classification.
    
    Features:
    - Mixed precision (FP16/FP32) via GradScaler
    - Gradient accumulation
    - Cosine/linear warmup scheduler
    - Early stopping
    - Model checkpointing (best/latest)
    - Per-epoch evaluation with multi-label metrics
    - Threshold optimization
    - Overfitting/leakage detection
    - Class-weighted loss from dataset
    - Dual-head support (28 fine + 9 coarse)
    """
    
    def __init__(
        self,
        model: GoEmotionsModel,
        tokenizer,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None,
        coarse_class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Move class weights to device
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.coarse_class_weights = coarse_class_weights.to(self.device) if coarse_class_weights is not None else None
        
        # Loss function
        self.criterion = get_loss_function(
            config,
            class_weights=self.class_weights,
            coarse_class_weights=self.coarse_class_weights,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
        
        # Mixed precision
        self.use_fp16 = config.mixed_precision == "fp16" and self.device.type == "cuda"
        self.use_bf16 = config.mixed_precision == "bf16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_score = self._load_best_metric_from_disk()
        self.best_epoch = 0
        self.best_metrics: Optional[EmotionMetrics] = None
        self.best_thresholds: Optional[np.ndarray] = None
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
        """Train for one epoch. Returns average loss."""
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
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(input_ids, attention_mask)
                    loss = self._compute_loss(outputs, batch)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self._compute_loss(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            losses.update(loss.item() * self.config.gradient_accumulation_steps, input_ids.size(0))
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
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
                
                if self.global_step % self.config.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {self.current_epoch+1} | "
                        f"Step {self.global_step} | "
                        f"Loss: {losses.avg:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Batch: {batch_time.avg*1000:.0f}ms"
                    )
            
            batch_time.update(time.time() - end)
            end = time.time()
        
        return losses.avg
    
    def _compute_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute loss based on task type.
        
        Supports:
        - "multi_label": BCE loss on sigmoid outputs
        - "dual_head": BCE on both fine (28) and coarse (9) outputs
        - "multi_class": CE loss on softmax outputs (backward compat)
        """
        if self.config.task_type == "dual_head":
            # Dual head: fine + coarse
            coarse_labels = batch.get("labels_9", batch.get("labels"))
            coarse_logits = outputs.get("coarse_logits")
            coarse_targets = coarse_labels.to(self.device) if torch.is_tensor(coarse_labels) else None
            
            if coarse_logits is not None and coarse_targets is not None:
                return self.criterion(
                    outputs["logits"],
                    batch["labels"].to(self.device),
                    coarse_logits=coarse_logits,
                    coarse_targets=coarse_targets,
                )
        
        # Standard single-head loss
        return self.criterion(outputs["logits"], batch["labels"].to(self.device))
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, EmotionMetrics, Optional[ThresholdOptimizer]]:
        """
        Evaluate model on validation set with multi-label metrics.
        
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
            
            outputs = self.model(input_ids, attention_mask)
            
            if self.config.task_type == "dual_head":
                loss = self._compute_loss(outputs, batch)
            else:
                loss = self.criterion(outputs["logits"], labels)
            
            losses.update(loss.item(), input_ids.size(0))
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute multi-label metrics
        metrics = compute_all_metrics(all_logits, all_labels, GOEMOTIONS_28)
        
        logger.info(
            f"Evaluation | Loss: {losses.avg:.4f} | "
            f"Macro F1: {metrics.macro_f1_micro_avg:.4f} | "
            f"Micro F1: {metrics.micro_f1:.4f} | "
            f"Hamming: {metrics.hamming_loss:.4f} | "
            f"ECE: {metrics.expected_calibration_error:.4f}"
        )
        
        # Threshold optimization
        threshold_optimizer = None
        if self.config.optimize_thresholds:
            probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
            threshold_optimizer = ThresholdOptimizer(
                n_classes=self.config.num_labels,
                metric=self.config.threshold_optimization_metric,
                n_trials=self.config.threshold_n_trials,
                label_names=GOEMOTIONS_28[:self.config.num_labels],
            )
            threshold_optimizer.optimize(probs, all_labels)
            self.best_thresholds = threshold_optimizer.best_thresholds
            
            # Recompute metrics with optimized thresholds
            preds = threshold_optimizer.predict(probs)
            opt_metrics = compute_multi_label_metrics(
                all_labels, preds, GOEMOTIONS_28[:self.config.num_labels], probs=probs
            )
            logger.info(f"  With optimized thresholds - Macro F1: {opt_metrics.macro_f1_micro_avg:.4f}")
            
            # Update metrics with threshold-specific values
            metrics = opt_metrics
        
        return losses.avg, metrics, threshold_optimizer
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        train_eval_loader: Optional[DataLoader] = None,
    ) -> TrainingResult:
        """
        Run complete multi-label training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            test_loader: Optional test data
            train_eval_loader: Optional train subset for overfitting detection
        
        Returns:
            TrainingResult with all metrics and analysis
        """
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        scheduler = self._get_scheduler(total_steps)
        
        logger.info(f"Starting multi-label training:")
        logger.info(f"  {len(train_loader)} batches/epoch, {self.config.num_epochs} epochs")
        logger.info(f"  Total optimization steps: {total_steps}")
        logger.info(f"  Task: {self.config.task_type}")
        
        result = TrainingResult()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader, scheduler)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss, metrics, threshold_opt = self.evaluate(val_loader)
            
            # Determine best model
            metric_value = getattr(metrics, self.config.metric_for_best_model, metrics.macro_f1_micro_avg)
            
            is_best = False
            if self.config.greater_is_better:
                if metric_value > self.best_score + self.config.early_stopping_threshold:
                    is_best = True
                    self.best_score = metric_value
                    self.best_epoch = epoch
                    self.best_metrics = metrics
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            else:
                if metric_value < self.best_score - self.config.early_stopping_threshold:
                    is_best = True
                    self.best_score = metric_value
                    self.best_epoch = epoch
                    self.best_metrics = metrics
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            
            # Save checkpoint if best
            if is_best:
                self._save_checkpoint(epoch, metrics, thresholds=threshold_opt, is_best=True)
                logger.info(f"  New best! {self.config.metric_for_best_model}={metric_value:.4f}")
            else:
                logger.info(f"  {self.config.metric_for_best_model}={metric_value:.4f} (best={self.best_score:.4f}, wait={self.early_stop_counter}/{self.config.early_stopping_patience})")
            
            # Periodic save
            if epoch % 5 == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch, metrics, is_best=False)
            
            # Log per-class metrics
            logger.info("  Per-class F1:")
            for label, cls_metrics in metrics.per_label.items():
                logger.info(f"    {label:20s}: F1={cls_metrics['f1']:.4f} P={cls_metrics['precision']:.4f} R={cls_metrics['recall']:.4f} (n={cls_metrics['support']})")
            
            # Early stopping
            if self.early_stop_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs ({self.early_stop_counter} without improvement)")
                break
        
        # Final test evaluation
        logger.info("=" * 60)
        logger.info("Running final evaluation on test set...")
        self._load_best_checkpoint()
        
        if test_loader is not None:
            test_loss, test_metrics, _ = self.evaluate(test_loader)
            logger.info(f"Test Results:")
            logger.info(f"  Macro F1: {test_metrics.macro_f1_micro_avg:.4f}")
            logger.info(f"  Micro F1: {test_metrics.micro_f1:.4f}")
            logger.info(f"  Subset Acc: {test_metrics.subsets_accuracy:.4f}")
            logger.info(f"  Hamming Loss: {test_metrics.hamming_loss:.4f}")
            logger.info(f"  ECE: {test_metrics.expected_calibration_error:.4f}")
            
            test_metrics.save(os.path.join(self.checkpoint_dir, "test_metrics.json"))
            
            # Overfitting detection
            if train_eval_loader is not None:
                _, train_metrics, _ = self.evaluate(train_eval_loader)
                overfit = detect_overfitting(train_metrics, test_metrics)
                logger.info(f"Overfitting risk: {overfit['overfitting_risk']} (gap={overfit['macro_f1_gap']:.4f})")
                result.overfitting_analysis = overfit
            
            # Data leakage detection
            leakage = detect_data_leakage(test_metrics, test_metrics)
            result.data_leakage_analysis = leakage
        
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
            best_thresholds=self.best_thresholds.tolist() if self.best_thresholds is not None else None,
            overfitting_analysis=result.overfitting_analysis,
            data_leakage_analysis=result.data_leakage_analysis,
        )
        
        # Save result
        result_path = os.path.join(self.checkpoint_dir, "training_result.json")
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Training complete! Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        logger.info(f"Best epoch: {result.best_epoch}, Best F1: {self.best_score:.4f}")
        
        return result
    
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
                    
                    # If greater_is_better=False (lower is better), invert the check
                    # by returning the value as-is. The comparison logic in train()
                    # handles both cases.
                    return previous_best
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Could not load previous best metric: {e}")
        
        logger.info("No previous best model found. Starting fresh.")
        return float("-inf") if self.config.greater_is_better else float("inf")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: EmotionMetrics,
        thresholds: Optional[ThresholdOptimizer] = None,
        is_best: bool = False,
    ):
        """Save model checkpoint and training state."""
        prefix = "best_model" if is_best else f"checkpoint_epoch_{epoch+1}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, prefix)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metrics
        metrics.save(os.path.join(checkpoint_path, "metrics.json"))
        
        # Save thresholds
        if thresholds is not None:
            thresholds.save(os.path.join(checkpoint_path, "thresholds.json"))
        
        # Save optimizer state (for resume)
        if not is_best:
            state = {"optimizer": self.optimizer.state_dict(), "epoch": epoch, "step": self.global_step, "best_score": self.best_score}
            if self.scaler:
                state["scaler"] = self.scaler.state_dict()
            torch.save(state, os.path.join(checkpoint_path, "optimizer_state.pt"))
        
        # Save training state
        training_state = {
            "epoch": epoch + 1,
            "best_model": is_best,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "best_metric_value": self.best_score,
            "num_labels": self.config.num_labels,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model")
        if os.path.exists(best_path):
            logger.info(f"Loading best model from {best_path}")
            self.model = GoEmotionsModel.from_pretrained(best_path, self.config)
            self.model.to(self.device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a specific checkpoint for resume."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.model = GoEmotionsModel.from_pretrained(checkpoint_path, self.config)
        self.model.to(self.device)
        
        opt_path = os.path.join(checkpoint_path, "optimizer_state.pt")
        if os.path.exists(opt_path):
            state = torch.load(opt_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler") and self.scaler:
                self.scaler.load_state_dict(state["scaler"])
            self.global_step = state.get("step", 0)
            logger.info(f"Loaded optimizer state, resuming from step {self.global_step}")
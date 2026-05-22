"""
Loss Functions
===============
Advanced loss functions for mental health classification with class imbalance.

Losses provided:
1. FocalLoss — Focuses on hard-to-classify examples by down-weighting easy ones
2. WeightedCrossEntropyLoss — Standard CE with class weights for imbalance
3. LabelSmoothingCrossEntropy — CE with label smoothing for regularization
4. get_loss_fn — Factory function to select loss

Why Focal Loss for mental health?
  - Mental health datasets are inherently imbalanced (Stress=6.9%, Suicidal=21.2%)
  - The "easy" negative samples (Normal) dominate the gradient
  - Focal Loss down-weights easy examples, forcing the model to focus on
    harder boundary cases (e.g., distinguishing Stress from Anxiety)
  - gamma=2.0 is the standard for most tasks
  - alpha can be set to class frequencies for additional balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List

from .config import MH_CLASS_WEIGHTS_TENSOR, NUM_MH_CLASSES, TrainingConfig

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the model's estimated probability for the target class
    - gamma controls the rate at which easy examples are down-weighted
    - alpha balances class frequencies (can be None for uniform)
    
    For mental health:
    - Higher gamma (2.0-3.0) → model focuses more on hard boundary cases
      (e.g., differentiating Stress vs Anxiety, Bipolar vs Depression)
    - alpha computed from inverse class frequencies → handles imbalance
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] — raw logits
            targets: [batch_size] — ground truth class indices
            
        Returns:
            Scalar loss value
        """
        # Move alpha to same device as logits (fixes GPU/CPU device mismatch)
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        
        # Compute cross-entropy first
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", weight=alpha
        )
        
        # Get probability of target class
        pt = torch.exp(-ce_loss)  # pt = softmax probability of target class
        
        # Compute focal loss: -(1-pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss with optional label smoothing.
    
    For mental health, the weights are computed inversely proportional
    to class frequencies:
      weight_c = total_samples / (n_classes * count_c)
    
    This gives minority classes (Stress) higher weight and
    majority classes (Suicidal) lower weight.
    """
    
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(logits.device) if self.weights is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident,
    which is important for mental health detection where:
    1. There's inherent label noise (self-reported diagnoses may be inaccurate)
    2. Conditions exist on a spectrum (not binary categories)
    3. Overconfident predictions can be dangerous in clinical settings
    
    Smoothing factor: 0.1 is standard (targets are 0.9 on true class,
    0.1/6 ≈ 0.017 on each other class)
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Number of classes
        n_classes = logits.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smoothed = torch.full_like(logits, self.smoothing / (n_classes - 1))
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # KL divergence
        loss = -(smoothed * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(config: TrainingConfig) -> nn.Module:
    """
    Factory function to select and configure the loss function.
    
    Args:
        config: TrainingConfig with loss_type and parameters
        
    Returns:
        Configured loss module
    """
    loss_type = config.loss_type
    
    if loss_type == "focal":
        # Compute alpha from class weights if not provided
        alpha = config.focal_alpha
        if alpha is None:
            alpha = MH_CLASS_WEIGHTS_TENSOR
        
        logger.info(f"Using Focal Loss: gamma={config.focal_gamma}")
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=alpha,
        )
    
    elif loss_type == "weighted_ce":
        logger.info("Using Weighted Cross-Entropy Loss")
        return WeightedCrossEntropyLoss(
            weights=MH_CLASS_WEIGHTS_TENSOR,
        )
    
    elif loss_type == "label_smooth_ce":
        logger.info(f"Using Label Smoothing CE: smoothing={config.label_smoothing}")
        return LabelSmoothingCrossEntropy(
            smoothing=config.label_smoothing,
        )
    
    elif loss_type == "ce":
        logger.info("Using plain Cross-Entropy Loss")
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Choose from: 'focal', 'weighted_ce', 'label_smooth_ce', 'ce'")
"""
Loss Functions
===============
Advanced loss functions for mental health classification with class imbalance.

Losses provided:
1. FocalLoss — Focuses on hard-to-classify examples by down-weighting easy ones
2. PerClassGammaFocalLoss — Focal loss with different gamma per class
3. ConfusionFocalLoss — Focal loss + confusion penalty for dangerous misclassifications
4. WeightedCrossEntropyLoss — Standard CE with class weights for imbalance
5. LabelSmoothingCrossEntropy — CE with label smoothing for regularization
6. get_loss_fn — Factory function to select loss

Why ConfusionFocalLoss for mental health?
  - The biggest failure mode: 50% of Suicidal cases predicted as Depression
  - Per-class gamma: higher gamma for Depression (3.5) and Suicidal (3.0)
    forces the model to learn better feature boundaries for these hard classes
  - Confusion penalty: explicitly penalizes dangerous misclassifications
    (e.g., Suicidal → Depression, Suicidal → Normal) with an auxiliary loss term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple

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
        # Move alpha to same device AND dtype as logits (handles fp16 vs fp32 mismatch)
        alpha = self.alpha.to(device=logits.device, dtype=logits.dtype) if self.alpha is not None else None
        
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


class PerClassGammaFocalLoss(nn.Module):
    """
    Focal Loss with per-class gamma values.
    
    Different classes need different focus:
    - Easy classes (Normal, Bipolar): low gamma (2.0) — standard focal
    - Hard classes (Depression, Suicidal): high gamma (3.0-3.5) — extreme focus
    - Medium classes (Anxiety): moderate gamma (2.5)
    
    Per-class gamma directly addresses the confusion matrix by forcing the
    model to spend more capacity on separating confused class pairs.
    """
    
    def __init__(
        self,
        per_class_gamma: List[float],
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.per_class_gamma = torch.tensor(per_class_gamma, dtype=torch.float32)
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        dtype = logits.dtype  # Match logits dtype (may be float16 from mixed precision)
        gamma_tensor = self.per_class_gamma.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype) if self.alpha is not None else None
        
        # Standard CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=alpha)
        
        # Get probability of target class
        pt = torch.exp(-ce_loss)
        
        # Per-class gamma: for each sample, use gamma of its target class
        sample_gamma = gamma_tensor[targets]
        
        # Focal loss with per-class gamma
        focal_loss = ((1 - pt) ** sample_gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ConfusionFocalLoss(nn.Module):
    """
    Combined loss: Per-class gamma Focal Loss + Confusion Penalty.
    
    The confusion penalty explicitly penalizes dangerous misclassifications
    by adding an auxiliary term for specific (true_class, wrong_prediction) pairs.
    
    For example, if Suicidal(5) is misclassified as Depression(1):
    - The model predicts high probability for Depression
    - The confusion penalty adds extra loss proportional to p(Depression | Suicidal)
    - This forces the model to learn better separating features
    
    Architecture:
        Total Loss = FocalLoss + lambda * ConfusionPenalty
    
    Where lambda = confusion_penalty_weight (default 0.3)
    """
    
    def __init__(
        self,
        per_class_gamma: List[float],
        penalty_pairs: List[Tuple[int, int]],
        penalty_weight: float = 0.3,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            per_class_gamma: Per-class gamma values for focal loss
            penalty_pairs: List of (true_class, wrong_class) pairs to penalize
            penalty_weight: Weight for the confusion penalty term
            alpha: Class weights
            reduction: Loss reduction method
        """
        super().__init__()
        self.focal_loss = PerClassGammaFocalLoss(
            per_class_gamma=per_class_gamma,
            alpha=alpha,
            reduction=reduction,
        )
        self.penalty_pairs = penalty_pairs
        self.penalty_weight = penalty_weight
        self.reduction = reduction
        
        # Build penalty matrix: [num_classes, num_classes]
        # penalty_matrix[i][j] = penalty weight when true=i predicts=j
        self.penalty_matrix = torch.zeros((len(per_class_gamma), len(per_class_gamma)))
        for true_class, wrong_class in penalty_pairs:
            self.penalty_matrix[true_class, wrong_class] = 1.0
        
        logger.info(
            f"ConfusionFocalLoss: gamma={per_class_gamma}, "
            f"penalty_weight={penalty_weight}, "
            f"penalty_pairs={penalty_pairs}"
        )
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] — raw logits
            targets: [batch_size] — ground truth class indices
            
        Returns:
            Total loss = focal_loss + confusion_penalty
        """
        # 1. Per-class gamma focal loss
        focal = self.focal_loss(logits, targets)
        
        # 2. Confusion penalty
        probs = F.softmax(logits, dim=-1)
        device = logits.device
        penalty_matrix = self.penalty_matrix.to(device)
        
        # For each sample, get which wrong classes are penalized
        # penalty_per_sample[b] = sum over wrong classes: penalty_matrix[target[b], j] * probs[b, j]
        penalty_for_target = penalty_matrix[targets]  # [batch, num_classes]
        confusion_penalty = (penalty_for_target * probs).sum(dim=-1)  # [batch]
        
        if self.reduction == "mean":
            confusion_penalty = confusion_penalty.mean()
        elif self.reduction == "sum":
            confusion_penalty = confusion_penalty.sum()
        
        total_loss = focal + self.penalty_weight * confusion_penalty
        
        return total_loss


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
    
    if loss_type == "confusion_focal":
        # Compute alpha from class weights if not provided
        alpha = config.focal_alpha
        if alpha is None:
            alpha = MH_CLASS_WEIGHTS_TENSOR
        
        # Use per-class gamma if provided, otherwise uniform gamma
        per_class_gamma = config.per_class_gamma if config.per_class_gamma is not None else [config.focal_gamma] * config.num_labels
        
        logger.info(
            f"Using ConfusionFocalLoss: per_class_gamma={per_class_gamma}, "
            f"penalty_weight={config.confusion_penalty_weight}, "
            f"penalty_pairs={config.confusion_penalty_pairs}"
        )
        
        return ConfusionFocalLoss(
            per_class_gamma=per_class_gamma,
            penalty_pairs=config.confusion_penalty_pairs,
            penalty_weight=config.confusion_penalty_weight,
            alpha=alpha,
        )
    
    elif loss_type == "focal":
        # Compute alpha from class weights if not provided
        alpha = config.focal_alpha
        if alpha is None:
            alpha = MH_CLASS_WEIGHTS_TENSOR
        
        # Check if per-class gamma is set
        per_class_gamma = config.per_class_gamma
        if per_class_gamma is not None:
            logger.info(f"Using PerClassGammaFocalLoss: gamma={per_class_gamma}")
            return PerClassGammaFocalLoss(
                per_class_gamma=per_class_gamma,
                alpha=alpha,
            )
        
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
                        f"Choose from: 'focal', 'weighted_ce', 'label_smooth_ce', 'ce', 'confusion_focal'")
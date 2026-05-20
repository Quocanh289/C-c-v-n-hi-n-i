"""
Loss Functions Module
======================
Advanced loss functions for multi-class emotion classification:
- Focal Loss: Focuses on hard-to-classify examples
- ASL (Asymmetric Loss): Different handling for positive/negative
- Combined Loss: Weighted combination of two losses
- Class-weighted Cross Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Down-weights easy examples and focuses on hard ones.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Focusing parameter (γ >= 0). Default: 2.0
        weight: Class weights tensor. Optional.
        reduction: 'mean' or 'sum'. Default: 'mean'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] raw scores
            targets: [batch_size] class indices
        Returns:
            Scalar loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather the probabilities of target classes
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Standard cross entropy
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")
        
        # Apply focal weighting
        loss = focal_weight * ce_loss
        
        # Apply class weights
        if self.weight is not None:
            class_weight = self.weight.to(logits.device).gather(0, targets)
            loss = loss * class_weight
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ASLLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Applies different focusing for positive vs negative samples.
    
    Args:
        gamma_neg: Focusing parameter for negative samples. Default: 4.0
        gamma_pos: Focusing parameter for positive samples. Default: 0.0
        clip: Probability clipping value. Default: 0.05
        reduction: 'mean' or 'sum'. Default: 'mean'
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] raw scores
            targets: [batch_size] class indices (for multi-class, we one-hot encode)
        Returns:
            Scalar loss value
        """
        # Convert to multi-label format
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Probability with numerical stability
        sigmoid = torch.sigmoid(logits)
        xs_pos = sigmoid
        xs_neg = 1 - sigmoid
        
        # ASL weights
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Asymmetric focusing
        pos_weight = xs_pos ** self.gamma_pos
        neg_weight = xs_neg ** self.gamma_neg
        
        # Binary cross entropy
        loss = (
            -targets_one_hot * torch.log(xs_pos.clamp(min=1e-8)) * pos_weight
            - (1 - targets_one_hot) * torch.log(xs_neg.clamp(min=1e-8)) * neg_weight
        )
        
        loss = loss.sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of primary and secondary loss functions.
    Typically: 0.7 * FocalLoss + 0.3 * CrossEntropyLoss
    """
    
    def __init__(
        self,
        primary_loss: str = "focal",
        secondary_loss: str = "bce",
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3,
        focal_gamma: float = 2.0,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        
        # Primary loss
        if primary_loss == "focal":
            self.primary = FocalLoss(gamma=focal_gamma, weight=class_weights)
        elif primary_loss == "asl":
            self.primary = ASLLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos)
        elif primary_loss == "bce":
            self.primary = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            self.primary = FocalLoss(gamma=focal_gamma, weight=class_weights)
        
        # Secondary loss
        if secondary_loss == "bce":
            self.secondary = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        elif secondary_loss == "focal":
            self.secondary = FocalLoss(gamma=focal_gamma, weight=class_weights)
        else:
            self.secondary = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined weighted loss.
        """
        loss1 = self.primary(logits, targets)
        loss2 = self.secondary(logits, targets)
        return self.primary_weight * loss1 + self.secondary_weight * loss2
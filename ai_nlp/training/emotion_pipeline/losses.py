"""
Loss Functions Module
=====================
Loss functions for multi-label emotion classification:
- BCEWithLogitsLoss: Standard multi-label binary cross-entropy
- MultiLabelFocalLoss: Focal loss adapted for multi-label
- AsymmetricLoss (ASL): State-of-the-art for multi-label with class imbalance
- CombinedLoss: Weighted combination

All losses accept class weights for handling label imbalance.
For multi-label, each label is treated as an independent binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union


class MultiLabelBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with Logits for multi-label classification.
    
    Treats each label as an independent binary classification.
    Supports per-label class weights for handling imbalance.
    
    Args:
        weight: Per-label weights tensor [num_labels]
        reduction: 'mean' or 'sum'
        pos_weight: Positive class weights (for asymmetric positive/negative weighting)
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            weight=weight,
            reduction=reduction,
            pos_weight=pos_weight,
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] raw scores (before sigmoid)
            targets: [batch_size, num_labels] binary multi-hot
        Returns:
            Scalar loss
        """
        return self.loss_fn(logits, targets)


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss adapted for multi-label classification.
    
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    For multi-label, applies focal weighting per-label independently.
    Focuses on hard-to-classify examples and addresses class imbalance.
    
    Args:
        gamma: Focusing parameter. Default: 2.0
        alpha: Per-class weighting. Optional tensor [num_labels].
        weight: Additional per-class weight. Optional tensor [num_labels].
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] raw scores
            targets: [batch_size, num_labels] binary multi-hot
        Returns:
            Scalar loss
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Focal weight: (1 - p_t)^gamma
        # For positive: p_t = probs, for negative: p_t = 1 - probs
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma,
        )
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Apply focal weighting
        loss = focal_weight * bce_loss
        
        # Apply alpha (class balancing)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha.unsqueeze(0)
        
        # Apply per-class weights
        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w.unsqueeze(0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    
    Applies different focusing for positive vs negative samples.
    This is important because in multi-label classification,
    negatives typically dominate (most labels are 0).
    
    ASL down-weights easy negatives more aggressively than easy positives,
    helping with extreme label sparsity in multi-label settings.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021)
    
    Args:
        gamma_neg: Focusing parameter for negative samples. Default: 4.0
        gamma_pos: Focusing parameter for positive samples. Default: 0.0
        clip: Probability clipping to avoid over-suppression. Default: 0.05
        eps: Numerical stability. Default: 1e-8
        reduction: 'mean' or 'sum'
        weight: Per-label class weights [num_labels]
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] raw scores
            targets: [batch_size, num_labels] binary multi-hot
        Returns:
            Scalar loss
        """
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)
        probs = probs.clamp(self.eps, 1 - self.eps)
        
        # Separate positive and negative probabilities
        xs_pos = probs
        xs_neg = 1 - probs
        
        # Asymmetric clamping (for negatives)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic cross-entropy loss
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric focusing weights
        pos_weight = (1 - xs_pos) ** self.gamma_pos  # For positives
        neg_weight = (xs_pos) ** self.gamma_neg       # For negatives (using xs_pos, not xs_neg)
        
        # Apply weights
        loss = -(
            loss_pos * pos_weight * (1 - targets + targets)
            + loss_neg * neg_weight * (1 - targets)
        )
        
        # Apply per-label weights
        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w.unsqueeze(0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of primary and secondary loss functions.
    
    Common configurations:
    - "asl" + "bce": ASL handles imbalance, BCE provides stability
    - "focal" + "bce": Focal handles hard examples, BCE for easy ones
    - "bce" + "none": Pure BCE (equivalent to single loss)
    
    Supports class weights, dual-head (28+9) losses for fine+coarse.
    """
    
    def __init__(
        self,
        primary_loss: str = "asl",
        secondary_loss: str = "bce",
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3,
        num_labels: int = 28,
        num_coarse_labels: int = 9,
        task_type: str = "multi_label",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        coarse_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        self.task_type = task_type
        self.num_labels = num_labels
        self.num_coarse_labels = num_coarse_labels
        
        # Primary loss
        self.primary = self._build_loss(
            primary_loss,
            focal_gamma, focal_alpha,
            asl_gamma_neg, asl_gamma_pos, asl_clip,
            class_weights,
        )
        
        # Secondary loss
        self.secondary = self._build_loss(
            secondary_loss if secondary_loss != "none" else primary_loss,
            focal_gamma, focal_alpha,
            asl_gamma_neg, asl_gamma_pos, asl_clip,
            class_weights,
        )
        
        # Optional coarse loss for dual-head models
        self.coarse_loss = None
        if task_type == "dual_head":
            self.coarse_loss = self._build_loss(
                primary_loss,
                focal_gamma, focal_alpha,
                asl_gamma_neg, asl_gamma_pos, asl_clip,
                coarse_class_weights,
            )
    
    def _build_loss(
        self,
        loss_type: str,
        focal_gamma: float,
        focal_alpha: Optional[List[float]],
        asl_gamma_neg: float,
        asl_gamma_pos: float,
        asl_clip: float,
        class_weights: Optional[torch.Tensor],
    ) -> nn.Module:
        """Build a loss function by type."""
        alpha_tensor = None
        if focal_alpha is not None:
            alpha_tensor = torch.tensor(focal_alpha, dtype=torch.float32)
        
        if loss_type == "bce":
            return MultiLabelBCEWithLogitsLoss(weight=class_weights)
        elif loss_type == "focal":
            return MultiLabelFocalLoss(
                gamma=focal_gamma,
                alpha=alpha_tensor,
                weight=class_weights,
            )
        elif loss_type == "asl":
            return AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
                weight=class_weights,
            )
        elif loss_type == "combined":
            # Nested combined loss
            return CombinedLoss(
                primary_loss="asl",
                secondary_loss="bce",
                primary_weight=0.7,
                secondary_weight=0.3,
                class_weights=class_weights,
            )
        else:
            return AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
                weight=class_weights,
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        coarse_logits: Optional[torch.Tensor] = None,
        coarse_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        For dual_head, expects:
        - logits: [batch, 28] fine-grained logits
        - coarse_logits: [batch, 9] coarse logits (optional)
        
        Args:
            logits: Primary output logits
            targets: Primary targets (multi-hot)
            coarse_logits: Optional coarse logits for dual head
            coarse_targets: Optional coarse targets
        
        Returns:
            Scalar loss
        """
        loss1 = self.primary(logits, targets)
        loss2 = self.secondary(logits, targets)
        
        total_loss = self.primary_weight * loss1 + self.secondary_weight * loss2
        
        # Add coarse loss if available (dual head)
        if coarse_logits is not None and coarse_targets is not None and self.coarse_loss:
            coarse_loss = self.coarse_loss(coarse_logits, coarse_targets)
            total_loss = total_loss + 0.3 * coarse_loss
        
        return total_loss


# ============================================================
# Loss Function Registry
# ============================================================

LOSS_REGISTRY = {
    "bce": MultiLabelBCEWithLogitsLoss,
    "focal": MultiLabelFocalLoss,
    "asl": AsymmetricLoss,
    "combined": CombinedLoss,
}


def get_loss_function(
    config,
    class_weights: Optional[torch.Tensor] = None,
    coarse_class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory function to get the appropriate loss function.
    
    Args:
        config: TrainingConfig object
        class_weights: Per-label weights for the primary head
        coarse_class_weights: Per-label weights for the coarse head (dual mode)
    
    Returns:
        Loss module
    """
    return CombinedLoss(
        primary_loss=config.primary_loss,
        secondary_loss=config.secondary_loss,
        primary_weight=config.primary_weight,
        secondary_weight=config.secondary_weight,
        num_labels=config.num_labels,
        num_coarse_labels=config.num_coarse_labels,
        task_type=config.task_type,
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha,
        asl_gamma_neg=config.asl_gamma_neg,
        asl_gamma_pos=config.asl_gamma_pos,
        asl_clip=config.asl_clip,
        class_weights=class_weights,
        coarse_class_weights=coarse_class_weights,
    )
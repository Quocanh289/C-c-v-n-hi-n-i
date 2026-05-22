"""
Evaluation Metrics
===================
Comprehensive evaluation metrics for mental health classification.

Metrics computed:
1. Accuracy — Overall correct predictions
2. Macro Precision — Average precision across all classes (unweighted)
3. Macro Recall — Average recall across all classes (unweighted)
4. Macro F1 — Harmonic mean of macro precision and recall (PRIMARY METRIC)
5. Weighted F1 — F1 weighted by number of samples per class
6. Per-class Precision, Recall, F1 — For identifying weak classes
7. Confusion Matrix — For analyzing misclassification patterns
8. Classification Report — Full sklearn-style report

Why Macro F1 is the primary metric:
  - Accuracy is MISLEADING for imbalanced data
    (e.g., 87% accuracy could mean just predicting all as Suicidal/Anxiety)
  - Macro F1 treats all 7 classes equally → measures true generalization
  - Weighted F1 is useful for deployment (gives more weight to frequent classes)
  - For mental health, BOTH macro F1 (research) and weighted F1 (production) matter
  
Expected performance:
  - DeBERTa-v3-base + LoRA: 0.88-0.92 macro F1
  - XLM-RoBERTa-base + LoRA: 0.86-0.90 macro F1
  - Baseline (most-frequent class): 0.12 macro F1
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
)

from .config import MENTAL_HEALTH_LABELS, IDX_TO_MH_LABELS

logger = logging.getLogger(__name__)


class MentalHealthMetrics:
    """
    Comprehensive metrics computation for mental health classification.
    
    Usage:
        metrics = MentalHealthMetrics()
        
        # Accumulate predictions
        for batch in dataloader:
            metrics.update(predictions, labels)
        
        # Compute final results
        results = metrics.compute()
    """
    
    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or MENTAL_HEALTH_LABELS
        self.num_classes = len(self.labels)
        
        # Accumulators
        self.all_predictions: List[int] = []
        self.all_labels: List[int] = []
        self.all_probs: List[np.ndarray] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Predicted class indices [batch]
            labels: Ground truth class indices [batch]
            probs: Class probabilities [batch, num_classes]
        """
        self.all_predictions.extend(predictions.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            Dict with all metrics, classification report, confusion matrix
        """
        if len(self.all_predictions) == 0:
            logger.warning("No predictions accumulated")
            return {"error": "no_predictions"}
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Macro metrics (unweighted average across classes)
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Weighted metrics (weighted by class frequency)
        weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class = {}
        for i, label in enumerate(self.labels):
            per_class[label] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int(np.sum(y_true == i)),
            }
        
        # Support (number of true instances per class)
        class_support = {}
        for i, label in enumerate(self.labels):
            class_support[label] = int(np.sum(y_true == i))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Matthews Correlation Coefficient (better than F1 for imbalance)
        mcc = float(matthews_corrcoef(y_true, y_pred))
        
        # ROC AUC (one-vs-rest)
        roc_auc = None
        if len(self.all_probs) > 0 and len(self.all_probs) == len(y_true):
            try:
                probs = np.array(self.all_probs)
                roc_auc = roc_auc_score(
                    y_true, probs, multi_class="ovr", average="macro"
                ).tolist()
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        # Classification report (sklearn format)
        report = classification_report(
            y_true, y_pred,
            target_names=self.labels,
            output_dict=True,
            zero_division=0,
        )
        
        # Find hardest classes (lowest F1)
        hardest_classes = sorted(
            [(label, per_class[label]["f1"], per_class[label]["support"])
             for label in self.labels],
            key=lambda x: x[1],
        )
        
        # Find most common misclassifications
        misclassifications = self._analyze_misclassifications(cm)
        
        results = {
            # Primary metrics
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "weighted_f1": float(weighted_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "mcc": mcc,
            
            # Per-class breakdown
            "per_class": per_class,
            "class_support": class_support,
            
            # Detailed analysis
            "hardest_classes": [
                {"label": l, "f1": f, "support": s}
                for l, f, s in hardest_classes
            ],
            "misclassifications": misclassifications,
            
            # Raw data
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(y_true),
        }
        
        if roc_auc is not None:
            results["roc_auc_macro"] = roc_auc
        
        return results
    
    def _analyze_misclassifications(self, cm: np.ndarray) -> List[Dict]:
        """
        Analyze the most common misclassification patterns.
        
        Returns:
            List of misclassification patterns sorted by frequency
        """
        misclassifications = []
        
        # For each true class, find where its samples are misclassified
        for true_idx in range(self.num_classes):
            true_label = self.labels[true_idx]
            
            # Get distribution of predictions for this true class
            pred_counts = cm[true_idx, :]
            total_true = pred_counts.sum()
            
            if total_true == 0:
                continue
            
            # Find misclassification targets (excluding correct predictions)
            for pred_idx in range(self.num_classes):
                count = int(pred_counts[pred_idx])
                if pred_idx != true_idx and count > 0:
                    pred_label = self.labels[pred_idx]
                    percentage = count / total_true * 100
                    
                    misclassifications.append({
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                        "percentage": round(percentage, 1),
                    })
        
        # Sort by count descending
        misclassifications.sort(key=lambda x: x["count"], reverse=True)
        
        return misclassifications[:10]  # Top 10
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to compute all metrics at once.
    
    Args:
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        y_probs: Class probabilities (optional, for ROC AUC)
        labels: Class label names
        
    Returns:
        Dict with all metrics
    """
    metrics = MentalHealthMetrics(labels=labels)
    metrics.all_labels = y_true.tolist()
    metrics.all_predictions = y_pred.tolist()
    if y_probs is not None:
        metrics.all_probs = y_probs.tolist() if isinstance(y_probs, np.ndarray) else y_probs
    return metrics.compute()
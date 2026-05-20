"""
Metrics Module
===============
Comprehensive evaluation metrics for emotion classification:
- Per-class precision, recall, F1
- Macro/micro/weighted averages
- Confusion matrix
- Matthews Correlation Coefficient (MCC)
- Accuracy
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field, asdict

from .config import COARSE_EMOTIONS, COARSE_TO_IDX, IDX_TO_COARSE


@dataclass
class EmotionMetrics:
    """Container for all evaluation metrics."""
    
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    mcc: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for JSON logging."""
        result = {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_precision": round(self.weighted_precision, 4),
            "weighted_recall": round(self.weighted_recall, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "mcc": round(self.mcc, 4),
        }
        
        # Per-class metrics
        for label, metrics in self.per_class.items():
            for metric_name, value in metrics.items():
                result[f"{label}_{metric_name}"] = round(value, 4)
        
        return result
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = COARSE_EMOTIONS,
) -> EmotionMetrics:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (indices)
        y_pred: Predicted labels (indices)
        labels: List of label names
    
    Returns:
        EmotionMetrics object with all computed metrics
    """
    n_classes = len(labels)
    
    # Confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    
    # Per-class metrics
    per_class = {}
    class_support = {}
    
    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()  # Total true instances
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
        class_support[i] = support
    
    # Macro averages
    macro_precision = np.mean([m["precision"] for m in per_class.values()])
    macro_recall = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class.values()])
    
    # Weighted averages
    total_support = sum(class_support.values())
    if total_support > 0:
        weighted_precision = sum(
            per_class[labels[i]]["precision"] * class_support[i]
            for i in range(n_classes)
        ) / total_support
        weighted_recall = sum(
            per_class[labels[i]]["recall"] * class_support[i]
            for i in range(n_classes)
        ) / total_support
        weighted_f1 = sum(
            per_class[labels[i]]["f1"] * class_support[i]
            for i in range(n_classes)
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    # Accuracy
    accuracy = (y_pred == y_true).sum() / len(y_true) if len(y_true) > 0 else 0.0
    
    # Matthews Correlation Coefficient (multi-class)
    mcc = _compute_multiclass_mcc(conf_matrix, n_classes)
    
    # Class distribution
    class_dist = Counter(y_true)
    dist_dict = {labels[i]: class_dist.get(i, 0) for i in range(n_classes)}
    
    return EmotionMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        mcc=mcc,
        per_class=per_class,
        confusion_matrix=conf_matrix.tolist(),
        class_distribution=dist_dict,
    )


def _compute_multiclass_mcc(conf_matrix: np.ndarray, n_classes: int) -> float:
    """Compute Matthews Correlation Coefficient for multi-class."""
    t_sum = conf_matrix.sum(axis=1, keepdims=True)  # row sums
    p_sum = conf_matrix.sum(axis=0, keepdims=True)  # column sums
    
    numerator = conf_matrix.sum() * np.trace(conf_matrix) - (t_sum * p_sum).sum()
    
    denominator = np.sqrt(
        (conf_matrix.sum() ** 2 - (p_sum ** 2).sum())
        * (conf_matrix.sum() ** 2 - (t_sum ** 2).sum())
    )
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = COARSE_EMOTIONS,
) -> EmotionMetrics:
    """
    Compute metrics from model logits and ground truth labels.
    
    Args:
        logits: Raw model outputs [num_samples, num_classes]
        labels: Ground truth indices [num_samples]
        label_names: List of class names
    
    Returns:
        EmotionMetrics object
    """
    predictions = np.argmax(logits, axis=1) if logits.ndim > 1 else logits
    return compute_class_metrics(labels, predictions, label_names)


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = COARSE_EMOTIONS,
) -> Dict[str, float]:
    """
    Find optimal classification thresholds for each class.
    This is useful for multi-label scenarios where you want
    per-class probability thresholds.
    
    Args:
        probs: Predicted probabilities [num_samples, num_classes]
        labels: Ground truth labels (one-hot or indices)
        label_names: List of label names
    
    Returns:
        Dict mapping label names to optimal thresholds
    """
    # Convert labels to one-hot if needed
    if labels.ndim == 1:
        n_classes = probs.shape[1]
        one_hot = np.zeros((len(labels), n_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        labels = one_hot
    
    best_thresholds = {}
    
    for i, label in enumerate(label_names):
        best_f1 = 0.0
        best_thr = 0.5
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.95, 0.05):
            pred = (probs[:, i] >= threshold).astype(int)
            true = labels[:, i]
            
            tp = (pred & true).sum()
            fp = (pred & ~true).sum()
            fn = (~pred & true).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thr = threshold
        
        best_thresholds[label] = best_thr
    
    return best_thresholds
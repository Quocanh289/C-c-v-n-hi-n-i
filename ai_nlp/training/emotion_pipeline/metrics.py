"""
Metrics Module
===============
Comprehensive multi-label evaluation metrics for emotion classification:
- Per-label precision, recall, F1
- Macro/Micro/Weighted/Subset averages
- Hamming Loss
- Subset Accuracy (Exact Match)
- Confusion matrices (per-label)
- Matthews Correlation Coefficient
- Calibration analysis (ECE)
- Overfitting and data leakage detection heuristics

Key difference from single-label: multi-label metrics evaluate each
label independently using binary classification metrics.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

from .config import (
    GOEMOTIONS_28,
    GOEMOTIONS_IDX_TO_28,
    COARSE_EMOTIONS,
)


@dataclass
class EmotionMetrics:
    """
    Container for all multi-label and single-label evaluation metrics.
    
    Multi-label metrics:
    - macro_f1_micro_avg: Macro-average of per-label F1 (each label weighted equally)
    - micro_f1: Micro-average F1 (global TP/FP/FN)
    - weighted_f1: Weighted F1 (each label weighted by support)
    - subsets_accuracy: Exact match ratio (all labels correct)
    - hamming_loss: Fraction of incorrect labels
    - per_label: Dict of per-label metrics
    
    Single-label / aggregation metrics (backward compat):
    - accuracy
    - macro_precision, macro_recall, macro_f1
    - mcc
    """
    
    # Multi-label metrics
    macro_f1_micro_avg: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    subsets_accuracy: float = 0.0
    hamming_loss: float = 0.0
    
    # Aggregate metrics (backward compat)
    accuracy: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1_old: float = 0.0
    mcc: float = 0.0
    
    # Per-label breakdown
    per_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Calibration metrics
    expected_calibration_error: float = 0.0
    
    # Label distribution
    label_frequencies: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for JSON serialization."""
        result = {
            "macro_f1_micro_avg": round(self.macro_f1_micro_avg, 4),
            "micro_f1": round(self.micro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "subsets_accuracy": round(self.subsets_accuracy, 4),
            "hamming_loss": round(self.hamming_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "weighted_precision": round(self.weighted_precision, 4),
            "weighted_recall": round(self.weighted_recall, 4),
            "mcc": round(self.mcc, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
        }
        
        # Per-label metrics
        for label, metrics in self.per_label.items():
            for metric_name, value in metrics.items():
                result[f"{label}_{metric_name}"] = round(value, 4)
        
        # Label frequencies
        for label, freq in self.label_frequencies.items():
            result[f"{label}_frequency"] = round(freq, 4)
        
        return result
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================
# Multi-Label Metrics Computation
# ============================================================

def compute_multi_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = GOEMOTIONS_28,
    thresholds: Optional[Union[float, List[float]]] = None,
    probs: Optional[np.ndarray] = None,
) -> EmotionMetrics:
    """
    Compute comprehensive multi-label classification metrics.
    
    Args:
        y_true: Ground truth multi-hot [num_samples, num_labels]
        y_pred: Predicted multi-hot [num_samples, num_labels]
        label_names: Names of labels
        thresholds: Classification thresholds (for analysis)
        probs: Raw probabilities (for calibration)
    
    Returns:
        EmotionMetrics with all computed values
    """
    n_samples, n_labels = y_true.shape
    
    # Ensure binary
    y_true_bin = (y_true > 0.5).astype(int)
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    # ============================================================
    # Per-label metrics
    # ============================================================
    per_label = {}
    label_supports = {}
    
    for i, label in enumerate(label_names):
        tp = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1))
        fp = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
        fn = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0))
        tn = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
        support = np.sum(y_true_bin[:, i])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        label_supports[i] = support
    
    # ============================================================
    # Macro averages (unweighted mean of per-label metrics)
    # ============================================================
    macro_precision = np.mean([m["precision"] for m in per_label.values()])
    macro_recall = np.mean([m["recall"] for m in per_label.values()])
    macro_f1 = np.mean([m["f1"] for m in per_label.values()])
    
    # ============================================================
    # Micro averages (global TP/FP/FN across all labels)
    # ============================================================
    global_tp = np.sum([m["tp"] for m in per_label.values()])
    global_fp = np.sum([m["fp"] for m in per_label.values()])
    global_fn = np.sum([m["fn"] for m in per_label.values()])
    
    micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # ============================================================
    # Weighted averages (weighted by support)
    # ============================================================
    total_support = sum(label_supports.values())
    if total_support > 0:
        weighted_precision = sum(
            per_label[label_names[i]]["precision"] * label_supports[i]
            for i in range(n_labels)
        ) / total_support
        weighted_recall = sum(
            per_label[label_names[i]]["recall"] * label_supports[i]
            for i in range(n_labels)
        ) / total_support
        weighted_f1 = sum(
            per_label[label_names[i]]["f1"] * label_supports[i]
            for i in range(n_labels)
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    # ============================================================
    # Subset Accuracy (Exact Match Ratio)
    # ============================================================
    exact_matches = np.all(y_true_bin == y_pred_bin, axis=1)
    subsets_accuracy = np.mean(exact_matches)
    
    # ============================================================
    # Hamming Loss
    # ============================================================
    hamming_loss = np.mean(y_true_bin != y_pred_bin)
    
    # ============================================================
    # MCC (Matthews Correlation Coefficient)
    # ============================================================
    mcc = _compute_multilabel_mcc(y_true_bin, y_pred_bin)
    
    # ============================================================
    # Label frequencies
    # ============================================================
    label_freqs = {label_names[i]: float(y_true_bin[:, i].mean()) for i in range(n_labels)}
    
    # ============================================================
    # Calibration (if probabilities provided)
    # ============================================================
    ece = 0.0
    if probs is not None:
        ece = compute_expected_calibration_error(probs, y_true_bin)
    
    # ============================================================
    # Aggregate accuracy (single-label equivalence)
    # ============================================================
    # For backward compatibility: convert multi-label to "predominant class"
    y_true_agg = np.argmax(y_true_bin, axis=1) if n_labels <= 28 else y_true_bin
    y_pred_agg = np.argmax(y_pred_bin, axis=1) if n_labels <= 28 else y_pred_bin
    accuracy = np.mean(y_true_agg == y_pred_agg)
    
    return EmotionMetrics(
        macro_f1_micro_avg=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        subsets_accuracy=subsets_accuracy,
        hamming_loss=hamming_loss,
        accuracy=accuracy,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        mcc=mcc,
        per_label=per_label,
        expected_calibration_error=ece,
        label_frequencies=label_freqs,
    )


def _compute_multilabel_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MCC for multi-label classification."""
    n_samples, n_labels = y_true.shape
    
    # Flatten to treat each label-sample pair independently
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ============================================================
# Calibration Analysis
# ============================================================

def compute_expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multi-label.
    
    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better. ECE < 0.05 is well-calibrated.
    
    For multi-label, computes ECE across all label predictions.
    
    Args:
        probs: Predicted probabilities [num_samples, num_labels]
        targets: Ground truth [num_samples, num_labels]
        n_bins: Number of confidence bins
    
    Returns:
        ECE score
    """
    # Flatten all predictions
    confidences = probs.flatten()
    accuracies = targets.flatten()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return float(ece)


def compute_reliability_diagram_data(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Get data for creating reliability diagrams.
    
    Returns:
        Dict with bin_confidences, bin_accuracies, bin_counts
    """
    confidences = probs.flatten()
    accuracies = targets.flatten()
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_confidences.append(float(confidences[in_bin].mean()))
            bin_accuracies.append(float(accuracies[in_bin].mean()))
            bin_counts.append(int(in_bin.sum()))
    
    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


# ============================================================
# Overfitting & Data Leakage Detection
# ============================================================

def detect_overfitting(
    train_metrics: EmotionMetrics,
    val_metrics: EmotionMetrics,
    threshold: float = 0.1,
) -> Dict:
    """
    Detect potential overfitting by comparing train/val metrics.
    
    Signs of overfitting:
    - Large gap between train and validation performance
    - Train metrics near perfect while validation is much lower
    - Per-label gaps > threshold for several labels
    
    Args:
        train_metrics: Metrics on training set
        val_metrics: Metrics on validation set
        threshold: Gap threshold for flagging
    
    Returns:
        Dict with overfitting_risk, gaps, flagged_labels
    """
    gaps = {}
    flagged = []
    
    # Compare macro F1
    f1_gap = train_metrics.macro_f1_micro_avg - val_metrics.macro_f1_micro_avg
    gaps["macro_f1_gap"] = f1_gap
    
    # Compare per-label F1
    for label in set(list(train_metrics.per_label.keys()) + list(val_metrics.per_label.keys())):
        train_f1 = train_metrics.per_label.get(label, {}).get("f1", 0)
        val_f1 = val_metrics.per_label.get(label, {}).get("f1", 0)
        gap = train_f1 - val_f1
        gaps[f"f1_gap_{label}"] = gap
        if gap > threshold:
            flagged.append(label)
    
    # Determine risk level
    risk = "low"
    if f1_gap > threshold * 3:
        risk = "high"
    elif f1_gap > threshold * 2:
        risk = "medium"
    elif f1_gap > threshold:
        risk = "moderate"
    
    return {
        "overfitting_risk": risk,
        "macro_f1_gap": round(f1_gap, 4),
        "flagged_labels": flagged,
        "n_flagged": len(flagged),
        "per_label_gaps": {k: round(v, 4) for k, v in gaps.items()},
    }


def detect_data_leakage(
    train_metrics: EmotionMetrics,
    test_metrics: EmotionMetrics,
    threshold: float = 0.15,
) -> Dict:
    """
    Detect potential data leakage between train and test.
    
    Signs of leakage:
    - Test metrics are suspiciously close to train metrics
    - Test metrics are too high (> 0.95) suggesting memorization
    - Distribution shift indicators
    
    Args:
        train_metrics: Metrics on training set
        test_metrics: Metrics on test set
        threshold: Gap threshold for flagging
    
    Returns:
        Dict with leakage_risk, gaps, indicators
    """
    indicators = {}
    
    # Super high test performance suggests leakage or overfitting
    indicators["test_macro_f1_suspiciously_high"] = float(test_metrics.macro_f1_micro_avg > 0.95)
    
    # Train-test gaps
    f1_gap = abs(train_metrics.macro_f1_micro_avg - test_metrics.macro_f1_micro_avg)
    indicators["train_test_f1_gap"] = round(f1_gap, 4)
    
    # Extremely high subset accuracy suggests memorization
    indicators["test_subsets_accuracy"] = round(test_metrics.subsets_accuracy, 4)
    
    # Risk assessment
    risk = "low"
    if test_metrics.macro_f1_micro_avg > 0.95 and f1_gap < 0.02:
        risk = "leakage_suspected"
    elif test_metrics.macro_f1_micro_avg > 0.90 and f1_gap < 0.03:
        risk = "moderate"
    
    return {
        "data_leakage_risk": risk,
        "indicators": indicators,
    }


def detect_distribution_shift(
    train_label_freqs: Dict[str, float],
    test_label_freqs: Dict[str, float],
    threshold: float = 0.05,
) -> Dict:
    """
    Detect distribution shift between train and test label distributions.
    
    Args:
        train_label_freqs: Label frequencies in training set
        test_label_freqs: Label frequencies in test set
        threshold: Frequency difference threshold
    
    Returns:
        Dict with shift_risk, shifted_labels, max_shift
    """
    all_labels = set(list(train_label_freqs.keys()) + list(test_label_freqs.keys()))
    
    shifts = {}
    max_shift = 0.0
    shifted_labels = []
    
    for label in all_labels:
        train_freq = train_label_freqs.get(label, 0)
        test_freq = test_label_freqs.get(label, 0)
        shift = abs(train_freq - test_freq)
        shifts[label] = round(shift, 4)
        max_shift = max(max_shift, shift)
        if shift > threshold:
            shifted_labels.append(label)
    
    risk = "low"
    if max_shift > threshold * 3:
        risk = "high"
    elif max_shift > threshold * 2:
        risk = "moderate"
    
    return {
        "distribution_shift_risk": risk,
        "max_shift": round(max_shift, 4),
        "shifted_labels": shifted_labels,
        "per_label_shifts": shifts,
    }


# ============================================================
# Convenience functions
# ============================================================

def compute_all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    label_names: List[str] = GOEMOTIONS_28,
    thresholds: Optional[Union[float, List[float]]] = None,
) -> EmotionMetrics:
    """
    Compute metrics from model logits and ground truth labels.
    
    Args:
        logits: Raw model outputs [num_samples, num_outputs]
        labels: Ground truth [num_samples, num_outputs]
        label_names: List of label names
        thresholds: Classification thresholds
    
    Returns:
        EmotionMetrics object
    """
    # Apply sigmoid or softmax
    if logits.shape[-1] == len(GOEMOTIONS_28):
        # Multi-label case
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        if thresholds is not None:
            if isinstance(thresholds, (int, float)):
                predictions = (probs > thresholds).astype(int)
            else:
                predictions = (probs > np.array(thresholds)).astype(int)
        else:
            predictions = (probs > 0.5).astype(int)
    else:
        # Single-label case (9-class softmax)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        predictions = np.argmax(probs, axis=1)
        # Convert to multi-hot for metrics
        multi_hot = np.zeros((len(labels), len(label_names)))
        multi_hot[np.arange(len(labels)), predictions] = 1
        predictions = multi_hot
    
    # Ensure labels are multi-hot
    if labels.ndim == 1:
        n_samples = len(labels)
        multi_hot_labels = np.zeros((n_samples, len(label_names)))
        multi_hot_labels[np.arange(n_samples), labels] = 1
        labels = multi_hot_labels
    
    return compute_multi_label_metrics(
        y_true=labels,
        y_pred=predictions,
        label_names=label_names,
        probs=probs,
    )
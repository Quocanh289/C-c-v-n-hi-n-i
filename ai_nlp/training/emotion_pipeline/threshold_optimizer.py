"""
Threshold Optimizer Module
===========================
Finds optimal per-label probability thresholds for multi-label emotion classification.
Uses Bayesian optimization-inspired random search over per-label thresholds.
Optimized thresholds improve Macro F1 by balancing precision-recall tradeoffs.

For multi-label, each label has an independent threshold [0, 1].
Default threshold is 0.5, but optimal thresholds vary by label due to:
- Class imbalance (rare labels need lower thresholds)
- Label difficulty (harder labels need different thresholds)
- Precision-recall preference per label
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .config import GOEMOTIONS_28
from .metrics import EmotionMetrics, compute_multi_label_metrics

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes per-label probability thresholds to maximize macro F1
    for multi-label emotion classification.
    
    Uses random search over continuous threshold space [0.05, 0.95]
    for each label independently, then refines with local search.
    
    For multi-label, each label threshold is optimized independently
    since they are independent binary classifications.
    """
    
    def __init__(
        self,
        n_classes: int = 28,
        metric: str = "macro_f1_micro_avg",
        n_trials: int = 100,
        label_names: List[str] = GOEMOTIONS_28,
    ):
        self.n_classes = n_classes
        self.metric = metric
        self.n_trials = n_trials
        self.label_names = label_names[:n_classes]
        self.best_thresholds: Optional[np.ndarray] = None
        self.best_score: float = 0.0
        self.optimization_history: List[Dict] = []
    
    def optimize(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        method: str = "per_label",
    ) -> np.ndarray:
        """
        Find optimal per-label thresholds.
        
        Two methods:
        1. "per_label": Optimize each label independently (recommended)
        2. "global": Random search over all thresholds simultaneously
        
        Args:
            probs: Sigmoid probabilities [num_samples, num_classes]
            labels: Ground truth multi-hot [num_samples, num_classes]
            method: Optimization method
        
        Returns:
            Optimal thresholds array [num_classes]
        """
        n_samples, n_classes = probs.shape
        assert n_classes == self.n_classes, f"Expected {self.n_classes} classes, got {n_classes}"
        
        if method == "per_label":
            return self._optimize_per_label(probs, labels)
        else:
            return self._optimize_global(probs, labels)
    
    def _optimize_per_label(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Optimize each label's threshold independently.
        More efficient for multi-label since labels are independent.
        """
        best_thresholds = np.ones(self.n_classes) * 0.5
        
        for i in range(self.n_classes):
            label_probs = probs[:, i]
            label_true = labels[:, i]
            
            best_f1 = 0.0
            best_thr = 0.5
            
            # Coarse-to-fine search
            for threshold in np.arange(0.05, 0.96, 0.05):
                preds = (label_probs >= threshold).astype(int)
                tp = np.sum((preds == 1) & (label_true == 1))
                fp = np.sum((preds == 1) & (label_true == 0))
                fn = np.sum((preds == 0) & (label_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = threshold
            
            # Fine-tune around best threshold
            for threshold in np.arange(max(0.01, best_thr - 0.04), min(0.99, best_thr + 0.05), 0.01):
                preds = (label_probs >= threshold).astype(int)
                tp = np.sum((preds == 1) & (label_true == 1))
                fp = np.sum((preds == 1) & (label_true == 0))
                fn = np.sum((preds == 0) & (label_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = threshold
            
            best_thresholds[i] = best_thr
        
        # Evaluate overall with these thresholds
        all_preds = (probs >= best_thresholds[np.newaxis, :]).astype(int)
        metrics = compute_multi_label_metrics(labels, all_preds, self.label_names)
        self.best_score = getattr(metrics, self.metric, metrics.macro_f1_micro_avg)
        self.best_thresholds = best_thresholds
        
        logger.info(f"Per-label threshold optimization complete:")
        logger.info(f"  Best {self.metric}: {self.best_score:.4f}")
        for i, label in enumerate(self.label_names):
            logger.info(f"  {label:20s}: threshold={best_thresholds[i]:.4f}")
        
        return best_thresholds
    
    def _optimize_global(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Global random search over all thresholds.
        More comprehensive but slower.
        """
        best_thresholds = np.ones(self.n_classes) * 0.5
        best_score = 0.0
        
        # Phase 1: Random exploration
        for trial in range(self.n_trials):
            # Sample random thresholds (skewed toward 0.5 for stability)
            thresholds = np.random.beta(a=5, b=5, size=self.n_classes)
            
            predictions = (probs >= thresholds[np.newaxis, :]).astype(int)
            metrics = compute_multi_label_metrics(labels, predictions, self.label_names)
            score = getattr(metrics, self.metric, metrics.macro_f1_micro_avg)
            
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
            
            self.optimization_history.append({
                "trial": trial,
                "score": float(score),
                "thresholds": thresholds.tolist(),
            })
        
        # Phase 2: Local refinement around best thresholds
        for i in range(self.n_classes):
            local_best = best_thresholds[i]
            for delta in np.arange(-0.05, 0.06, 0.01):
                candidate = np.clip(local_best + delta, 0.01, 0.99)
                thresholds = best_thresholds.copy()
                thresholds[i] = candidate
                
                predictions = (probs >= thresholds[np.newaxis, :]).astype(int)
                metrics = compute_multi_label_metrics(labels, predictions, self.label_names)
                score = getattr(metrics, self.metric, metrics.macro_f1_micro_avg)
                
                if score > best_score:
                    best_score = score
                    best_thresholds[i] = candidate
        
        self.best_thresholds = best_thresholds
        self.best_score = best_score
        
        logger.info(f"Global threshold optimization complete: best {self.metric}={best_score:.4f}")
        for i, label in enumerate(self.label_names):
            logger.info(f"  {label:20s}: threshold={best_thresholds[i]:.4f}")
        
        return best_thresholds
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Make predictions using optimized thresholds.
        
        Args:
            probs: Sigmoid probabilities [num_samples, num_classes]
        
        Returns:
            Binary predictions [num_samples, num_classes]
        """
        if self.best_thresholds is None:
            logger.warning("No optimized thresholds. Using default 0.5.")
            return (probs >= 0.5).astype(int)
        return (probs >= self.best_thresholds[np.newaxis, :]).astype(int)
    
    def apply_and_evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, EmotionMetrics]:
        """
        Apply optimized thresholds and return predictions and metrics.
        
        Args:
            probs: Sigmoid probabilities [num_samples, num_classes]
            labels: Ground truth [num_samples, num_classes]
        
        Returns:
            Tuple of (predictions, metrics)
        """
        if self.best_thresholds is None:
            self.optimize(probs, labels)
        
        predictions = self.predict(probs)
        metrics = compute_multi_label_metrics(labels, predictions, self.label_names, probs=probs)
        
        return predictions, metrics
    
    def get_threshold_dict(self) -> Dict[str, float]:
        """Get thresholds as {label: threshold} dict."""
        if self.best_thresholds is None:
            return {}
        return {
            label: float(self.best_thresholds[i])
            for i, label in enumerate(self.label_names)
        }
    
    def save(self, path: str):
        """Save thresholds to JSON."""
        data = {
            "thresholds": self.best_thresholds.tolist() if self.best_thresholds is not None else None,
            "labels": self.label_names,
            "best_score": self.best_score,
            "metric": self.metric,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ThresholdOptimizer":
        """Load thresholds from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        optimizer = cls(
            n_classes=len(data["labels"]),
            metric=data.get("metric", "macro_f1_micro_avg"),
            label_names=data["labels"],
        )
        if data["thresholds"] is not None:
            optimizer.best_thresholds = np.array(data["thresholds"])
            optimizer.best_score = data["best_score"]
        
        return optimizer
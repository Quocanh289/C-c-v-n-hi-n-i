"""
Threshold Optimizer Module
===========================
Finds optimal probability thresholds for each emotion class
to maximize macro F1 score on the validation set.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import COARSE_EMOTIONS
from .metrics import EmotionMetrics, compute_class_metrics

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes per-class probability thresholds to maximize macro F1.
    
    In multi-class classification with softmax, the default threshold
    is simply argmax. However, optimizing per-class thresholds can
    improve performance, especially for imbalanced datasets.
    """
    
    def __init__(
        self,
        n_classes: int = 9,
        metric: str = "macro_f1",
        n_trials: int = 100,
        label_names: List[str] = COARSE_EMOTIONS,
    ):
        self.n_classes = n_classes
        self.metric = metric
        self.n_trials = n_trials
        self.label_names = label_names
        self.best_thresholds: Optional[np.ndarray] = None
        self.best_score: float = 0.0
    
    def optimize(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Find optimal per-class thresholds using random search.
        
        Args:
            probs: Softmax probabilities [num_samples, num_classes]
            labels: Ground truth indices [num_samples]
        
        Returns:
            Optimal thresholds array [num_classes]
        """
        n_samples, n_classes = probs.shape
        best_thresholds = np.ones(n_classes) * (1.0 / n_classes)
        best_score = 0.0
        
        # Random search over threshold combinations
        for trial in range(self.n_trials):
            # Generate random thresholds
            thresholds = np.random.dirichlet(np.ones(n_classes))
            
            # Make predictions using thresholds
            predictions = self._threshold_predict(probs, thresholds)
            
            # Compute metrics
            metrics = compute_class_metrics(labels, predictions, self.label_names)
            score = getattr(metrics, self.metric, metrics.macro_f1)
            
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
        
        self.best_thresholds = best_thresholds
        self.best_score = best_score
        
        logger.info(f"Threshold optimization complete: best {self.metric}={best_score:.4f}")
        for i, label in enumerate(self.label_names):
            logger.info(f"  {label}: threshold={best_thresholds[i]:.4f}")
        
        return best_thresholds
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Make predictions using optimized thresholds.
        
        Args:
            probs: Softmax probabilities [num_samples, num_classes]
        
        Returns:
            Predicted class indices [num_samples]
        """
        if self.best_thresholds is None:
            return np.argmax(probs, axis=1)
        return self._threshold_predict(probs, self.best_thresholds)
    
    def _threshold_predict(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Make predictions by comparing probability ratios to thresholds.
        """
        # Normalize thresholds to sum to 1
        thresholds = thresholds / thresholds.sum()
        
        # Compute decision scores: prob - threshold for each class
        scores = probs - thresholds[np.newaxis, :]
        
        # Predict the class with highest score
        return np.argmax(scores, axis=1)
    
    def apply_and_evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, EmotionMetrics]:
        """
        Apply optimized thresholds and return predictions and metrics.
        
        Args:
            probs: Softmax probabilities [num_samples, num_classes]
            labels: Ground truth indices [num_samples]
        
        Returns:
            Tuple of (predictions, metrics)
        """
        if self.best_thresholds is None:
            self.optimize(probs, labels)
        
        predictions = self.predict(probs)
        metrics = compute_class_metrics(labels, predictions, self.label_names)
        
        return predictions, metrics
"""
Threshold Optimizer
====================
Optimizes per-class decision thresholds for mental health classification.

For multi-class classification, the default threshold is argmax (threshold=0.5).
However, per-class thresholds can improve F1 by adjusting the confidence
required for each class prediction.

This is particularly useful for mental health detection because:
- Some classes (Suicidal, Depression) should have LOWER thresholds
  (better to flag and be wrong than miss a real case)
- Other classes (Normal) can have HIGHER thresholds
  (better to be confident before labeling as non-clinical)

Uses Optuna for hyperparameter optimization of thresholds.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .config import MENTAL_HEALTH_LABELS, IDX_TO_MH_LABELS, TrainingConfig
from .model import MentalHealthClassifier
from .metrics import MentalHealthMetrics

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimize per-class decision thresholds for mental health classification.
    
    Strategy:
    - Use Optuna to search for optimal thresholds per class
    - Objective: maximize macro F1 on validation set
    - Thresholds are searched in [0.1, 0.9] range
    - Uses model probabilities directly
    
    For production:
    - Lower thresholds for critical conditions (Suicidal, Depression)
    - Higher thresholds for Normal class
    - Balanced thresholds for ambiguous classes (Bipolar vs Personality_disorder)
    """
    
    def __init__(
        self,
        n_trials: int = 50,
        metric: str = "macro_f1",
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.metric = metric
        self.random_state = random_state
        self.best_thresholds = None
        self.best_score = -1.0
        
    def optimize(
        self,
        model: MentalHealthClassifier,
        val_loader: DataLoader,
        num_classes: int = 7,
    ) -> Tuple[np.ndarray, float]:
        """
        Run Optuna optimization for per-class thresholds.
        
        Args:
            model: Trained model
            val_loader: Validation data loader
            num_classes: Number of classes
            
        Returns:
            (best_thresholds, best_score)
        """
        try:
            import optuna
        except ImportError:
            logger.warning(
                "Optuna not installed. Using default thresholds (0.5). "
                "Install with: pip install optuna"
            )
            return np.ones(num_classes) * 0.5, 0.0
        
        device = next(model.parameters()).device
        model.eval()
        
        # Collect all predictions and labels
        all_probs = []
        all_labels = []
        
        logger.info("Collecting validation probabilities for threshold optimization...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                
                outputs = model(input_ids, attention_mask)
                probs = outputs["probs"].cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Collected {len(all_labels)} validation samples")
        
        # Define Optuna objective
        def objective(trial):
            thresholds = np.zeros(num_classes)
            for i in range(num_classes):
                # Suggest threshold for each class
                thresholds[i] = trial.suggest_float(f"threshold_{i}", 0.1, 0.9)
            
            # Apply thresholds to get predictions
            # For each sample, predict the class with highest probability
            # that exceeds its threshold
            predictions = self._apply_thresholds(all_probs, thresholds)
            
            # Compute metric
            if self.metric == "macro_f1":
                score = f1_score(all_labels, predictions, average="macro", zero_division=0)
            elif self.metric == "weighted_f1":
                score = f1_score(all_labels, predictions, average="weighted", zero_division=0)
            else:
                score = f1_score(all_labels, predictions, average="macro", zero_division=0)
            
            return score
        
        # Run optimization
        logger.info(f"Running Optuna with {self.n_trials} trials...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best results
        self.best_score = study.best_value
        self.best_thresholds = np.zeros(num_classes)
        for i in range(num_classes):
            self.best_thresholds[i] = study.best_params[f"threshold_{i}"]
        
        logger.info(f"Best {self.metric}: {self.best_score:.4f}")
        logger.info("Best thresholds per class:")
        for i, label in enumerate(MENTAL_HEALTH_LABELS):
            logger.info(f"  {label:25s}: {self.best_thresholds[i]:.3f}")
        
        return self.best_thresholds, self.best_score
    
    def _apply_thresholds(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Apply per-class thresholds to get predictions.
        
        For each sample:
        1. Find classes where probability >= threshold
        2. If multiple classes pass threshold, take highest probability
        3. If no class passes threshold, take argmax (highest probability)
        """
        predictions = np.zeros(len(probs), dtype=int)
        
        for i in range(len(probs)):
            # Find classes that pass threshold
            above_threshold = np.where(probs[i] >= thresholds)[0]
            
            if len(above_threshold) > 0:
                # Among passing classes, pick the one with highest probability
                above_probs = probs[i][above_threshold]
                predictions[i] = above_threshold[np.argmax(above_probs)]
            else:
                # If none pass, use argmax
                predictions[i] = np.argmax(probs[i])
        
        return predictions
    
    def save_thresholds(self, save_path: str):
        """Save optimized thresholds to JSON."""
        if self.best_thresholds is None:
            logger.warning("No thresholds to save. Run optimize() first.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        threshold_data = {
            "thresholds": self.best_thresholds.tolist(),
            "labels": MENTAL_HEALTH_LABELS,
            "per_class": {
                label: float(self.best_thresholds[i])
                for i, label in enumerate(MENTAL_HEALTH_LABELS)
            },
            "best_score": self.best_score,
            "metric": self.metric,
        }
        
        with open(save_path, "w") as f:
            json.dump(threshold_data, f, indent=2)
        
        logger.info(f"Thresholds saved to {save_path}")
    
    @staticmethod
    def load_thresholds(load_path: str) -> Optional[np.ndarray]:
        """Load thresholds from JSON."""
        if not os.path.exists(load_path):
            return None
        
        with open(load_path) as f:
            data = json.load(f)
        
        return np.array(data["thresholds"])
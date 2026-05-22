"""
Pipeline Orchestrator
======================
End-to-end pipeline for training the mental health classifier.

Orchestrates:
1. Data loading, cleaning, and splitting
2. Tokenizer loading
3. DataLoader creation
4. Model building (DeBERTa-v3 + LoRA)
5. Training with mixed precision + gradient accumulation
6. Evaluation with comprehensive metrics
7. Threshold optimization
8. Model export and saving
9. Error analysis and reporting
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

from .config import (
    MENTAL_HEALTH_LABELS,
    TrainingConfig,
    MHConfig,
)
from .preprocessor import RedditTextPreprocessor, analyze_text_quality
from .dataset import create_dataloaders, get_tokenizer
from .model import MentalHealthClassifier, build_model
from .trainer import MentalHealthTrainer
from .metrics import MentalHealthMetrics, compute_metrics
from .threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


class MentalHealthPipeline:
    """
    Full training pipeline for mental health classification.
    
    Usage:
        config = TrainingConfig()
        mh_config = MHConfig()
        pipeline = MentalHealthPipeline(config, mh_config)
        results = pipeline.run()
        
        # For inference later:
        results = pipeline.predict("I feel so hopeless and tired all the time")
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        mh_config: Optional[MHConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.mh_config = mh_config or MHConfig()
        
        # Set up logging
        self._setup_logging()
        
        # Components (initialized during run)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_df = None
        
        logger.info("MentalHealthPipeline initialized")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Max seq length: {self.mh_config.max_length}")
        logger.info(f"Loss: {self.config.loss_type}")
        logger.info(f"Mixed precision: {self.config.mixed_precision}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
    
    def _setup_logging(self):
        """Configure logging with file and console handlers."""
        log_dir = self.config.get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir,
            f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        
        logger.info(f"Logging to {log_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full training pipeline.
        
        Steps:
        1. Load and preprocess data
        2. Create tokenizer
        3. Create dataloaders
        4. Build model
        5. Train
        6. Evaluate on test set
        7. Optimize thresholds
        8. Save model
        9. Generate report
        
        Returns:
            Dict with training results, metrics, and model path
        """
        logger.info("=" * 60)
        logger.info("MENTAL HEALTH TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load tokenizer
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Loading tokenizer")
        logger.info("=" * 40)
        self.tokenizer = get_tokenizer(self.config)
        
        # Step 2: Create dataloaders
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Creating dataloaders")
        logger.info("=" * 40)
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config, self.mh_config, self.tokenizer
        )
        
        # Step 3: Build model
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: Building model")
        logger.info("=" * 40)
        self.model = build_model(self.config)
        
        # Step 4: Train
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: Training")
        logger.info("=" * 40)
        self.trainer = MentalHealthTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
        )
        
        training_results = self.trainer.train()
        
        # Step 5: Evaluate on test set
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: Test set evaluation")
        logger.info("=" * 40)
        test_results = self.trainer.test(self.test_loader)
        
        # Step 6: Optimize thresholds
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: Threshold optimization")
        logger.info("=" * 40)
        threshold_optimizer = ThresholdOptimizer(
            n_trials=self.config.threshold_n_trials,
            metric=self.config.threshold_optimization_metric,
        )
        best_thresholds, best_threshold_score = threshold_optimizer.optimize(
            self.model, self.val_loader
        )
        
        # Save thresholds
        threshold_path = os.path.join(
            self.config.get_checkpoint_dir(), "best_model", "thresholds.json"
        )
        threshold_optimizer.save_thresholds(threshold_path)
        
        # Step 7: Save final model
        logger.info("\n" + "=" * 40)
        logger.info("STEP 7: Saving final model")
        logger.info("=" * 40)
        final_model_path = self.config.get_output_model_dir()
        self.model.save_pretrained(final_model_path)
        
        # Also save to checkpoint best_model directory
        best_model_path = os.path.join(
            self.config.get_checkpoint_dir(), "best_model"
        )
        # If best model was saved during training, it's already there
        
        # Save config
        config_path = os.path.join(self.config.get_checkpoint_dir(), "training_config.json")
        self.config.save(config_path)
        
        logger.info(f"Model saved to: {final_model_path}")
        
        # Step 8: Generate final report
        logger.info("\n" + "=" * 60)
        logger.info("FINAL REPORT")
        logger.info("=" * 60)
        
        results = {
            "model_config": {
                "model_name": self.config.model_name,
                "num_labels": self.config.num_labels,
                "max_seq_length": self.mh_config.max_length,
                "loss_type": self.config.loss_type,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
                "use_lora": self.config.use_lora,
                "lora_r": self.config.lora_r,
                "mixed_precision": self.config.mixed_precision,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "scheduler": self.config.scheduler,
                "warmup_ratio": self.config.warmup_ratio,
            },
            "training": {
                "best_epoch": training_results["best_epoch"],
                "best_metric": training_results["best_metric"],
                "total_time_seconds": training_results["total_time_seconds"],
                "total_epochs_completed": training_results["total_epochs_completed"],
                "early_stopped": training_results["early_stopped"],
            },
            "test_metrics": test_results,
            "thresholds": {
                "values": best_thresholds.tolist() if isinstance(best_thresholds, np.ndarray) else best_thresholds,
                "per_class": {
                    label: float(best_thresholds[i])
                    for i, label in enumerate(MENTAL_HEALTH_LABELS)
                } if isinstance(best_thresholds, np.ndarray) else {},
                "best_macro_f1": best_threshold_score,
            },
            "model_paths": {
                "checkpoint_dir": self.config.get_checkpoint_dir(),
                "output_model_dir": final_model_path,
            },
        }
        
        # Save comprehensive results
        results_path = os.path.join(
            self.config.get_log_dir(),
            f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Data size (train/val/test): "
                    f"{len(self.train_loader.dataset)}/"
                    f"{len(self.val_loader.dataset)}/"
                    f"{len(self.test_loader.dataset)}")
        logger.info(f"Training time: {training_results['total_time_seconds']:.1f}s")
        logger.info(f"Best epoch: {training_results['best_epoch'] + 1}")
        logger.info(f"Best macro F1 (val): {training_results['best_metric']:.4f}")
        logger.info(f"Best thresholds macro F1: {best_threshold_score:.4f}")
        logger.info(f"\n--- Test Set ---")
        logger.info(f"Accuracy: {test_results.get('accuracy', 0.0):.4f}")
        logger.info(f"Macro F1: {test_results.get('macro_f1', 0.0):.4f}")
        logger.info(f"Weighted F1: {test_results.get('weighted_f1', 0.0):.4f}")
        logger.info(f"MCC: {test_results.get('mcc', 0.0):.4f}")
        
        logger.info("\n--- Per-class F1 ---")
        per_class = test_results.get("per_class", {})
        for label in MENTAL_HEALTH_LABELS:
            if label in per_class:
                m = per_class[label]
                logger.info(f"  {label:25s}: F1={m['f1']:.4f} (n={m['support']})")
        
        logger.info("\n--- Hardest Classes ---")
        for hc in test_results.get("hardest_classes", []):
            logger.info(f"  {hc['label']:25s}: F1={hc['f1']:.4f} (support={hc['support']})")
        
        logger.info("\n--- Top Misclassifications ---")
        for mc in test_results.get("misclassifications", [])[:5]:
            logger.info(f"  True={mc['true_label']:25s} → Pred={mc['predicted_label']:25s}: {mc['count']} ({mc['percentage']:.1f}%)")
        
        logger.info("=" * 60)
        
        return results
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict mental health condition for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Run pipeline.run() first.")
        
        self.model.eval()
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.mh_config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        
        # Predict
        outputs = self.model(input_ids, attention_mask)
        probs = outputs["probs"].cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        # Get top 3
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "label": MENTAL_HEALTH_LABELS[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in top3_indices
        ]
        
        # Load thresholds if available
        threshold_path = os.path.join(
            self.config.get_checkpoint_dir(), "best_model", "thresholds.json"
        )
        thresholds = ThresholdOptimizer.load_thresholds(threshold_path)
        
        result = {
            "text": text[:200],
            "primary_condition": MENTAL_HEALTH_LABELS[prediction],
            "primary_confidence": confidence,
            "all_scores": {
                label: float(probs[i])
                for i, label in enumerate(MENTAL_HEALTH_LABELS)
            },
            "top_predictions": top3,
            "needs_attention": MENTAL_HEALTH_LABELS[prediction] != "Normal" and confidence >= 0.3,
            "num_labels": len(MENTAL_HEALTH_LABELS),
            "thresholds": thresholds.tolist() if thresholds is not None else None,
        }
        
        return result
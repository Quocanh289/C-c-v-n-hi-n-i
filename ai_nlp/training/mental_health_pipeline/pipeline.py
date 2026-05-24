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

Ensemble mode:
- Trains 3 models independently (sequentially): DeBERTa-v3, XLM-R, PhoBERT
- Each model uses the same data but different architectures
- Inference uses weighted averaging of all 3 softmax outputs
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

from .config import (
    MENTAL_HEALTH_LABELS,
    TrainingConfig,
    MHConfig,
    ENSEMBLE_MODELS,
)
from .preprocessor import RedditTextPreprocessor, analyze_text_quality
from .dataset import create_dataloaders, get_tokenizer
from .model import (
    MentalHealthClassifier,
    EnsembleMentalHealthClassifier,
    build_model,
    load_trained_model,
)
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
        
        # Ensemble training:
        config.use_ensemble = True
        results = pipeline.run_ensemble()
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
        logger.info(f"Ensemble mode: {self.config.use_ensemble}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Max seq length: {self.mh_config.max_length}")
    
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
        Run the full training pipeline (single model mode).
        
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
        if self.config.use_ensemble:
            return self.run_ensemble()
        
        logger.info("=" * 60)
        logger.info("MENTAL HEALTH TRAINING PIPELINE (Single Model)")
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
        
        # Save config
        config_path = os.path.join(self.config.get_checkpoint_dir(), "training_config.json")
        self.config.save(config_path)
        
        logger.info(f"Model saved to: {final_model_path}")
        
        # Step 8: Generate final report
        results = self._generate_results(training_results, test_results, best_thresholds, best_threshold_score, final_model_path)
        
        return results
    
    def run_ensemble(self) -> Dict[str, Any]:
        """
        Run ensemble training: train 3 models independently (sequentially).
        
        Models: DeBERTa-v3, XLM-RoBERTa, PhoBERT
        Each model is trained from scratch on the same data split.
        After training, all 3 are saved separately for inference.
        
        Returns:
            Dict with per-model results and ensemble summary
        """
        logger.info("=" * 60)
        logger.info("MENTAL HEALTH TRAINING PIPELINE (Ensemble)")
        logger.info(f"Models: {self.config.ensemble_model_keys}")
        logger.info("=" * 60)
        
        model_keys = self.config.ensemble_model_keys
        all_results = {}
        
        # Train each model independently
        for model_key in model_keys:
            model_info = ENSEMBLE_MODELS[model_key]
            
            logger.info("\n" + "=" * 50)
            logger.info(f"TRAINING MODEL: {model_key} ({model_info['name']})")
            logger.info("=" * 50)
            
            # Create model-specific config
            model_config = TrainingConfig()
            # Copy common settings
            for field_name in self.config.__dataclass_fields__:
                if hasattr(self.config, field_name):
                    setattr(model_config, field_name, getattr(self.config, field_name))
            
            # Override with model-specific settings
            model_config.model_name = model_info["name"]
            model_config.lora_target_modules = model_info["lora_target_modules"]
            model_config.learning_rate = model_info["learning_rate"]
            model_config.batch_size = model_info["batch_size"]
            model_config.checkpoint_dir = self.config.checkpoint_dir
            
            # Create model-specific tokenizer
            logger.info(f"Loading tokenizer for {model_key}...")
            tokenizer = get_tokenizer(model_config)
            
            # Create dataloaders (same data split)
            logger.info(f"Creating dataloaders for {model_key}...")
            train_loader, val_loader, test_loader = create_dataloaders(
                model_config, self.mh_config, tokenizer
            )
            
            # Build model
            logger.info(f"Building model {model_key}...")
            model = build_model(model_config)
            
            # Train
            logger.info(f"Training {model_key}...")
            trainer = MentalHealthTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=model_config,
            )
            training_results = trainer.train()
            
            # Test
            logger.info(f"Testing {model_key}...")
            test_results = trainer.test(test_loader)
            
            # Save model to its checkpoint dir
            checkpoint_dir = model_config.get_ensemble_checkpoint_dir(model_key)
            model.save_pretrained(checkpoint_dir)
            
            all_results[model_key] = {
                "model_name": model_info["name"],
                "weight": model_info["weight"],
                "training": training_results,
                "test_metrics": test_results,
                "checkpoint_dir": checkpoint_dir,
            }
            
            logger.info(f"  ✓ {model_key} complete! Macro F1: {test_results.get('macro_f1', 0.0):.4f}")
            
            # Clear GPU cache before next model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute ensemble metrics (simulated: average of per-model metrics)
        logger.info("\n" + "=" * 50)
        logger.info("ENSEMBLE RESULTS SUMMARY")
        logger.info("=" * 50)
        
        ensemble_results = {}
        for model_key in model_keys:
            r = all_results[model_key]
            macro_f1 = r["test_metrics"].get("macro_f1", 0.0)
            accuracy = r["test_metrics"].get("accuracy", 0.0)
            logger.info(f"  {model_key:12s} ({ENSEMBLE_MODELS[model_key]['name']:30s}): "
                       f"Macro F1={macro_f1:.4f}, Acc={accuracy:.4f}, "
                       f"Weight={ENSEMBLE_MODELS[model_key]['weight']}")
        
        # Save ensemble config
        ensemble_config = {
            "models": {
                key: {
                    "name": ENSEMBLE_MODELS[key]["name"],
                    "weight": ENSEMBLE_MODELS[key]["weight"],
                    "checkpoint_dir": all_results[key]["checkpoint_dir"],
                }
                for key in model_keys
            },
            "num_labels": self.config.num_labels,
            "labels": MENTAL_HEALTH_LABELS,
            "trained_at": time.strftime('%Y%m%d_%H%M%S'),
        }
        ensemble_dir = self.config.get_ensemble_output_dir()
        os.makedirs(ensemble_dir, exist_ok=True)
        with open(os.path.join(ensemble_dir, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info(f"Ensemble config saved to {ensemble_dir}")
        
        return {
            "ensemble": True,
            "model_keys": model_keys,
            "per_model": all_results,
            "ensemble_config_path": os.path.join(ensemble_dir, "ensemble_config.json"),
        }
    
    def _generate_results(self, training_results, test_results, best_thresholds, best_threshold_score, final_model_path):
        """Generate final results dict."""
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
                "was_best_updated": training_results.get("was_best_updated", False),
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
        logger.info(f"Training time: {training_results['total_time_seconds']:.1f}s")
        logger.info(f"Best epoch: {training_results['best_epoch'] + 1}")
        logger.info(f"Best macro F1 (val): {training_results['best_metric']:.4f}")
        logger.info(f"\n--- Test Set ---")
        self._log_test_summary(test_results)
        
        return results
    
    def _log_test_summary(self, test_results):
        """Log test results summary."""
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
            logger.info(f"  True={mc['true_label']:25s} -> Pred={mc['predicted_label']:25s}: {mc['count']} ({mc['percentage']:.1f}%)")
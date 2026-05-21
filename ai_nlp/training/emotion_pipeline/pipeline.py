"""
Pipeline Orchestrator
=====================
High-level pipeline that orchestrates the complete multi-label training workflow:
1. Dataset download/preparation with Reddit/internet preprocessing
2. Model & tokenizer initialization (XLM-RoBERTa or RoBERTa)
3. Multi-label training with validation and threshold optimization
4. Evaluation on test set with comprehensive multi-label metrics
5. Overfitting, leakage, and distribution shift detection
6. Model export for deployment
7. Optional model comparison (RoBERTa vs XLM-RoBERTa)
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from .config import TrainingConfig, GoEmotionsConfig
from .dataset import GoEmotionsDataset, get_kaggle_goemotions, MultiLabelDataCollator
from .model import GoEmotionsModel, get_model_and_tokenizer, compare_models
from .trainer import Trainer, TrainingResult
from .gpu_utils import get_device, optimize_memory
from .metrics import (
    EmotionMetrics,
    compute_all_metrics,
    detect_distribution_shift,
)

logger = logging.getLogger(__name__)


def setup_logging(config: TrainingConfig):
    """Configure logging for the pipeline."""
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return log_file


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def create_data_loaders(
    config: TrainingConfig,
    tokenizer,
    goemotions_config: Optional[GoEmotionsConfig] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Create train, validation, and test data loaders for multi-label GoEmotions.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_eval_loader,
                  class_weights_28, class_weights_9)
    """
    go_config = goemotions_config or GoEmotionsConfig()

    # Download or locate dataset
    data_paths = get_kaggle_goemotions(config)

    # Create datasets
    train_dataset = GoEmotionsDataset(
        file_path=data_paths["train"],
        tokenizer=tokenizer,
        config=config,
        goemotions_config=go_config,
        split="train",
        augment=config.use_augmentation,
        max_samples=go_config.max_train_samples,
    )

    val_dataset = GoEmotionsDataset(
        file_path=data_paths["val"],
        tokenizer=tokenizer,
        config=config,
        goemotions_config=go_config,
        split="val",
        augment=False,
        max_samples=go_config.max_val_samples,
    )

    test_dataset = None
    if os.path.exists(data_paths.get("test", "")):
        test_dataset = GoEmotionsDataset(
            file_path=data_paths["test"],
            tokenizer=tokenizer,
            config=config,
            goemotions_config=go_config,
            split="test",
            augment=False,
            max_samples=go_config.max_test_samples,
        )

    # Create train evaluation subset (for overfitting detection)
    train_eval_dataset = None
    if len(train_dataset) > 1000:
        # Use a small subset of training data for train-set metrics
        subset_indices = list(range(min(500, len(train_dataset))))
        train_eval_dataset = Subset(train_dataset, subset_indices)

    # Get class weights
    class_weights_28 = train_dataset.class_weights_28
    class_weights_9 = train_dataset.class_weights_9

    # Create data collator
    collator = MultiLabelDataCollator()

    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Test loader
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Train eval loader
    train_eval_loader = None
    if train_eval_dataset:
        train_eval_loader = DataLoader(
            train_eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    logger.info(
        f"Train: {len(train_dataset)} samples | "
        f"Val: {len(val_dataset)} samples"
    )
    if test_dataset:
        logger.info(f"Test: {len(test_dataset)} samples")

    return (
        train_loader,
        val_loader,
        test_loader,
        train_eval_loader,
        class_weights_28,
        class_weights_9,
    )


def train_pipeline(
    config: Optional[TrainingConfig] = None,
    goemotions_config: Optional[GoEmotionsConfig] = None,
    model_path: Optional[str] = None,
    compare_models_flag: bool = False,
) -> TrainingResult:
    """
    Main multi-label training pipeline.

    Orchestrates:
    1. Setup logging and seeding
    2. Dataset preparation with Reddit/internet preprocessing
    3. Model initialization (XLM-RoBERTa or RoBERTa with optional LoRA)
    4. Multi-label training with validation
    5. Threshold optimization
    6. Test evaluation with overfitting/leakage detection
    7. Model export
    8. Optional RoBERTa vs XLM-RoBERTa comparison

    Args:
        config: Training configuration
        goemotions_config: GoEmotions dataset config
        model_path: Path to resume training from
        compare_models_flag: If True, run model comparison after training

    Returns:
        TrainingResult with best metrics and analysis
    """
    if config is None:
        config = TrainingConfig()

    # Setup
    log_file = setup_logging(config)
    set_seed(config.seed)

    logger.info("=" * 70)
    logger.info("EMOTION LENS - GoEmotions Multi-Label Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Task: {config.task_type} ({config.num_labels} labels)")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Loss: {config.loss_type} (primary={config.primary_loss}, secondary={config.secondary_loss})")
    logger.info(f"  LoRA: {config.use_lora} (r={config.lora_r})")
    logger.info(f"  Precision: {config.mixed_precision}")
    logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    logger.info(f"  Balanced sampling: {config.use_balanced_sampling}")

    # Detect hardware
    device = get_device()
    optimize_memory(device)

    # Create output directories
    os.makedirs(config.get_checkpoint_dir(), exist_ok=True)
    os.makedirs(config.get_output_model_dir(), exist_ok=True)

    # Save training config
    config_path = os.path.join(config.get_checkpoint_dir(), "training_config.json")
    config.save(config_path)
    logger.info(f"Training config saved to {config_path}")

    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config=config, load_path=model_path)

    # Create data loaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_eval_loader,
        class_weights_28,
        class_weights_9,
    ) = create_data_loaders(config=config, tokenizer=tokenizer, goemotions_config=goemotions_config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        class_weights=class_weights_28,
        coarse_class_weights=class_weights_9,
    )

    # Train
    logger.info("=" * 70)
    logger.info("Starting multi-label training...")
    logger.info("=" * 70)

    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_eval_loader=train_eval_loader,
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best epoch: {result.best_epoch}")
    logger.info(f"Training time: {result.training_time:.1f}s ({result.training_time/60:.1f} min)")
    
    if result.best_metrics:
        logger.info("Best metrics:")
        for metric_name, value in sorted(result.best_metrics.items()):
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value:.4f}")
    
    if result.overfitting_analysis:
        logger.info(f"Overfitting risk: {result.overfitting_analysis.get('overfitting_risk', 'N/A')}")
    
    if result.data_leakage_analysis:
        logger.info(f"Data leakage risk: {result.data_leakage_analysis.get('data_leakage_risk', 'N/A')}")

    # Optional model comparison
    if compare_models_flag and test_loader is not None:
        logger.info("=" * 70)
        logger.info("Running RoBERTa vs XLM-RoBERTa comparison...")
        logger.info("=" * 70)

        def metric_fn(model, loader):
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids, attention_mask)
                    all_logits.append(outputs["logits"].cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            
            all_logits = np.concatenate(all_logits, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            metrics = compute_all_metrics(all_logits, all_labels)
            return metrics.to_dict()

        comp_result = compare_models(config, metric_fn, test_loader, device)
        comp_result.print_summary()
        comp_result.save(os.path.join(config.get_checkpoint_dir(), "model_comparison.json"))

    return result


def evaluate_model(
    model_path: str,
    data_path: str,
    config: Optional[TrainingConfig] = None,
) -> EmotionMetrics:
    """
    Evaluate a trained model on a dataset with multi-label metrics.

    Args:
        model_path: Path to saved model
        data_path: Path to evaluation data CSV
        config: Training configuration

    Returns:
        EmotionMetrics from evaluation
    """
    if config is None:
        config = TrainingConfig()

    device = get_device()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config=config, load_path=model_path)
    model.to(device)
    model.eval()

    # Create dataset
    dataset = GoEmotionsDataset(
        file_path=data_path,
        tokenizer=tokenizer,
        config=config,
        split="eval",
        augment=False,
    )

    collator = MultiLabelDataCollator()
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
    )

    # Evaluate
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_all_metrics(all_logits, all_labels)

    logger.info(f"Evaluation on {data_path}:")
    logger.info(f"  Macro F1: {metrics.macro_f1_micro_avg:.4f}")
    logger.info(f"  Micro F1: {metrics.micro_f1:.4f}")
    logger.info(f"  Hamming Loss: {metrics.hamming_loss:.4f}")
    logger.info(f"  Subset Acc: {metrics.subsets_accuracy:.4f}")
    logger.info(f"  ECE: {metrics.expected_calibration_error:.4f}")

    return metrics
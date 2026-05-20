"""
Pipeline Orchestrator
======================
High-level pipeline that orchestrates the complete training workflow:
1. Dataset download/preparation
2. Model & tokenizer initialization
3. Training with validation
4. Evaluation on test set
5. Model export and deployment
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
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler

from .config import TrainingConfig, GoEmotionsConfig, COARSE_EMOTIONS
from .dataset import (
    GoEmotionsDataset,
    get_kaggle_goemotions,
    EmotionLabelMapper,
)
from .model import GoEmotionsModel, get_model_and_tokenizer
from .trainer import Trainer, TrainingResult
from .gpu_utils import get_device, optimize_memory
from .metrics import EmotionMetrics, compute_class_metrics, compute_all_metrics
from .threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


def setup_logging(config: TrainingConfig):
    """Configure logging for the pipeline."""
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Training configuration
        tokenizer: HuggingFace tokenizer
        goemotions_config: GoEmotions-specific configuration
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
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
    
    # Create test dataset if available
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
    
    # Log class distributions
    logger.info("Training class distribution:")
    for cls, count in sorted(train_dataset.get_class_distribution().items()):
        logger.info(f"  {cls}: {count}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    if test_dataset:
        logger.info(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def train_pipeline(
    config: Optional[TrainingConfig] = None,
    goemotions_config: Optional[GoEmotionsConfig] = None,
    model_path: Optional[str] = None,
) -> TrainingResult:
    """
    Main training pipeline.
    
    This function orchestrates the complete workflow:
    1. Set up logging and seeding
    2. Prepare the dataset
    3. Initialize the model
    4. Train with validation
    5. Evaluate on test set
    6. Save final model and export for deployment
    
    Args:
        config: Training configuration (uses defaults if None)
        goemotions_config: GoEmotions dataset configuration
        model_path: Path to pre-trained model to continue training
    
    Returns:
        TrainingResult with best metrics and model info
    """
    # Default config
    if config is None:
        config = TrainingConfig()
    
    # Setup
    log_file = setup_logging(config)
    set_seed(config.seed)
    
    logger.info("=" * 70)
    logger.info("EMOTION LENS - GoEmotions Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Configuration: {json.dumps({'model': config.model_name, 'epochs': config.num_epochs, 'batch_size': config.batch_size, 'lr': config.learning_rate, 'loss': config.primary_loss, 'precision': config.mixed_precision}, indent=2)}")
    
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
    model, tokenizer = get_model_and_tokenizer(
        config=config,
        load_path=model_path,
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        tokenizer=tokenizer,
        goemotions_config=goemotions_config,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best epoch: {result.best_epoch}")
    logger.info(f"Training time: {result.training_time:.1f}s ({result.training_time/60:.1f} min)")
    logger.info(f"Best metrics:")
    for metric_name, value in result.best_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}")
    
    return result


def evaluate_model(
    model_path: str,
    data_path: str,
    config: Optional[TrainingConfig] = None,
) -> EmotionMetrics:
    """
    Evaluate a trained model on a dataset.
    
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
    
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
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
    
    metrics = compute_all_metrics(all_logits, all_labels, COARSE_EMOTIONS)
    
    logger.info(f"Evaluation on {data_path}:")
    logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"  Macro F1: {metrics.macro_f1:.4f}")
    logger.info(f"  Weighted F1: {metrics.weighted_f1:.4f}")
    logger.info(f"  MCC: {metrics.mcc:.4f}")
    
    return metrics
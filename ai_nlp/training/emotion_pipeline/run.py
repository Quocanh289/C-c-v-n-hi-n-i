"""
Run Script for GoEmotions Training Pipeline
============================================
Entry point for running the complete training pipeline.

Usage:
    python -m ai_nlp.training.emotion_pipeline.run
    python -m ai_nlp.training.emotion_pipeline.run --epochs 20 --batch_size 32
    python -m ai_nlp.training.emotion_pipeline.run --help
"""

import os
import sys
import argparse
import json
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Emotion Lens - GoEmotions Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing GoEmotions CSV files")
    parser.add_argument("--use_kaggle", action="store_true", default=True,
                       help="Download dataset from Kaggle")
    parser.add_argument("--filter_neutral", action="store_true", default=True,
                       help="Filter out neutral samples")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                       help="Limit validation samples")
    parser.add_argument("--augment", action="store_true", default=True,
                       help="Use data augmentation")
    
    # Model
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base",
                       help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained model for resuming training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                       help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--grad_accum", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts"],
                       help="LR scheduler type")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing")
    
    # Loss
    parser.add_argument("--primary_loss", type=str, default="focal",
                       choices=["bce", "focal", "asl", "combined"],
                       help="Primary loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter")
    parser.add_argument("--no_class_weights", action="store_true",
                       help="Disable class weights")
    
    # Optimization
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["fp16", "bf16", "no"],
                       help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    
    # Threshold optimization
    parser.add_argument("--optimize_thresholds", action="store_true", default=True,
                       help="Optimize classification thresholds")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def run_with_args():
    """Run training with parsed command-line arguments."""
    args = parse_args()
    
    # Import here to avoid slow imports during --help
    from .config import TrainingConfig, GoEmotionsConfig
    from .pipeline import train_pipeline
    
    # Build configuration from args
    config = TrainingConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        scheduler=args.scheduler,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        primary_loss=args.primary_loss,
        focal_gamma=args.focal_gamma,
        class_weights=not args.no_class_weights,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping_patience=args.patience,
        optimize_thresholds=args.optimize_thresholds,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        use_augmentation=args.augment,
        max_seq_length=args.max_length,
        seed=args.seed,
    )
    
    # Override paths if provided
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.log_dir = os.path.join(args.output_dir, "logs")
    
    # Build GoEmotions config
    goemotions_config = GoEmotionsConfig(
        filter_neutral=args.filter_neutral,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    
    # Run training
    result = train_pipeline(
        config=config,
        goemotions_config=goemotions_config,
        model_path=args.model_path,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Model: {config.model_name}")
    print(f"  Best epoch: {result.best_epoch}")
    print(f"  Training time: {result.training_time:.1f}s ({result.training_time/60:.1f} min)")
    print(f"  Checkpoint: {result.checkpoint_dir}")
    print(f"  Best metrics:")
    for metric_name, value in sorted(result.best_metrics.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"    {metric_name}: {value:.4f}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    # Set up basic logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    run_with_args()
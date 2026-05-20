"""
Run Script for GoEmotions Multi-Label Training Pipeline
========================================================
Entry point for running the complete multi-label training pipeline.

Usage:
    python -m ai_nlp.training.emotion_pipeline.run
    python -m ai_nlp.training.emotion_pipeline.run --task_type multi_label --epochs 30 --model_name xlm-roberta-base
    python -m ai_nlp.training.emotion_pipeline.run --task_type multi_class --epochs 15  # Backward compat
    python -m ai_nlp.training.emotion_pipeline.run --compare  # Run RoBERTa vs XLM-RoBERTa comparison
    python -m ai_nlp.training.emotion_pipeline.run --help
"""

import os
import sys
import argparse
import json
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for multi-label training."""
    parser = argparse.ArgumentParser(
        description="Emotion Lens - GoEmotions Multi-Label Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- Task Configuration -----
    parser.add_argument("--task_type", type=str, default="multi_label",
                        choices=["multi_label", "multi_class", "dual_head"],
                        help="Task type: multi_label (28), multi_class (9), dual_head (28+9)")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run RoBERTa vs XLM-RoBERTa comparison after training")

    # ----- Dataset -----
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing GoEmotions CSV files")
    parser.add_argument("--use_kaggle", action="store_true", default=True,
                        help="Download dataset from Kaggle")
    parser.add_argument("--filter_neutral", action="store_true", default=False,
                        help="Filter out neutral samples (not recommended)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Use data augmentation")

    # ----- Model -----
    parser.add_argument("--model_name", type=str, default="FacebookAI/xlm-roberta-base",
                        help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model for resuming training")

    # ----- Training -----
    parser.add_argument("--epochs", type=int, default=30,
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

    # ----- Loss -----
    parser.add_argument("--primary_loss", type=str, default="asl",
                        choices=["bce", "focal", "asl", "combined"],
                        help="Primary loss function (asl recommended for multi-label)")
    parser.add_argument("--secondary_loss", type=str, default="bce",
                        choices=["bce", "focal", "asl", "none"],
                        help="Secondary loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0,
                        help="ASL negative gamma")
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0,
                        help="ASL positive gamma")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Disable class weights")

    # ----- Optimization -----
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "no"],
                        help="Mixed precision training")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")

    # ----- Threshold -----
    parser.add_argument("--optimize_thresholds", action="store_true", default=True,
                        help="Optimize per-label classification thresholds")
    parser.add_argument("--threshold_trials", type=int, default=100,
                        help="Threshold optimization trials")

    # ----- Preprocessing -----
    parser.add_argument("--no_emoji_handling", action="store_true",
                        help="Disable emoji handling")
    parser.add_argument("--no_repeated_chars", action="store_true",
                        help="Disable repeated character normalization")

    # ----- Output -----
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ----- Balancing -----
    parser.add_argument("--no_balanced_sampling", action="store_true",
                        help="Disable balanced sampling")
    parser.add_argument("--balancing_strategy", type=str, default="labels",
                        choices=["none", "labels", "samples"],
                        help="Multi-label class balancing strategy")

    return parser.parse_args()


def run_with_args():
    """Run training with parsed command-line arguments."""
    args = parse_args()

    # Import here to avoid slow imports during --help
    from .config import TrainingConfig, GoEmotionsConfig
    from .pipeline import train_pipeline

    # Build configuration from args
    config = TrainingConfig(
        task_type=args.task_type,
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
        primary_loss=args.primary_loss,
        secondary_loss=args.secondary_loss,
        focal_gamma=args.focal_gamma,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        class_weights=not args.no_class_weights,
        use_balanced_sampling=not args.no_balanced_sampling,
        balancing_strategy=args.balancing_strategy,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        early_stopping_patience=args.patience,
        optimize_thresholds=args.optimize_thresholds,
        threshold_n_trials=args.threshold_trials,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_augmentation=args.augment,
        max_seq_length=args.max_length,
        handle_emojis=not args.no_emoji_handling,
        handle_repeated_chars=not args.no_repeated_chars,
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

    # Print configuration summary
    print("\n" + "=" * 70)
    print("EMOTION LENS - Multi-Label GoEmotions Training")
    print("=" * 70)
    print(f"  Task type:  {config.task_type}")
    print(f"  Model:      {config.model_name}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print(f"  Loss:       {config.primary_loss} + {config.secondary_loss}")
    print(f"  LoRA:       {config.use_lora} (r={config.lora_r})")
    print(f"  Precision:  {config.mixed_precision}")
    print(f"  Scheduler:  {config.scheduler}")
    print("=" * 70)

    # Run training
    result = train_pipeline(
        config=config,
        goemotions_config=goemotions_config,
        model_path=args.model_path,
        compare_models_flag=args.compare,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Model: {config.model_name}")
    print(f"  Task: {config.task_type}")
    print(f"  Best epoch: {result.best_epoch}")
    print(f"  Training time: {result.training_time:.1f}s ({result.training_time/60:.1f} min)")
    print(f"  Checkpoint: {result.checkpoint_dir}")

    if result.best_metrics:
        print(f"  Best metrics:")
        for metric_name, value in sorted(result.best_metrics.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"    {metric_name}: {value:.4f}")

    if result.overfitting_analysis:
        print(f"  Overfitting risk: {result.overfitting_analysis.get('overfitting_risk', 'N/A')}")

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
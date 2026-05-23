"""
CLI Entry Point
===============
Command-line interface for the mental health training pipeline.

Usage:
    # Run full training pipeline
    python -m ai_nlp.training.mental_health_pipeline.run --mode train
    
    # Run with custom config
    python -m ai_nlp.training.mental_health_pipeline.run \
        --mode train \
        --model microsoft/deberta-v3-base \
        --lr 3e-5 \
        --batch-size 16 \
        --epochs 20 \
        --loss focal
    
    # Predict using trained model
    python -m ai_nlp.training.mental_health_pipeline.run \
        --mode predict \
        --text "I feel so hopeless and depressed, nothing matters anymore"
    
    # Run EDA on the dataset
    python -m ai_nlp.training.mental_health_pipeline.run --mode eda
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_nlp.training.mental_health_pipeline import (
    TrainingConfig,
    MHConfig,
    MentalHealthPipeline,
    RedditTextPreprocessor,
)
from ai_nlp.training.mental_health_pipeline.preprocessor import analyze_text_quality

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mental Health Text Classification Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (DeBERTa-v3-base + LoRA + Focal Loss)
  python run.py --mode train
  
  # Train with XLM-RoBERTa (for multilingual later)
  python run.py --mode train --model FacebookAI/xlm-roberta-base --lr 2e-5
  
  # Train with custom settings for RTX 4060
  python run.py --mode train --batch-size 16 --accum 2 --fp16
  
  # Quick prediction
  python run.py --mode predict --text "I can't stop worrying about everything"
  
  # Analyze dataset
  python run.py --mode eda
        """
    )
    
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "predict", "eda", "export"],
        help="Pipeline mode (default: train)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Maximum sequence length (default: 256)"
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Disable LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora-r", type=int, default=None,
        help="LoRA rank (default: 8)"
    )
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 3e-5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 20)")
    parser.add_argument("--accum", type=int, default=None, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--loss", type=str, default=None, choices=["focal", "weighted_ce", "ce", "label_smooth_ce", "confusion_focal"])
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision")
    parser.add_argument("--no-aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-balanced", action="store_true", help="Disable balanced sampling")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    
    # Inference
    parser.add_argument("--text", type=str, default=None, help="Text to classify (predict mode)")
    parser.add_argument("--input-file", type=str, default=None, help="JSON file with texts to classify")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for predictions")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    return parser.parse_args()


def run_training(args):
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("MENTAL HEALTH TRAINING PIPELINE — TRAIN MODE")
    logger.info("=" * 60)
    
    # Create configuration
    config = TrainingConfig()
    mh_config = MHConfig()
    
    # Override from args
    if args.model:
        config.model_name = args.model
    if args.max_length:
        config.max_seq_length = args.max_length
        mh_config.max_length = args.max_length
    if args.no_lora:
        config.use_lora = False
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.accum:
        config.gradient_accumulation_steps = args.accum
    if args.loss:
        config.loss_type = args.loss
    if args.fp16:
        config.mixed_precision = "fp16"
    if args.no_aug:
        config.use_augmentation = False
    if args.no_balanced:
        config.use_balanced_sampling = False
    if args.patience:
        config.early_stopping_patience = args.patience
    if args.seed:
        config.seed = args.seed
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Log configuration
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Grad accum: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"Max seq length: {mh_config.max_length}")
    logger.info(f"Loss: {config.loss_type}")
    logger.info(f"LoRA: {config.use_lora} (r={config.lora_r})")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"Augmentation: {config.use_augmentation}")
    logger.info(f"Balanced sampling: {config.use_balanced_sampling}")
    logger.info(f"Early stopping patience: {config.early_stopping_patience}")
    
    # Run pipeline
    pipeline = MentalHealthPipeline(config=config, mh_config=mh_config)
    results = pipeline.run()
    
    logger.info("Training complete!")
    
    return results


def run_prediction(args):
    """Run inference using trained model."""
    logger.info("=" * 60)
    logger.info("MENTAL HEALTH PREDICTION MODE")
    logger.info("=" * 60)
    
    config = TrainingConfig()
    mh_config = MHConfig()
    
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.max_length:
        mh_config.max_length = args.max_length
    if args.model:
        config.model_name = args.model
    
    pipeline = MentalHealthPipeline(config=config, mh_config=mh_config)
    
    # Check if model exists
    best_model_path = os.path.join(
        config.get_checkpoint_dir(), "best_model", "adapter_model.safetensors"
    )
    if not os.path.exists(best_model_path):
        logger.error(
            f"No trained model found at {best_model_path}. "
            f"Run training first: python run.py --mode train"
        )
        return None
    
    # Load model
    pipeline.tokenizer = pipeline._get_tokenizer()
    pipeline.model = build_model(config)
    
    # Try loading LoRA adapter
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )
    pipeline.model.base_model = PeftModel.from_pretrained(
        base_model, os.path.join(config.get_checkpoint_dir(), "best_model")
    )
    pipeline.model.to(config.device)
    pipeline.model.eval()
    
    # Predict single text
    if args.text:
        result = pipeline.predict(args.text)
        
        print("\n" + "=" * 50)
        print(f"Text: {result['text'][:100]}...")
        print(f"Primary Condition: {result['primary_condition']}")
        print(f"Confidence: {result['primary_confidence']:.4f}")
        print(f"Needs Attention: {result['needs_attention']}")
        print("\nAll Scores:")
        for label, score in sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"  {label:25s}: {score:.4f} {bar}")
        
        print("\nTop 3 Predictions:")
        for p in result["top_predictions"]:
            print(f"  {p['label']:25s}: {p['confidence']:.4f}")
        
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
    
    # Predict from file
    if args.input_file:
        with open(args.input_file) as f:
            data = json.load(f)
        
        texts = data if isinstance(data, list) else data.get("texts", [data])
        results = []
        
        for text in texts:
            result = pipeline.predict(text)
            results.append(result)
        
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
    
    return results


def run_eda(args):
    """Run exploratory data analysis on the dataset."""
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    import pandas as pd
    
    config = TrainingConfig()
    data_path = config.get_data_file_path()
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isna().sum().to_dict()}")
    
    # Label distribution
    if config.label_column in df.columns:
        print(f"\nLabel Distribution:")
        dist = df[config.label_column].value_counts()
        for label, count in dist.items():
            print(f"  {label:30s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    # Text analysis
    if config.text_column in df.columns:
        print(f"\nText Analysis:")
        texts = df[config.text_column].dropna()
        
        quality = analyze_text_quality(df, config.text_column)
        print(f"  Non-null texts: {quality['non_null_texts']}")
        print(f"  URL mentions: {quality['urls_found']}")
        print(f"  Subreddit mentions: {quality['subreddits_found']}")
        print(f"  Username mentions: {quality['usernames_found']}")
        print(f"  Avg char length: {quality['avg_char_length']:.1f}")
        print(f"  Median char length: {quality['median_char_length']:.0f}")
        print(f"  Min char length: {quality['min_char_length']}")
        print(f"  Max char length: {quality['max_char_length']}")
        
        # Duplicates
        print(f"\nDuplicates:")
        print(f"  Exact duplicates (text): {texts.duplicated().sum()}")
    
    print(f"\nPreprocessing Preview:")
    preprocessor = RedditTextPreprocessor()
    sample_texts = df[config.text_column].dropna().sample(min(5, len(df))).tolist()
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.clean_text(text)
        print(f"\n  Sample {i+1}:")
        print(f"    Original ({len(text)} chars): {text[:150]}...")
        print(f"    Cleaned ({len(cleaned)} chars): {cleaned[:150]}...")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_prediction(args)
    elif args.mode == "eda":
        run_eda(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
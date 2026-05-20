"""Hugging Face / PyTorch / LoRA training integration helpers."""

from pathlib import Path
from typing import Optional

from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


def load_for_training(
    dataset_path: str,
    text_column: str = "text",
    label_column: str = "emotion",
):
    """
    Load exported dataset for XLM-RoBERTa fine-tuning.

    Example usage in training script:

        from datasets import load_from_disk
        from data_engine.dataset.huggingface_integration import load_for_training

        ds = load_for_training("./data/datasets/emotion_lens_social_v1")
        labeled = ds.filter(lambda x: x[label_column] is not None)
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    logger.info("dataset_loaded", path=dataset_path, rows=len(ds))
    return ds


def prepare_lora_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 128,
    label_map: Optional[dict[str, int]] = None,
):
    """Tokenize dataset for LoRA fine-tuning with Hugging Face Trainer."""
    ds = load_for_training(dataset_path)

    label_map = label_map or {
        "joy": 0, "anger": 1, "sadness": 2, "anxiety": 3,
        "fear": 4, "surprise": 5, "neutral": 6, "toxic": 7, "sarcastic": 8,
    }

    def tokenize_fn(batch):
        encodings = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encodings["labels"] = [
            label_map.get(e, 6) if e else 6 for e in batch.get("emotion", [])
        ]
        return encodings

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    return tokenized

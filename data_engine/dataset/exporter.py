"""Dataset export — JSON, JSONL, Hugging Face datasets."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from data_engine.config.settings import get_settings
from data_engine.core.models import DatasetExportRecord
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class DatasetExporter:
    """Export structured records for XLM-RoBERTa / PyTorch / LoRA training."""

    def __init__(self, output_dir: Optional[str] = None):
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.dataset_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version = settings.dataset_version

    def to_export_record(self, record: dict[str, Any]) -> DatasetExportRecord:
        return DatasetExportRecord(
            text=record.get("text_raw") or record.get("text", ""),
            language=record.get("language", "mixed"),
            source=record.get("source", "unknown"),
            timestamp=record.get("created_at", datetime.utcnow()).isoformat()
            if isinstance(record.get("created_at"), datetime)
            else str(record.get("created_at", datetime.utcnow().isoformat())),
            emotion=record.get("emotion"),
            toxicity=record.get("toxicity"),
            sarcasm=record.get("sarcasm"),
            labels=record.get("labels", {}),
            metadata={
                "text_hash": record.get("text_hash"),
                "url": record.get("url"),
                "dataset_version": record.get("dataset_version", self.version),
            },
        )

    def export_jsonl(
        self,
        records: Iterator[dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        filename = filename or f"emotion_dataset_{self.version}_{datetime.utcnow():%Y%m%d}.jsonl"
        path = self.output_dir / filename

        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                export = self.to_export_record(record)
                f.write(export.model_dump_json() + "\n")
                count += 1

        logger.info("jsonl_export_complete", path=str(path), count=count)
        return path

    def export_json(
        self,
        records: list[dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        filename = filename or f"emotion_dataset_{self.version}.json"
        path = self.output_dir / filename

        exports = [self.to_export_record(r).model_dump() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(exports, f, ensure_ascii=False, indent=2)

        logger.info("json_export_complete", path=str(path), count=len(exports))
        return path

    def export_huggingface(
        self,
        records: list[dict[str, Any]],
        dataset_name: str = "emotion_lens_social",
    ):
        """Export as Hugging Face Dataset for training pipelines."""
        from datasets import Dataset

        exports = [self.to_export_record(r).model_dump() for r in records]
        ds = Dataset.from_list(exports)
        hf_path = self.output_dir / f"{dataset_name}_{self.version}"
        ds.save_to_disk(str(hf_path))
        logger.info("huggingface_export_complete", path=str(hf_path))
        return ds

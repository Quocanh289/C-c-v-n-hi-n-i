"""Save crawled + cleaned records to disk (and optionally PostgreSQL)."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import orjson

from data_engine.config.settings import get_settings
from data_engine.core.models import ProcessedRecord, RawPost
from data_engine.dataset.exporter import DatasetExporter
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


def processed_to_labeling_dict(record: ProcessedRecord) -> dict[str, Any]:
    """Format ready for Label Studio / human annotation."""
    return {
        "text": record.text_raw,
        "text_clean": record.text,
        "text_hash": record.text_hash,
        "language": record.language,
        "source": record.source.value if hasattr(record.source, "value") else str(record.source),
        "timestamp": record.timestamp.isoformat(),
        "url": record.url,
        "emotion": record.emotion,
        "toxicity": record.toxicity,
        "sarcasm": record.sarcasm,
        "labels": record.labels,
        "tokens": record.tokens,
        "emojis": record.emojis,
        "is_spam": record.is_spam,
        "metadata": record.metadata,
    }


def raw_to_dict(post: RawPost) -> dict[str, Any]:
    d = post.model_dump()
    d["platform"] = post.platform.value
    return d


class CrawlPipelineStore:
    """
    Three-layer local storage (no Docker required):

    data/crawls/
      raw/       — untouched crawler output
      clean/     — after ProcessingPipeline (non-spam)
      labeling/  — same as export format, emotion/toxicity/sarcasm = null
    """

    def __init__(self, base_dir: Optional[str | Path] = None):
        settings = get_settings()
        root = Path(base_dir or settings.dataset_output_dir).parent / "crawls"
        self.raw_dir = root / "raw"
        self.clean_dir = root / "clean"
        self.labeling_dir = root / "labeling"
        for d in (self.raw_dir, self.clean_dir, self.labeling_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._exporter = DatasetExporter(
            output_dir=str(root.parent / "datasets")
        )

    def _paths(self, platform: str, run_id: str) -> dict[str, Path]:
        prefix = f"{platform}_{run_id}"
        return {
            "raw": self.raw_dir / f"{prefix}_raw.jsonl",
            "clean": self.clean_dir / f"{prefix}_clean.jsonl",
            "labeling": self.labeling_dir / f"{prefix}_for_labeling.jsonl",
        }

    def append_raw(self, path: Path, post: RawPost) -> None:
        with open(path, "ab") as f:
            f.write(orjson.dumps(raw_to_dict(post)) + b"\n")

    def append_clean(self, path: Path, record: ProcessedRecord) -> None:
        with open(path, "ab") as f:
            f.write(orjson.dumps(processed_to_labeling_dict(record)) + b"\n")

    def save_to_postgres(self, record: ProcessedRecord) -> bool:
        """Optional: requires PostgreSQL running."""
        try:
            from data_engine.storage.postgres_repo import PostgresRepository

            pg = PostgresRepository()
            return pg.save_dataset_record({
                "text_hash": record.text_hash,
                "text": record.text,
                "text_raw": record.text_raw,
                "language": record.language,
                "source": record.source.value,
                "platform_post_id": record.platform_post_id,
                "url": record.url,
                "labels": record.labels,
                "is_spam": record.is_spam,
                "dataset_version": get_settings().dataset_version,
                "metadata": record.metadata,
            })
        except Exception as exc:
            logger.warning("postgres_save_skipped", error=str(exc))
            return False

    def finalize_run(
        self,
        platform: str,
        paths: dict[str, Path],
        stats: dict[str, int],
    ) -> dict[str, Any]:
        logger.info(
            "crawl_saved",
            platform=platform,
            raw_path=str(paths["raw"]),
            clean_path=str(paths["clean"]),
            labeling_path=str(paths["labeling"]),
            raw_count=stats.get("raw", 0),
            clean_count=stats.get("clean", 0),
            spam_count=stats.get("spam", 0),
        )
        return {
            "raw_path": str(paths["raw"]),
            "clean_path": str(paths["clean"]),
            "labeling_path": str(paths["labeling"]),
            **stats,
        }


def new_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

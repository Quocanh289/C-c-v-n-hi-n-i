"""Celery tasks — crawl, process, slang drift, export."""

import asyncio
from datetime import datetime, timedelta

import redis

from data_engine.config.settings import get_settings
from data_engine.core.models import CrawlJob, Platform
from data_engine.crawlers import get_crawler
from data_engine.dataset.exporter import DatasetExporter
from data_engine.embeddings.qdrant_store import QdrantVectorStore
from data_engine.monitoring.logging import configure_logging, get_logger
from data_engine.processing.pipeline import ProcessingPipeline
from data_engine.slang.drift_detector import SlangDriftDetector
from data_engine.storage.mongo_store import MongoStore
from data_engine.storage.postgres_repo import PostgresRepository
from data_engine.workers.celery_app import celery_app

configure_logging()
logger = get_logger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_redis():
    settings = get_settings()
    return redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=False)


@celery_app.task(bind=True, max_retries=3, name="data_engine.workers.tasks.run_crawl_job")
def run_crawl_job(self, platform: str, target: str, params: dict | None = None):
    """Distributed crawl task — one job per worker."""
    settings = get_settings()
    redis_client = _get_redis()

    async def _execute():
        crawler = get_crawler(platform, redis_client)
        job = CrawlJob(
            platform=Platform(platform),
            target=target,
            params=params or {},
        )
        pipeline = ProcessingPipeline()
        mongo = MongoStore()
        pg = PostgresRepository()
        qdrant = QdrantVectorStore()
        qdrant.ensure_collection()

        await mongo.ensure_indexes()
        collected = 0

        try:
            async for raw_post in crawler.run_job(job):
                processed = pipeline.process(raw_post)
                if not processed or processed.is_spam:
                    continue

                is_sem_dup, _ = qdrant.is_semantic_duplicate(processed.text_raw)
                if is_sem_dup:
                    continue

                await mongo.insert_raw_post(raw_post, processed.text_hash)
                pg.save_dataset_record({
                    "text_hash": processed.text_hash,
                    "text": processed.text,
                    "text_raw": processed.text_raw,
                    "language": processed.language,
                    "source": processed.source.value,
                    "platform_post_id": processed.platform_post_id,
                    "url": processed.url,
                    "labels": processed.labels,
                    "is_spam": processed.is_spam,
                    "dataset_version": settings.dataset_version,
                    "metadata": processed.metadata,
                })
                qdrant.upsert(
                    processed.text_raw,
                    processed.text_hash,
                    {"source": processed.source.value, "language": processed.language},
                )
                collected += 1

        finally:
            await crawler.close()

        return {"job_id": job.id, "collected": collected, "status": job.status.value}

    return _run_async(_execute())


@celery_app.task(name="data_engine.workers.tasks.run_slang_drift")
def run_slang_drift(window_days: int = 7):
    """Detect emerging slang from token frequency drift."""
    settings = get_settings()

    async def _execute():
        mongo = MongoStore()
        pg = PostgresRepository()
        now = datetime.utcnow()
        current_start = now - timedelta(days=window_days)
        previous_start = current_start - timedelta(days=window_days)

        current_texts = await mongo.get_texts_for_window(current_start, now)
        previous_texts = await mongo.get_texts_for_window(previous_start, current_start)

        detector = SlangDriftDetector()
        candidates = detector.detect_emerging(current_texts, previous_texts)

        saved = pg.save_slang_candidates([
            {
                "term": c.term,
                "language": c.language,
                "frequency_current": c.frequency_current,
                "frequency_previous": c.frequency_previous,
                "growth_rate": c.growth_rate,
                "contexts": c.contexts,
                "status": c.status,
            }
            for c in candidates
        ])

        return {"candidates": len(candidates), "saved": saved}

    return _run_async(_execute())


@celery_app.task(name="data_engine.workers.tasks.export_dataset")
def export_dataset(format: str = "jsonl"):
    """Export labeled dataset for training."""
    pg = PostgresRepository()
    # In production: query all non-spam records
    samples = pg.get_unlabeled_samples(limit=10000)
    exporter = DatasetExporter()

    if format == "json":
        return str(exporter.export_json(samples))
    return str(exporter.export_jsonl(iter(samples)))

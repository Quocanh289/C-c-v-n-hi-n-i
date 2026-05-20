"""FastAPI control plane for the data engine."""

from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_engine.config.settings import get_settings
from data_engine.core.models import Platform
from data_engine.core.scheduler import CrawlScheduler
from data_engine.monitoring.logging import configure_logging, get_logger
from data_engine.workers.tasks import export_dataset, run_crawl_job, run_slang_drift

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


class CrawlRequestBody(BaseModel):
    platform: str
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5


class ExportRequest(BaseModel):
    format: str = "jsonl"
    version: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=False,
    )
    app.state.scheduler = CrawlScheduler()
    logger.info("data_engine_api_started")
    yield
    await app.state.redis.close()


app = FastAPI(
    title="Emotion Lens Data Engine",
    description="Vietnamese social-media crawling & dataset generation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    try:
        await app.state.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": redis_ok,
        "environment": settings.environment,
    }


@app.post("/api/v1/crawl")
async def start_crawl(body: CrawlRequestBody):
    """Enqueue a distributed crawl job."""
    if not app.state.scheduler.platform_enabled(body.platform):
        raise HTTPException(400, f"Platform disabled: {body.platform}")

    try:
        Platform(body.platform)
    except ValueError:
        raise HTTPException(400, f"Unknown platform: {body.platform}")

    task = run_crawl_job.delay(body.platform, body.target, body.params)
    logger.info("crawl_enqueued", platform=body.platform, task_id=task.id)
    return {"task_id": task.id, "status": "queued"}


@app.get("/api/v1/crawl/{task_id}")
async def crawl_status(task_id: str):
    result = run_crawl_job.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


@app.post("/api/v1/slang/drift")
async def trigger_slang_drift(window_days: int = 7):
    task = run_slang_drift.delay(window_days)
    return {"task_id": task.id, "status": "queued"}


@app.post("/api/v1/dataset/export")
async def trigger_export(body: ExportRequest):
    task = export_dataset.delay(body.format)
    return {"task_id": task.id, "format": body.format}


@app.get("/api/v1/platforms")
async def list_platforms():
    config = app.state.scheduler.config.get("platforms", {})
    return {
        name: {"enabled": cfg.get("enabled"), "mode": cfg.get("mode")}
        for name, cfg in config.items()
    }

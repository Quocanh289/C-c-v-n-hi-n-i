"""Celery application for distributed crawl and processing tasks."""

from celery import Celery
from celery.schedules import crontab

from data_engine.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "data_engine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "data_engine.workers.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "data_engine.workers.tasks.run_crawl_job": {"queue": "crawl"},
        "data_engine.workers.tasks.process_raw_batch": {"queue": "process"},
        "data_engine.workers.tasks.run_slang_drift": {"queue": "analytics"},
        "data_engine.workers.tasks.export_dataset": {"queue": "export"},
    },
    beat_schedule={
        "slang-drift-daily": {
            "task": "data_engine.workers.tasks.run_slang_drift",
            "schedule": crontab(hour=2, minute=0),
        },
        "dataset-export-weekly": {
            "task": "data_engine.workers.tasks.export_dataset",
            "schedule": crontab(hour=4, minute=0, day_of_week=0),
        },
    },
)

"""Crawl job scheduling — integrates with Celery Beat."""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from croniter import croniter

from data_engine.core.models import CrawlJob, Platform
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


def load_platform_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config" / "platforms.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CrawlScheduler:
    """Build CrawlJob instances from YAML schedules and API requests."""

    def __init__(self):
        self.config = load_platform_config()

    def platform_enabled(self, platform: str) -> bool:
        platforms = self.config.get("platforms", {})
        cfg = platforms.get(platform, {})
        return cfg.get("enabled", False)

    def get_rate_limit(self, platform: str) -> int:
        platforms = self.config.get("platforms", {})
        return platforms.get(platform, {}).get("rate_limit_rpm", 30)

    def create_job(
        self,
        platform: Platform,
        target: str,
        params: dict[str, Any] | None = None,
        priority: int = 5,
    ) -> CrawlJob:
        return CrawlJob(
            platform=platform,
            target=target,
            params=params or {},
            priority=priority,
        )

    def due_schedules(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Return schedule entries that should fire now."""
        now = now or datetime.utcnow()
        due = []
        schedules = self.config.get("schedules", {})

        for name, spec in schedules.items():
            cron_expr = spec.get("cron")
            if not cron_expr:
                continue
            cron = croniter(cron_expr, now)
            prev_run = cron.get_prev(datetime)
            # Fire if within last minute of scheduled time
            delta = (now - prev_run).total_seconds()
            if 0 <= delta < 60:
                due.append({"name": name, **spec})

        return due

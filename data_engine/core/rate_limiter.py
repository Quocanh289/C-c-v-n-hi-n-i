"""Token-bucket rate limiter backed by Redis for distributed crawling."""

import asyncio
import time
from typing import Optional

import redis.asyncio as aioredis

from data_engine.config.settings import get_settings
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class DistributedRateLimiter:
    """
    Sliding-window rate limiter using Redis INCR + EXPIRE.
    Safe for multiple Celery workers / crawler pods.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        key_prefix: str = "ratelimit",
    ):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.settings = get_settings()

    def _key(self, platform: str, window_sec: int = 60) -> str:
        bucket = int(time.time()) // window_sec
        return f"{self.key_prefix}:{platform}:{bucket}"

    async def acquire(
        self,
        platform: str,
        limit_rpm: Optional[int] = None,
        window_sec: int = 60,
    ) -> bool:
        """Try to acquire one request slot. Returns False if rate exceeded."""
        limit = limit_rpm or self.settings.default_rate_limit_rpm
        key = self._key(platform, window_sec)

        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_sec + 5)
        count, _ = await pipe.execute()

        if count > limit:
            logger.debug(
                "rate_limit_exceeded",
                platform=platform,
                count=count,
                limit=limit,
            )
            return False
        return True

    async def wait_for_slot(
        self,
        platform: str,
        limit_rpm: Optional[int] = None,
        max_wait_sec: float = 120.0,
    ) -> None:
        """Block until a rate limit slot is available."""
        start = time.monotonic()
        while True:
            if await self.acquire(platform, limit_rpm):
                return
            if time.monotonic() - start > max_wait_sec:
                raise TimeoutError(f"Rate limit wait exceeded for {platform}")
            await asyncio.sleep(1.0)

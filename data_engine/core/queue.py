"""Redis-backed distributed crawl request queue."""

import json
from typing import AsyncIterator, Optional

import redis.asyncio as aioredis

from data_engine.core.models import CrawlRequest, Platform
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class CrawlQueue:
    """
    Priority queue using Redis sorted sets.
    Score = priority (lower = higher priority) + timestamp tie-breaker.
    """

    QUEUE_KEY = "crawl:queue"
    PROCESSING_KEY = "crawl:processing"
    DEAD_LETTER_KEY = "crawl:dead_letter"

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def enqueue(
        self,
        request: CrawlRequest,
        priority: int = 5,
    ) -> None:
        payload = request.model_dump_json()
        score = float(f"{priority}.{hash(request.url) % 10000:04d}")
        await self.redis.zadd(self.QUEUE_KEY, {payload: score})
        logger.info(
            "enqueued",
            platform=request.platform.value,
            url=request.url[:80],
            job_id=request.job_id,
        )

    async def dequeue(self, timeout_sec: int = 5) -> Optional[CrawlRequest]:
        """Blocking pop — moves item to processing set."""
        result = await self.redis.bzpopmin(self.QUEUE_KEY, timeout=timeout_sec)
        if not result:
            return None

        _, payload, _ = result
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        await self.redis.sadd(self.PROCESSING_KEY, payload)
        return CrawlRequest.model_validate_json(payload)

    async def ack(self, request: CrawlRequest) -> None:
        payload = request.model_dump_json()
        await self.redis.srem(self.PROCESSING_KEY, payload)

    async def nack(
        self,
        request: CrawlRequest,
        requeue: bool = True,
    ) -> None:
        payload = request.model_dump_json()
        await self.redis.srem(self.PROCESSING_KEY, payload)

        request.retry_count += 1
        if request.retry_count >= 5:
            await self.redis.lpush(self.DEAD_LETTER_KEY, payload)
            logger.error("dead_letter", url=request.url, retries=request.retry_count)
            return

        if requeue:
            await self.enqueue(request, priority=10 + request.retry_count)

    async def size(self) -> int:
        return await self.redis.zcard(self.QUEUE_KEY)

    async def iter_batch(
        self,
        batch_size: int = 10,
    ) -> AsyncIterator[list[CrawlRequest]]:
        batch: list[CrawlRequest] = []
        while True:
            req = await self.dequeue(timeout_sec=2)
            if req is None:
                if batch:
                    yield batch
                    batch = []
                break
            batch.append(req)
            if len(batch) >= batch_size:
                yield batch
                batch = []

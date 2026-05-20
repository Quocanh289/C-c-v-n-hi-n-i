"""Retry policies for crawler HTTP and browser operations."""

import asyncio
from typing import Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from data_engine.config.settings import get_settings
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


RETRYABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


async def with_retry(
    coro_factory: Callable[[], T],
    max_attempts: int | None = None,
    operation_name: str = "crawl_request",
) -> T:
    """Execute async callable with exponential backoff + jitter."""
    settings = get_settings()
    attempts = max_attempts or settings.max_retries

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(
            initial=settings.retry_base_delay_sec,
            max=60,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    ):
        with attempt:
            try:
                return await coro_factory()
            except RETRYABLE_EXCEPTIONS as exc:
                logger.warning(
                    "retryable_error",
                    operation=operation_name,
                    attempt=attempt.retry_state.attempt_number,
                    error=str(exc),
                )
                raise

    raise RuntimeError("Unreachable retry exit")

#!/usr/bin/env python3
"""CLI to run a single crawl job locally (no Celery)."""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from any cwd: `python data_engine/scripts/run_crawl.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_engine.config.settings import get_settings
from data_engine.core.models import CrawlJob, Platform
from data_engine.core.playwright_check import PlaywrightNotInstalledError, verify_playwright_async
from data_engine.crawlers import get_crawler

BROWSER_PLATFORMS = frozenset({
    Platform.TIKTOK.value,
    Platform.FACEBOOK.value,
    Platform.THREADS.value,
})
from data_engine.monitoring.logging import configure_logging, get_logger
from data_engine.processing.pipeline import ProcessingPipeline
from data_engine.storage.pipeline_store import CrawlPipelineStore, new_run_id

configure_logging()
logger = get_logger(__name__)


async def _connect_redis():
    """Return Redis client if reachable, else None (use in-memory backends)."""
    import redis.asyncio as aioredis

    settings = get_settings()
    client = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=False,
    )
    try:
        await asyncio.wait_for(client.ping(), timeout=2.0)
        return client
    except Exception as exc:
        logger.warning(
            "redis_unavailable",
            error=str(exc),
            msg="Using in-memory dedup and rate limiting (--redis to require Redis)",
        )
        await client.aclose()
        return None


def _needs_playwright(platform: str) -> bool:
    if platform in BROWSER_PLATFORMS:
        return True
    # YouTube without API key uses Playwright fallback
    if platform == Platform.YOUTUBE.value:
        return not get_settings().youtube_api_key
    return False


async def main(
    platform: str,
    target: str,
    limit: int,
    use_redis: bool,
    save: bool,
    save_db: bool,
):
    if _needs_playwright(platform):
        try:
            mode = await verify_playwright_async()
            logger.info("playwright_ready", launch_mode=mode)
        except PlaywrightNotInstalledError as exc:
            raise SystemExit(str(exc)) from exc

    if platform == Platform.FACEBOOK.value and not target.startswith("http"):
        from data_engine.crawlers.facebook import resolve_facebook_url
        logger.info("facebook_url_hint", resolved=resolve_facebook_url(target))

    redis_client = None
    local_mode = True

    if use_redis:
        redis_client = await _connect_redis()
        if redis_client is None:
            raise SystemExit(
                "Redis is not running. Start Redis or omit --redis to use local mode.\n"
                "  docker compose up -d redis\n"
                "  python run_crawl.py --platform reddit --target vietnam"
            )
        local_mode = False
    else:
        # Default: try Redis, fall back to in-memory
        redis_client = await _connect_redis()
        local_mode = redis_client is None

    crawler = get_crawler(platform, redis_client, local_mode=local_mode)
    pipeline = ProcessingPipeline()
    store = CrawlPipelineStore() if (save or save_db) else None
    run_id = new_run_id() if store else ""
    paths = store._paths(platform, run_id) if store else {}

    job = CrawlJob(
        platform=Platform(platform),
        target=target,
        params={"limit": limit},
    )

    stats = {"clean": 0, "spam": 0, "raw": 0}
    try:
        async for post in crawler.run_job(job):
            stats["raw"] += 1
            if store:
                store.append_raw(paths["raw"], post)

            processed = pipeline.process(post)
            if not processed:
                continue
            if processed.is_spam:
                stats["spam"] += 1
                continue

            stats["clean"] += 1
            print(processed.text_raw[:120])

            if store:
                store.append_clean(paths["clean"], processed)
                store.append_clean(paths["labeling"], processed)
            if save_db and store:
                store.save_to_postgres(processed)
    finally:
        await crawler.close()
        if redis_client:
            await redis_client.aclose()

    if store:
        result = store.finalize_run(platform, paths, stats)
        print("\n--- Saved ---")
        print(f"  Raw:      {result['raw_path']}")
        print(f"  Clean:    {result['clean_path']}")
        print(f"  Labeling: {result['labeling_path']}")
        print(f"  Records:  {result['clean']} clean, {result['spam']} spam")

    logger.info("done", local_mode=local_mode, **stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local crawl")
    parser.add_argument("--platform", required=True, choices=[p.value for p in Platform])
    parser.add_argument(
        "--target",
        required=True,
        help="reddit: subreddit | youtube: video_id | facebook: page URL or slug | tiktok: video/tag URL",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Require Redis (fail if not running). Default: auto-fallback to in-memory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save raw + cleaned JSONL under data/crawls/ (ready for labeling)",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Also insert clean records into PostgreSQL (requires DB running)",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.platform,
            args.target,
            args.limit,
            args.redis,
            args.save,
            args.save_db,
        )
    )

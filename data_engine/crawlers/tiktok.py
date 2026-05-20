"""TikTok comments crawler — Playwright for dynamic content."""

import re
from typing import AsyncIterator

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)

# Selectors may need updates as TikTok DOM changes
COMMENT_SELECTORS = [
    '[data-e2e="comment-level-1"]',
    '.comment-item',
    'p[data-e2e="comment-level-1"]',
]


class TikTokCrawler(BaseCrawler):
    platform = Platform.TIKTOK
    rate_limit_rpm = 15
    use_browser = True

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        url = job.target or job.params.get("url")
        if url:
            yield url
        else:
            hashtag = job.params.get("hashtag", "vietnam")
            yield f"https://www.tiktok.com/tag/{hashtag}"

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        max_scrolls = job.params.get("max_scrolls", 15)

        await self.rate_limiter.wait_for_slot(self.platform.value, self.rate_limit_rpm)

        async with self.browser.acquire_page(locale="vi-VN") as page:
            await page.goto(target, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)

            # Open comments panel if on video page
            comment_btn = page.locator('[data-e2e="browse-comment-icon"]')
            if await comment_btn.count() > 0:
                await comment_btn.first.click()
                await page.wait_for_timeout(2000)

            await self.browser.infinite_scroll(
                page,
                max_scrolls=max_scrolls,
                wait_ms=2000,
            )

            seen_texts: set[str] = set()
            for selector in COMMENT_SELECTORS:
                elements = page.locator(selector)
                count = await elements.count()

                for i in range(count):
                    el = elements.nth(i)
                    text = (await el.inner_text()).strip()
                    text = re.sub(r"\n+", " ", text)

                    if len(text) < 3 or text in seen_texts:
                        continue
                    seen_texts.add(text)

                    yield RawPost(
                        platform=Platform.TIKTOK,
                        text=text,
                        url=target,
                        metadata={"selector": selector, "index": i},
                    )

                if seen_texts:
                    break

            logger.info(
                "tiktok_crawl_done",
                url=target[:80],
                comments=len(seen_texts),
            )

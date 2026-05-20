"""Meta Threads crawler — browser automation."""

import re
from typing import AsyncIterator

from data_engine.core.base_crawler import BaseCrawler
from data_engine.core.models import CrawlJob, Platform, RawPost


class ThreadsCrawler(BaseCrawler):
    platform = Platform.THREADS
    rate_limit_rpm = 15
    use_browser = True

    async def discover(self, job: CrawlJob) -> AsyncIterator[str]:
        url = job.target or job.params.get("url", "https://www.threads.net/tag/vietnam")
        yield url

    async def fetch_posts(self, target: str, job: CrawlJob) -> AsyncIterator[RawPost]:
        max_scrolls = job.params.get("max_scrolls", 15)

        await self.rate_limiter.wait_for_slot(self.platform.value, self.rate_limit_rpm)

        async with self.browser.acquire_page() as page:
            await page.goto(target, wait_until="domcontentloaded")
            await page.wait_for_timeout(2500)
            await self.browser.infinite_scroll(page, max_scrolls=max_scrolls)

            # Threads posts often in span elements with dir=auto
            posts = page.locator('span[dir="auto"]')
            count = await posts.count()
            seen: set[str] = set()

            for i in range(min(count, 200)):
                text = (await posts.nth(i).inner_text()).strip()
                text = re.sub(r"\s+", " ", text)

                if len(text) < 5 or text in seen or len(text) > 2000:
                    continue
                seen.add(text)

                yield RawPost(
                    platform=Platform.THREADS,
                    text=text,
                    url=target,
                    metadata={"index": i},
                )

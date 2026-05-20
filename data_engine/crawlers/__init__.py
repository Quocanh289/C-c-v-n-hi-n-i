"""Platform-specific crawlers."""

from data_engine.crawlers.reddit import RedditCrawler
from data_engine.crawlers.youtube import YouTubeCrawler
from data_engine.crawlers.tiktok import TikTokCrawler
from data_engine.crawlers.facebook import FacebookCrawler
from data_engine.crawlers.threads import ThreadsCrawler
from data_engine.crawlers.twitter import TwitterCrawler

CRAWLER_REGISTRY = {
    "reddit": RedditCrawler,
    "youtube": YouTubeCrawler,
    "tiktok": TikTokCrawler,
    "facebook": FacebookCrawler,
    "threads": ThreadsCrawler,
    "twitter": TwitterCrawler,
}


def get_crawler(platform: str, redis_client=None, *, local_mode: bool = False):
    cls = CRAWLER_REGISTRY.get(platform)
    if not cls:
        raise ValueError(f"Unknown platform: {platform}")
    return cls(redis_client, local_mode=local_mode or redis_client is None)

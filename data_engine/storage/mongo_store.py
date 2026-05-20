"""MongoDB store for raw crawled documents and crawl metadata."""

from datetime import datetime
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from data_engine.config.settings import get_settings
from data_engine.core.models import RawPost
from data_engine.monitoring.logging import get_logger

logger = get_logger(__name__)


class MongoStore:
    """
    Collections:
    - raw_posts: unprocessed crawled documents
    - crawl_logs: per-job execution logs
    - token_snapshots: slang frequency snapshots by window
    """

    def __init__(self, client: Optional[AsyncIOMotorClient] = None):
        settings = get_settings()
        self.client = client or AsyncIOMotorClient(settings.mongo_uri)
        self.db: AsyncIOMotorDatabase = self.client[settings.mongo_db]

    @property
    def raw_posts(self):
        return self.db.raw_posts

    @property
    def crawl_logs(self):
        return self.db.crawl_logs

    @property
    def token_snapshots(self):
        return self.db.token_snapshots

    async def ensure_indexes(self) -> None:
        await self.raw_posts.create_index([("platform", 1), ("crawled_at", -1)])
        await self.raw_posts.create_index("post_id", unique=True, sparse=True)
        await self.raw_posts.create_index("text_hash")
        await self.token_snapshots.create_index([("window_start", -1), ("term", 1)])

    async def insert_raw_post(self, post: RawPost, text_hash: str) -> str:
        doc = {
            **post.model_dump(),
            "platform": post.platform.value,
            "text_hash": text_hash,
            "crawled_at": post.crawled_at,
            "inserted_at": datetime.utcnow(),
        }
        result = await self.raw_posts.insert_one(doc)
        return str(result.inserted_id)

    async def get_texts_for_window(
        self,
        start: datetime,
        end: datetime,
        platform: Optional[str] = None,
        limit: int = 50000,
    ) -> list[str]:
        query: dict[str, Any] = {
            "crawled_at": {"$gte": start, "$lt": end},
        }
        if platform:
            query["platform"] = platform

        cursor = self.raw_posts.find(query, {"text": 1}).limit(limit)
        return [doc["text"] async for doc in cursor]

    async def save_token_snapshot(
        self,
        window_start: datetime,
        window_end: datetime,
        frequencies: dict[str, int],
    ) -> None:
        await self.token_snapshots.insert_one({
            "window_start": window_start,
            "window_end": window_end,
            "frequencies": frequencies,
            "term_count": len(frequencies),
            "created_at": datetime.utcnow(),
        })

"""Shared data models for crawlers and pipeline."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Platform(str, Enum):
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    THREADS = "threads"
    TWITTER = "twitter"


class CrawlStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RawPost(BaseModel):
    """Raw crawled item before processing."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: Platform
    text: str
    author_id: Optional[str] = None
    post_id: Optional[str] = None
    parent_id: Optional[str] = None
    url: Optional[str] = None
    language_hint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    crawled_at: datetime = Field(default_factory=datetime.utcnow)


class ProcessedRecord(BaseModel):
    """Cleaned record ready for dataset storage."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    text_raw: str
    text_hash: str
    language: str
    source: Platform
    platform_post_id: Optional[str] = None
    url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotion: Optional[str] = None
    toxicity: Optional[float] = None
    sarcasm: Optional[float] = None
    labels: dict[str, Any] = Field(default_factory=dict)
    tokens: list[str] = Field(default_factory=list)
    emojis: list[str] = Field(default_factory=list)
    is_duplicate: bool = False
    is_spam: bool = False
    dataset_version: str = "v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrawlJob(BaseModel):
    """Distributed crawl job definition."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: Platform
    target: str  # subreddit, video_id, hashtag, page_url, etc.
    params: dict[str, Any] = Field(default_factory=dict)
    status: CrawlStatus = CrawlStatus.PENDING
    priority: int = 5
    worker_id: Optional[str] = None
    items_collected: int = 0
    items_skipped: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CrawlRequest(BaseModel):
    """Single URL/item in the crawl queue."""

    url: str
    platform: Platform
    job_id: str
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0


class SlangCandidate(BaseModel):
    term: str
    language: str = "mixed"
    frequency_current: int = 0
    frequency_previous: int = 0
    growth_rate: float = 0.0
    contexts: list[str] = Field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    status: str = "candidate"  # candidate | emerging | verified | rejected


class DatasetExportRecord(BaseModel):
    """JSON export format for training."""

    text: str
    language: str
    source: str
    timestamp: str
    emotion: Optional[str] = None
    toxicity: Optional[float] = None
    sarcasm: Optional[float] = None
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

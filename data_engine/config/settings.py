"""Central configuration for the data engine."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db_queue: int = Field(default=1, alias="REDIS_DB_QUEUE")
    redis_db_dedup: int = Field(default=2, alias="REDIS_DB_DEDUP")
    redis_db_rate: int = Field(default=3, alias="REDIS_DB_RATE")

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def celery_broker_url(self) -> str:
        return f"{self.redis_url}/0"

    @property
    def celery_result_backend(self) -> str:
        return f"{self.redis_url}/0"

    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="emotionlens", alias="POSTGRES_DB")
    postgres_user: str = Field(default="emotionlens", alias="POSTGRES_USER")
    postgres_password: str = Field(default="changeme", alias="POSTGRES_PASSWORD")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # MongoDB
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        alias="MONGO_URI",
    )
    mongo_db: str = Field(default="emotionlens_raw", alias="MONGO_DB")

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        default="social_text_embeddings",
        alias="QDRANT_COLLECTION",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # Crawler
    max_concurrent_requests: int = Field(default=20, alias="MAX_CONCURRENT_REQUESTS")
    default_rate_limit_rpm: int = Field(default=30, alias="DEFAULT_RATE_LIMIT_RPM")
    request_timeout_sec: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SEC")
    max_retries: int = Field(default=5, alias="MAX_RETRIES")
    retry_base_delay_sec: float = Field(default=1.0, alias="RETRY_BASE_DELAY_SEC")
    crawl_batch_size: int = Field(default=50, alias="CRAWL_BATCH_SIZE")
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        alias="CRAWLER_USER_AGENT",
    )

    # Playwright
    playwright_headless: bool = Field(default=True, alias="PLAYWRIGHT_HEADLESS")
    playwright_slow_mo_ms: int = Field(default=0, alias="PLAYWRIGHT_SLOW_MO_MS")

    # Platform API keys (optional — API-first when available)
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: Optional[str] = Field(default=None, alias="REDDIT_USER_AGENT")
    youtube_api_key: Optional[str] = Field(default=None, alias="YOUTUBE_API_KEY")
    twitter_bearer_token: Optional[str] = Field(default=None, alias="TWITTER_BEARER_TOKEN")

    # Processing
    min_text_length: int = Field(default=3, alias="MIN_TEXT_LENGTH")
    max_text_length: int = Field(default=2000, alias="MAX_TEXT_LENGTH")
    semantic_dedup_threshold: float = Field(default=0.92, alias="SEMANTIC_DEDUP_THRESHOLD")
    minhash_threshold: float = Field(default=0.85, alias="MINHASH_THRESHOLD")

    # Slang drift
    slang_growth_threshold: float = Field(default=3.0, alias="SLANG_GROWTH_THRESHOLD")
    slang_min_frequency: int = Field(default=5, alias="SLANG_MIN_FREQUENCY")
    slang_window_days: int = Field(default=7, alias="SLANG_WINDOW_DAYS")

    # Labeling
    label_studio_url: Optional[str] = Field(default=None, alias="LABEL_STUDIO_URL")
    label_studio_api_key: Optional[str] = Field(default=None, alias="LABEL_STUDIO_API_KEY")
    active_learning_confidence_low: float = Field(default=0.35, alias="AL_CONFIDENCE_LOW")
    active_learning_confidence_high: float = Field(default=0.65, alias="AL_CONFIDENCE_HIGH")

    # Dataset export
    dataset_output_dir: str = Field(default="./data/datasets", alias="DATASET_OUTPUT_DIR")
    dataset_version: str = Field(default="v1", alias="DATASET_VERSION")

    # Proxy (anti-ban)
    proxy_url: Optional[str] = Field(default=None, alias="PROXY_URL")
    proxy_rotation_enabled: bool = Field(default=False, alias="PROXY_ROTATION_ENABLED")


@lru_cache
def get_settings() -> Settings:
    return Settings()

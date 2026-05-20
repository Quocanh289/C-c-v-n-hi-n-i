"""PostgreSQL repository for crawl jobs, dataset records, slang."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.sql import func
import uuid

from data_engine.config.settings import get_settings


class Base(DeclarativeBase):
    pass


class CrawlJobRecord(Base):
    __tablename__ = "crawl_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(String(32), nullable=False, index=True)
    target = Column(Text, nullable=False)
    status = Column(String(16), nullable=False, default="pending", index=True)
    params = Column(JSONB, default={})
    items_collected = Column(Integer, default=0)
    items_skipped = Column(Integer, default=0)
    error_message = Column(Text)
    priority = Column(Integer, default=5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))


class DatasetRecord(Base):
    __tablename__ = "dataset_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text_hash = Column(String(64), nullable=False, unique=True, index=True)
    text = Column(Text, nullable=False)
    text_raw = Column(Text, nullable=False)
    language = Column(String(8), nullable=False, index=True)
    source = Column(String(32), nullable=False, index=True)
    platform_post_id = Column(String(128))
    url = Column(Text)
    emotion = Column(String(32))
    toxicity = Column(Float)
    sarcasm = Column(Float)
    labels = Column(JSONB, default={})
    is_spam = Column(Boolean, default=False)
    is_labeled = Column(Boolean, default=False, index=True)
    dataset_version = Column(String(16), default="v1", index=True)
    record_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SlangCandidateRecord(Base):
    __tablename__ = "slang_candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term = Column(String(128), nullable=False, index=True)
    language = Column(String(8), default="mixed")
    frequency_current = Column(Integer, default=0)
    frequency_previous = Column(Integer, default=0)
    growth_rate = Column(Float, default=0.0)
    contexts = Column(JSONB, default=[])
    status = Column(String(16), default="candidate", index=True)
    reviewed_by = Column(String(128))
    reviewed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(32), unique=True, nullable=False)
    record_count = Column(Integer, default=0)
    export_path = Column(Text)
    version_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PostgresRepository:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(settings.postgres_sync_dsn, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save_dataset_record(self, record: dict[str, Any]) -> bool:
        with self.SessionLocal() as session:
            existing = session.execute(
                select(DatasetRecord).where(
                    DatasetRecord.text_hash == record["text_hash"]
                )
            ).scalar_one_or_none()
            if existing:
                return False

            row = {**record}
            if "metadata" in row:
                row["record_metadata"] = row.pop("metadata")
            session.add(DatasetRecord(**row))
            session.commit()
            return True

    def get_unlabeled_samples(
        self,
        limit: int = 100,
        confidence_range: Optional[tuple[float, float]] = None,
    ) -> list[dict]:
        with self.SessionLocal() as session:
            q = select(DatasetRecord).where(
                DatasetRecord.is_labeled == False,
                DatasetRecord.is_spam == False,
            ).limit(limit)
            rows = session.execute(q).scalars().all()
            return [
                {
                    "id": str(r.id),
                    "text": r.text_raw,
                    "language": r.language,
                    "source": r.source,
                }
                for r in rows
            ]

    def save_slang_candidates(self, candidates: list[dict]) -> int:
        saved = 0
        with self.SessionLocal() as session:
            for c in candidates:
                session.add(SlangCandidateRecord(**c))
                saved += 1
            session.commit()
        return saved

"""
User-saved text history models for dashboard and analysis tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class UserSavedText(Base):
    __tablename__ = "user_saved_texts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    source_text = Column(Text, nullable=False)
    source_url = Column(String(2048), nullable=True)
    predicted_issue = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SaveTextRequest(BaseModel):
    uid: str = Field(..., description="Anonymous user UUID")
    text: str = Field(..., min_length=1, description="Highlighted text snippet")
    sourceUrl: str = Field(default="", description="URL of the page where text was saved")


class UserSavedTextResponse(BaseModel):
    id: str
    user_id: str
    source_text: str
    source_url: Optional[str] = None
    predicted_issue: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}

    @classmethod
    def model_validate(cls, obj: UserSavedText) -> "UserSavedTextResponse":
        return cls(
            id=str(obj.id),
            user_id=str(obj.user_id),
            source_text=obj.source_text,
            source_url=obj.source_url,
            predicted_issue=obj.predicted_issue,
            confidence=obj.confidence,
            created_at=obj.created_at.isoformat() if obj.created_at else None,
        )
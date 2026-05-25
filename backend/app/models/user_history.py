from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# SQLAlchemy 2.0 Declarative base
class Base(DeclarativeBase):
    pass


class UserSavedText(Base):
    __tablename__ = "user_saved_texts"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False, index=True
    )
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    predicted_issue: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, default="Normal")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<UserSavedText id={self.id!s} user_id={self.user_id!s} "
            f"predicted_issue={self.predicted_issue!r} confidence={self.confidence}>"
        )


# Pydantic models used by routes
class SaveTextRequest(BaseModel):
    uid: str
    text: str
    sourceUrl: Optional[str] = None
    # keep compatibility: some code may try to access `source_text`
    source_text: Optional[str] = None


class UserSavedTextResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source_text: str
    source_url: Optional[str] = None
    predicted_issue: Optional[str] = None
    confidence: Optional[float] = 0.0
    created_at: datetime

    model_config = {"from_attributes": True}
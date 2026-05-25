# backend/app/models/user_history.py
from sqlalchemy import Column, String, Text, Float, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

Base = declarative_base()

# 1. SQLAlchemy Model (Dùng để giao tiếp Database)
class UserSavedText(Base):
    __tablename__ = "user_saved_texts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    source_text = Column(Text, nullable=False)
    source_url = Column(Text, nullable=True)
    predicted_issue = Column(String(128))
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# 2. Pydantic Models (Dùng để Validate API Request / Response)
class SaveTextRequest(BaseModel):
    uid: str
    text: str
    sourceUrl: Optional[str] = ""

class UserSavedTextResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source_text: str
    source_url: Optional[str]
    predicted_issue: Optional[str]
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    meta: Mapped[Any | None] = mapped_column(JSON, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Pydantic DTOs ──────────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    title: str | None = None
    meta: dict | None = None


class SessionRead(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    meta: dict | None = None

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    id: uuid.UUID
    title: str | None
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str


class MessageRead(BaseModel):
    id: int
    session_id: uuid.UUID
    role: str
    content: str
    sources: list[dict] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    content: str

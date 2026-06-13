from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    category: Mapped[str] = mapped_column(String(100), default="default")
    value: Mapped[float] = mapped_column(Float, default=1.0)
    attributes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class EventRead(BaseModel):
    id: int
    lat: float
    lon: float
    timestamp: datetime
    category: str
    value: float
    attributes: dict | None = None

    model_config = {"from_attributes": True}


class EventCreate(BaseModel):
    lat: float
    lon: float
    category: str = "default"
    value: float = 1.0
    attributes: dict | None = None


class HeatmapPoint(BaseModel):
    lat: float
    lon: float
    weight: float


class StatsResponse(BaseModel):
    total: int
    by_category: dict[str, int]


class DocumentAnalysisResponse(BaseModel):
    filename: str
    content_type: str | None = None
    text: str
    text_source: str
    needs_ocr: bool
    pages: int | None = None
    warnings: list[str] = Field(default_factory=list)


class RagChunk(BaseModel):
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagEmbeddingRequest(BaseModel):
    text: str
    source_id: str = "manual"
    chunk_size: int = 900
    overlap: int = 120


class RagEmbeddingResponse(BaseModel):
    source_id: str
    model: str
    dimension: int
    chunks: list[RagChunk]
    warnings: list[str] = Field(default_factory=list)


class RagIndexResponse(BaseModel):
    source_id: str
    indexed_chunks: int
    total_chunks: int
    warnings: list[str] = Field(default_factory=list)


class RagSearchMatch(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagAskRequest(BaseModel):
    question: str
    top_k: int = 3
    source_id: str | None = None


class RagAskResponse(BaseModel):
    question: str
    answer: str
    matches: list[RagSearchMatch]
    warnings: list[str] = Field(default_factory=list)


class VisionAnalysisResponse(BaseModel):
    filename: str
    model: str
    answer: str
    warnings: list[str] = Field(default_factory=list)

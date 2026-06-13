from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from pydantic import BaseModel
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

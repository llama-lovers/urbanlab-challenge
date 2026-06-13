"""Reusable spatial query helpers using SQLAlchemy + GeoAlchemy2."""

from geoalchemy2 import Geography
from geoalchemy2.functions import (
    ST_X,
    ST_Y,
    ST_ClusterKMeans,
    ST_Distance,
    ST_DWithin,
    ST_MakePoint,
    ST_SetSRID,
)
from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import Event


def point_geo(lon: float, lat: float):
    """SQLAlchemy geography expression for a lon/lat point."""
    return cast(ST_SetSRID(ST_MakePoint(lon, lat), 4326), Geography)


async def events_within_radius(
    db: AsyncSession,
    lat: float,
    lon: float,
    radius_m: float,
) -> list:
    ref = point_geo(lon, lat)
    result = await db.execute(
        select(
            Event.id,
            ST_Y(Event.geom).label("lat"),
            ST_X(Event.geom).label("lon"),
            Event.category,
            Event.value,
            ST_Distance(cast(Event.geom, Geography), ref).label("distance_m"),
        )
        .where(ST_DWithin(cast(Event.geom, Geography), ref, radius_m))
        .order_by(ST_Distance(cast(Event.geom, Geography), ref))
    )
    return result.mappings().all()


async def cluster_events(db: AsyncSession, n_clusters: int = 5) -> list:
    """K-means clustering of event points via PostGIS ST_ClusterKMeans."""
    result = await db.execute(
        select(
            ST_ClusterKMeans(Event.geom, n_clusters).over().label("cluster_id"),
            ST_Y(Event.geom).label("lat"),
            ST_X(Event.geom).label("lon"),
            Event.category,
            Event.value,
        )
    )
    return result.mappings().all()


async def bbox_events(
    db: AsyncSession,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> list:
    """Return events inside a bounding box."""
    bbox = func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
    result = await db.execute(
        select(
            Event.id,
            ST_Y(Event.geom).label("lat"),
            ST_X(Event.geom).label("lon"),
            Event.category,
            Event.value,
        ).where(func.ST_Within(Event.geom, bbox))
    )
    return result.mappings().all()

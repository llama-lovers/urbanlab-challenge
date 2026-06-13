from fastapi import APIRouter, Depends, Query
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_X, ST_Y, ST_Distance, ST_DWithin, ST_MakePoint, ST_SetSRID
from sqlalchemy import cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.schemas import Event, HeatmapPoint

router = APIRouter()


def _make_point_geo(lon: float, lat: float):
    """Build a geography-typed point from lon/lat for distance queries."""
    return cast(ST_SetSRID(ST_MakePoint(lon, lat), 4326), Geography)


@router.get("/heatmap", response_model=list[HeatmapPoint])
async def get_heatmap(
    category: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return point data for deck.gl HeatmapLayer."""
    q = select(
        ST_Y(Event.geom).label("lat"),
        ST_X(Event.geom).label("lon"),
        Event.value.label("weight"),
    )
    if category:
        q = q.where(Event.category == category)
    result = await db.execute(q)
    return [HeatmapPoint(lat=r.lat, lon=r.lon, weight=r.weight) for r in result]


@router.get("/nearby")
async def get_nearby(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(1000.0, description="Radius in metres"),
    db: AsyncSession = Depends(get_db),
):
    """Return events within radius_m metres of (lat, lon)."""
    ref = _make_point_geo(lon, lat)
    event_geo = cast(Event.geom, Geography)

    q = (
        select(
            Event.id,
            ST_Y(Event.geom).label("lat"),
            ST_X(Event.geom).label("lon"),
            Event.category,
            Event.value,
            ST_Distance(event_geo, ref).label("distance_m"),
        )
        .where(ST_DWithin(event_geo, ref, radius_m))
        .order_by(ST_Distance(event_geo, ref))
    )
    result = await db.execute(q)
    return result.mappings().all()


@router.get("/choropleth")
async def get_choropleth(
    category: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate events per district and return a GeoJSON FeatureCollection.

    Requires a `districts` table with columns (id, name, geom).
    Load your district GeoJSON first (e.g. via ogr2ogr or pipeline/ingest.py).

    from app.models.schemas import District  # add District model when ready
    from geoalchemy2.functions import ST_Within, ST_AsGeoJSON

    q = (
        select(
            District.id,
            District.name,
            func.count(Event.id).label("event_count"),
            func.coalesce(func.sum(Event.value), 0).label("total_value"),
            ST_AsGeoJSON(District.geom).label("geometry"),
        )
        .outerjoin(Event, ST_Within(Event.geom, District.geom))
        .group_by(District.id, District.name, District.geom)
        .order_by(func.sum(Event.value).desc())
    )
    ...
    """
    return {"message": "Add a districts table and uncomment the query above."}

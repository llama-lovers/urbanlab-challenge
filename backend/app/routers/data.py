from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.schemas import Event, EventCreate, EventRead, StatsResponse

router = APIRouter()


@router.get("/", response_model=list[EventRead])
async def list_events(
    category: str | None = Query(None),
    limit: int = Query(1000, le=10000),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    q = select(
        Event.id,
        func.ST_Y(Event.geom).label("lat"),
        func.ST_X(Event.geom).label("lon"),
        Event.timestamp,
        Event.category,
        Event.value,
        Event.attributes,
    )
    if category:
        q = q.where(Event.category == category)
    q = q.offset(offset).limit(limit)
    result = await db.execute(q)
    rows = result.mappings().all()
    return [EventRead(**row) for row in rows]


@router.post("/", response_model=EventRead, status_code=201)
async def create_event(payload: EventCreate, db: AsyncSession = Depends(get_db)):
    event = Event(
        geom=f"SRID=4326;POINT({payload.lon} {payload.lat})",
        category=payload.category,
        value=payload.value,
        attributes=payload.attributes,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    row = await db.execute(
        select(
            Event.id,
            func.ST_Y(Event.geom).label("lat"),
            func.ST_X(Event.geom).label("lon"),
            Event.timestamp,
            Event.category,
            Event.value,
            Event.attributes,
        ).where(Event.id == event.id)
    )
    return EventRead(**row.mappings().one())


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_result = await db.execute(select(func.count()).select_from(Event))
    total = total_result.scalar_one()

    cat_result = await db.execute(
        select(Event.category, func.count().label("n"))
        .group_by(Event.category)
        .order_by(func.count().desc())
    )
    by_category = {row.category: row.n for row in cat_result}

    return StatsResponse(total=total, by_category=by_category)

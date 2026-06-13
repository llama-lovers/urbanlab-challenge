"""
Test fixtures — wires up an in-memory SQLite DB per test so no Postgres is needed.

Both the FastAPI `get_db` dependency and the `AsyncSessionLocal` used directly
inside the SSE generator are pointed at the same SQLite session factory.
"""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app  # side-effect: registers all ORM models with Base


@pytest.fixture(autouse=True)
async def sqlite_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db

    # Patch the module-level AsyncSessionLocal used inside the SSE generator
    import app.routers.chat as chat_module
    _orig = chat_module.AsyncSessionLocal
    chat_module.AsyncSessionLocal = factory

    yield

    chat_module.AsyncSessionLocal = _orig
    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()

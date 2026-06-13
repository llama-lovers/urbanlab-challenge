from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import chat  # noqa: F401 — registers chat ORM models with Base
from app.models import user  # noqa: F401 — registers user ORM model with Base
from app.routers import assistant
from app.routers import auth as auth_router
from app.routers import chat as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="Asystent UrbanLab Lublin",
    version="0.1.0",
    description=(
        "Chat orchestration layer for the Lublin AI Assistant — manages sessions, "
        "messages, and model response streaming."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assistant.router, prefix="/api/assistant", tags=["assistant"])
app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat_router.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health():
    return {"status": "ok"}

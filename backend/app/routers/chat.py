"""
Chat endpoints — sessions, message history, and SSE streaming.

SSE event contract (POST /sessions/{id}/messages):
    event: delta    data: {"text": "..."}         repeated per chunk
    event: sources  data: [{"title","url"}, ...]   once, if model returns citations
    event: done     data: {"message_id": int}       terminal
    event: error    data: {"detail": "..."}         on failure
"""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.limiter import limiter
from app.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatSession,
    MessageRead,
    SessionListItem,
    SessionRead,
)
from app.models.user import User
from app.services.auth import get_current_user
from app.services.model_client import DeltaChunk, SourcesChunk, get_model_client
from app.services.rag_service import get_rag_service

router = APIRouter()


def _make_title(content: str, max_len: int = 60) -> str:
    content = content.strip()
    if len(content) <= max_len:
        return content
    truncated = content[:max_len].rsplit(" ", 1)[0]
    return truncated + "…"


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(user_id=current_user.id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/sessions", response_model=list[SessionListItem])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(desc(ChatSession.updated_at))
    )
    return result.scalars().all()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def get_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    if session_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    return result.scalars().all()


@router.post(
    "/sessions/{session_id}/messages",
    response_description=(
        "SSE stream. Events: "
        "`delta` {text}, `sources` [{title,url}…], `done` {message_id}, `error` {detail}."
    ),
)
@limiter.limit("20/minute")
async def send_message(
    request: Request,
    session_id: uuid.UUID,
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    needs_title = session.title is None

    user_msg = ChatMessage(session_id=session_id, role="user", content=payload.content)
    db.add(user_msg)
    await db.commit()

    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(desc(ChatMessage.created_at))
        .limit(settings.history_limit)
    )
    messages_for_model = [
        {"role": m.role, "content": m.content}
        for m in reversed(history_result.scalars().all())
    ]

    try:
        rag_context, rag_sources = await get_rag_service().retrieve(payload.content)
    except Exception:
        rag_context, rag_sources = "", []

    if rag_context:
        augmented_messages: list[dict] = [
            {"role": "system", "content": settings.rag_system_prompt.format(context=rag_context)},
            *messages_for_model,
        ]
    else:
        augmented_messages = messages_for_model

    model_client = get_model_client()

    logger.info("RAG context retrieved: %d sources", len(rag_sources))

    async def generate():
        full_text = ""
        final_sources: list[dict] | None = rag_sources or None

        if rag_sources:
            yield f"event: sources\ndata: {json.dumps(rag_sources)}\n\n"

        try:
            async for chunk in model_client.stream(augmented_messages, str(session_id)):
                if await request.is_disconnected():
                    return
                if isinstance(chunk, DeltaChunk):
                    full_text += chunk.text
                    yield f"event: delta\ndata: {json.dumps({'text': chunk.text})}\n\n"
                elif isinstance(chunk, SourcesChunk):
                    final_sources = chunk.sources
                    yield f"event: sources\ndata: {json.dumps(chunk.sources)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        # Persist the assistant turn in a fresh session — the request-scoped `db`
        # is closed by FastAPI after the view function returns.
        assistant_id: int | None = None
        async with AsyncSessionLocal() as new_db:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_text,
                sources=final_sources,
            )
            new_db.add(assistant_msg)
            session_values: dict = {"updated_at": func.now()}
            if needs_title:
                session_values["title"] = _make_title(payload.content)
            await new_db.execute(
                update(ChatSession)
                .where(ChatSession.id == session_id)
                .values(**session_values)
            )
            await new_db.commit()
            await new_db.refresh(assistant_msg)
            assistant_id = assistant_msg.id

        yield f"event: done\ndata: {json.dumps({'message_id': assistant_id})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()

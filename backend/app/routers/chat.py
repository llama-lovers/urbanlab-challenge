"""
Chat endpoints — sessions, message history, and SSE streaming.

SSE event contract (POST /sessions/{id}/messages):
    event: delta    data: {"text": "..."}         repeated per chunk
    event: sources  data: [{"title","url"}, ...]   once, if model returns citations
    event: done     data: {"message_id": int}       terminal
    event: error    data: {"detail": "..."}         on failure
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, get_db
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

router = APIRouter()


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
async def send_message(
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
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Session not found")

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

    model_client = get_model_client()

    async def generate():
        full_text = ""
        final_sources = None

        try:
            async for chunk in model_client.stream(messages_for_model, str(session_id)):
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
            await new_db.execute(
                update(ChatSession)
                .where(ChatSession.id == session_id)
                .values(updated_at=func.now())
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

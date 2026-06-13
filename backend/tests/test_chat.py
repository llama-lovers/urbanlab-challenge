import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_session_crud():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/chat/sessions")
        assert resp.status_code == 201
        data = resp.json()
        sid = data["id"]
        assert "created_at" in data
        assert "updated_at" in data

        try:
            resp = await client.get("/api/chat/sessions")
            assert resp.status_code == 200
            assert any(s["id"] == sid for s in resp.json())

            resp = await client.get(f"/api/chat/sessions/{sid}/messages")
            assert resp.status_code == 200
            assert resp.json() == []

        finally:
            resp = await client.delete(f"/api/chat/sessions/{sid}")
            assert resp.status_code == 204

        resp = await client.delete(f"/api/chat/sessions/{sid}")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_stream():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=30
    ) as client:
        resp = await client.post("/api/chat/sessions")
        assert resp.status_code == 201
        sid = resp.json()["id"]

        try:
            event_types: list[str] = []
            done_data: dict = {}
            current_event = "message"

            async with client.stream(
                "POST",
                f"/api/chat/sessions/{sid}/messages",
                json={"content": "Jak załatwić dowód osobisty w Lublinie?"},
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers.get("content-type", "")

                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        event_types.append(current_event)
                    elif line.startswith("data:") and current_event == "done":
                        done_data = json.loads(line[5:].strip())

            assert "delta" in event_types, "expected at least one delta event"
            assert "done" in event_types, "expected a done event"
            assert isinstance(done_data.get("message_id"), int)

            resp = await client.get(f"/api/chat/sessions/{sid}/messages")
            messages = resp.json()
            assert len(messages) == 2
            roles = {m["role"] for m in messages}
            assert roles == {"user", "assistant"}
            assistant = next(m for m in messages if m["role"] == "assistant")
            assert assistant["content"].strip()

        finally:
            await client.delete(f"/api/chat/sessions/{sid}")

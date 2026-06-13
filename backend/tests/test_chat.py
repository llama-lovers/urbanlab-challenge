import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_session_crud(auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/chat/sessions", headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        sid = data["id"]
        assert "created_at" in data
        assert "updated_at" in data

        try:
            resp = await client.get("/api/chat/sessions", headers=auth_headers)
            assert resp.status_code == 200
            assert any(s["id"] == sid for s in resp.json())

            resp = await client.get(f"/api/chat/sessions/{sid}/messages", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json() == []

        finally:
            resp = await client.delete(f"/api/chat/sessions/{sid}", headers=auth_headers)
            assert resp.status_code == 204

        resp = await client.delete(f"/api/chat/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_isolation(auth_headers):
    """A user cannot access another user's sessions."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a session as user A
        resp = await client.post("/api/chat/sessions", headers=auth_headers)
        sid = resp.json()["id"]

        # Register user B and get their token
        resp = await client.post(
            "/api/auth/register",
            json={"email": "other@example.com", "password": "otherpass123"},
        )
        other_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

        # User B should not see the session
        resp = await client.get("/api/chat/sessions", headers=other_headers)
        assert not any(s["id"] == sid for s in resp.json())

        resp = await client.get(f"/api/chat/sessions/{sid}/messages", headers=other_headers)
        assert resp.status_code == 404

        resp = await client.delete(f"/api/chat/sessions/{sid}", headers=other_headers)
        assert resp.status_code == 404

        # Clean up
        await client.delete(f"/api/chat/sessions/{sid}", headers=auth_headers)


@pytest.mark.asyncio
async def test_chat_stream(auth_headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=30
    ) as client:
        resp = await client.post("/api/chat/sessions", headers=auth_headers)
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
                headers=auth_headers,
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

            resp = await client.get(f"/api/chat/sessions/{sid}/messages", headers=auth_headers)
            messages = resp.json()
            assert len(messages) == 2
            roles = {m["role"] for m in messages}
            assert roles == {"user", "assistant"}
            assistant = next(m for m in messages if m["role"] == "assistant")
            assert assistant["content"].strip()

        finally:
            await client.delete(f"/api/chat/sessions/{sid}", headers=auth_headers)


@pytest.mark.asyncio
async def test_unauthenticated_requests():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/api/chat/sessions")).status_code == 401
        assert (await client.get("/api/chat/sessions")).status_code == 401

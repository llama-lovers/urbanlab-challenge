"""
Model service adapter — single seam between the orchestration layer and the
teammates' AI model service.

Request contract (POST {MODEL_SERVICE_URL}/chat):
    Body:    {"messages": [{"role": str, "content": str}, ...], "session_id": str}
    Headers: Authorization: Bearer {MODEL_SERVICE_API_KEY}  (when key is set)

Response contract (SSE stream):
    event: delta    data: {"text": "..."}
    event: sources  data: [{"title": "...", "url": "..."}]
    event: done     data: {}

Alternative: newline-delimited JSON — {"type": "delta"|"sources", "text"|"sources": ...}
Both formats are handled by RealModelClient.

When MODEL_SERVICE_URL is empty, MockModelClient is used so the full stack can be
demoed before the model service is live.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from app.config import settings


@dataclass
class DeltaChunk:
    text: str


@dataclass
class SourcesChunk:
    sources: list[dict]


ChatChunk = DeltaChunk | SourcesChunk


class MockModelClient:
    _REPLY = (
        "Cześć! Jestem asystentem UrbanLab Lublin. "
        "Aby wyrobić dowód osobisty w Lublinie, należy udać się do Wydziału Spraw "
        "Obywatelskich przy ul. Wieniawskiej 14. "
        "Można też złożyć wniosek online przez portal gov.pl. "
        "Potrzebujesz zdjęcia biometrycznego i ważnego dokumentu tożsamości. "
        "Czas oczekiwania wynosi zazwyczaj do 30 dni."
    )
    _SOURCES: list[dict] = [
        {
            "title": "Urząd Miasta Lublin – Dowód osobisty",
            "url": "https://lublin.eu/mieszkancy/sprawy-urzedowe/dowod-osobisty/",
        },
        {
            "title": "gov.pl – Wniosek o dowód online",
            "url": "https://www.gov.pl/web/gov/uzyskaj-dowod-osobisty",
        },
    ]

    async def stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[ChatChunk]:
        for word in self._REPLY.split():
            yield DeltaChunk(text=word + " ")
            await asyncio.sleep(0.02)
        yield SourcesChunk(sources=self._SOURCES)


class RealModelClient:
    async def stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[ChatChunk]:
        headers: dict[str, str] = {"Accept": "text/event-stream"}
        if settings.model_service_api_key:
            headers["Authorization"] = f"Bearer {settings.model_service_api_key}"

        async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
            async with client.stream(
                "POST",
                f"{settings.model_service_url}/chat",
                json={"messages": messages, "session_id": session_id},
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    raw = line[5:].strip() if line.startswith("data:") else line.strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(payload, list):
                        yield SourcesChunk(sources=payload)
                    elif payload.get("type") == "delta" or "text" in payload:
                        yield DeltaChunk(text=payload.get("text", ""))
                    elif payload.get("type") == "sources" or "sources" in payload:
                        yield SourcesChunk(sources=payload.get("sources", []))


def get_model_client() -> MockModelClient | RealModelClient:
    if settings.model_service_url:
        return RealModelClient()
    return MockModelClient()

"""
Model service adapter — single seam between the orchestration layer and the AI model.

Priority order for get_model_client():
  1. OPENROUTER_API_KEY set  → OpenAICompatibleModelClient (OpenRouter cloud)
  2. MODEL_SERVICE_URL set   → RealModelClient (teammates' custom HTTP service)
  3. fallback                → MockModelClient (demo mode, no external calls)

Custom model service contract (RealModelClient):
    POST {MODEL_SERVICE_URL}/chat
    Body:    {"messages": [...], "session_id": str}
    Headers: Authorization: Bearer {MODEL_SERVICE_API_KEY}
    SSE:     event: delta   data: {"text": "..."}
             event: sources data: [{title, url}, ...]
             event: done    data: {}
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


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


class OpenAICompatibleModelClient:
    """Streams from any OpenAI-compatible /chat/completions endpoint (OpenRouter, Ollama, vLLM)."""

    async def stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[ChatChunk]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Accept": "text/event-stream",
        }
        if settings.openrouter_send_app_headers:
            headers["HTTP-Referer"] = "https://urbanlab.lublin.eu"
            headers["X-Title"] = "UrbanLab Lublin"

        body = {
            "model": settings.openrouter_model,
            "messages": messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    content = (
                        payload.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if content:
                        yield DeltaChunk(text=content)


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


class OllamaChatModelClient:
    async def stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[ChatChunk]:
        body = {
            "model": settings.chat_llm_model,
            "messages": messages,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{settings.chat_llm_base_url.rstrip('/')}/api/chat",
                    json=body,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        content = payload.get("message", {}).get("content")
                        if content:
                            yield DeltaChunk(text=content)
        except (httpx.ConnectError, httpx.HTTPStatusError):
            logger.warning("Ollama chat service unavailable; falling back to mock model")
            async for chunk in MockModelClient().stream(messages, session_id):
                yield chunk


def get_model_client() -> MockModelClient | OpenAICompatibleModelClient | RealModelClient | OllamaChatModelClient:
    if settings.openrouter_api_key:
        return OpenAICompatibleModelClient()
    if settings.chat_llm_provider == "ollama" and settings.chat_llm_base_url:
        return OllamaChatModelClient()
    if settings.model_service_url:
        return RealModelClient()
    return MockModelClient()

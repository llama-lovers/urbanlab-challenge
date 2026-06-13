"""
Model service adapter — single seam between the orchestration layer and the AI model.

Priority order for get_model_client():
  1. OPENROUTER_API_KEY set  → OpenAICompatibleModelClient (OpenRouter cloud)
  2. MODEL_SERVICE_URL set   → OpenAICompatibleModelClient (local Ollama via Docker, or any
                               OpenAI-compatible endpoint; set MODEL_SERVICE_MODEL accordingly)
  3. fallback                → MockModelClient (demo mode, no external calls)

Both live clients share the same OpenAI-compatible streaming + tool-calling implementation.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_RESERVATION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_reservation_slots",
        "description": (
            "Pobiera dostępne terminy wizyt w urzędzie miasta Lublin (system Qmatic). "
            "Użyj gdy użytkownik pyta o wolne terminy, chce umówić wizytę, "
            "lub potrzebuje informacji o dostępnych godzinach w urzędzie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": (
                        "Nazwa usługi po polsku, np. 'Dowody osobiste', "
                        "'Rejestracja Pojazdów', 'Meldunki', 'Podatki', 'Kasy'."
                    ),
                },
                "date_from": {
                    "type": "string",
                    "description": "Data od której szukać terminów (YYYY-MM-DD). Domyślnie od dzisiaj.",
                },
            },
            "required": ["service_name"],
        },
    },
}

_TOOLS = [_RESERVATION_TOOL]


async def _execute_tool(name: str, arguments_json: str) -> str:
    from app.services.reservation_service import get_reservation_slots

    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid arguments JSON"})

    if name == "get_reservation_slots":
        result = await get_reservation_slots(
            service_name=args.get("service_name", ""),
            date_from=args.get("date_from"),
        )
        return json.dumps(result, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Chunk types
# ---------------------------------------------------------------------------


@dataclass
class DeltaChunk:
    text: str


@dataclass
class SourcesChunk:
    sources: list[dict]


ChatChunk = DeltaChunk | SourcesChunk


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


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
    """
    Streams from any OpenAI-compatible /chat/completions endpoint with tool-calling support.

    Used for both:
    - OpenRouter (cloud): api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL
    - Local Ollama (Docker): api_key="" or "ollama", base_url=http://ollama:11434/v1
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        send_app_headers: bool = False,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._send_app_headers = send_app_headers

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }
        if self._send_app_headers:
            h["HTTP-Referer"] = "https://urbanlab.lublin.eu"
            h["X-Title"] = "UrbanLab Lublin"
        return h

    async def _stream_request(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        body: dict = {"model": self._model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
            async with client.stream(
                "POST", self._endpoint, json=body, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        continue

    async def stream(
        self, messages: list[dict], session_id: str
    ) -> AsyncIterator[ChatChunk]:
        # --- first pass: stream response, accumulate any tool calls ---
        tool_calls_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        assistant_text = ""

        async for payload in self._stream_request(messages, tools=_TOOLS):
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason") or finish_reason

            content = delta.get("content")
            if content:
                assistant_text += content
                yield DeltaChunk(text=content)

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                acc = tool_calls_acc[idx]
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                acc["arguments"] += fn.get("arguments", "")

        if finish_reason != "tool_calls" or not tool_calls_acc:
            return

        # --- execute tools ---
        assistant_tool_calls = [
            {
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"]},
            }
            for acc in tool_calls_acc.values()
        ]
        tool_messages: list[dict] = []
        for acc in tool_calls_acc.values():
            logger.info("Calling tool %s args=%s", acc["name"], acc["arguments"])
            result_str = await _execute_tool(acc["name"], acc["arguments"])
            logger.debug("Tool %s result: %s", acc["name"], result_str[:200])
            tool_messages.append(
                {"role": "tool", "tool_call_id": acc["id"], "content": result_str}
            )

        messages_with_result = [
            *messages,
            {"role": "assistant", "content": assistant_text or None, "tool_calls": assistant_tool_calls},
            *tool_messages,
        ]

        # --- second pass: stream final response ---
        async for payload in self._stream_request(messages_with_result):
            content = (
                (payload.get("choices") or [{}])[0].get("delta", {}).get("content")
            )
            if content:
                yield DeltaChunk(text=content)


def get_model_client() -> MockModelClient | OpenAICompatibleModelClient:
    if settings.openrouter_api_key:
        return OpenAICompatibleModelClient(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            send_app_headers=settings.openrouter_send_app_headers,
        )
    if settings.model_service_url:
        return OpenAICompatibleModelClient(
            base_url=settings.model_service_url,
            api_key=settings.model_service_api_key or "local",
            model=settings.model_service_model,
            send_app_headers=False,
        )
    return MockModelClient()

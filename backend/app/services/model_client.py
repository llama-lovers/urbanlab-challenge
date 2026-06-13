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
import re
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
                    "description": (
                        "Data od której szukać terminów (YYYY-MM-DD). Domyślnie od dzisiaj."
                    ),
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

    async def stream(self, messages: list[dict], session_id: str) -> AsyncIterator[ChatChunk]:
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

        logger.info(
            "OpenRouter request: model=%s messages=%d tools=%d",
            self._model,
            len(messages),
            len(tools) if tools else 0,
        )
        logger.debug("OpenRouter messages: %s", json.dumps(messages, ensure_ascii=False))

        async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
            async with client.stream(
                "POST", self._endpoint, json=body, headers=self._headers()
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    error_body = body_bytes.decode(errors="replace")
                    logger.error("OpenRouter error %d: %s", resp.status_code, error_body)
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {error_body}",
                        request=resp.request,
                        response=resp,
                    )
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

    async def stream(self, messages: list[dict], session_id: str) -> AsyncIterator[ChatChunk]:
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
            tool_messages.append({"role": "tool", "tool_call_id": acc["id"], "content": result_str})

        messages_with_result = [
            *messages,
            {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": assistant_tool_calls,
            },
            *tool_messages,
        ]

        # --- second pass: stream final response ---
        async for payload in self._stream_request(messages_with_result):
            content = (payload.get("choices") or [{}])[0].get("delta", {}).get("content")
            if content:
                yield DeltaChunk(text=content)


class OllamaChatModelClient:
    async def stream(self, messages: list[dict], session_id: str) -> AsyncIterator[ChatChunk]:
        body = {
            "model": settings.chat_llm_model,
            "messages": messages,
            "stream": True,
        }

        ollama_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text = next((p["text"] for p in msg["content"] if p["type"] == "text"), "")
                images = [
                    p["image_url"]["url"].split(",", 1)[1]
                    for p in msg["content"]
                    if p["type"] == "image_url"
                ]
                ollama_messages.append({"role": msg["role"], "content": text, "images": images})
            else:
                ollama_messages.append(msg)

        body["messages"] = ollama_messages

        logger.info(
            "Ollama request: model=%s messages=%d",
            settings.chat_llm_model,
            len(ollama_messages),
        )
        logger.debug("Ollama messages: %s", json.dumps(ollama_messages, ensure_ascii=False))

        try:
            async with httpx.AsyncClient(timeout=settings.model_timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{settings.chat_llm_base_url.rstrip('/')}/api/chat",
                    json=body,
                ) as resp:
                    if resp.status_code >= 400:
                        body_bytes = await resp.aread()
                        error_body = body_bytes.decode(errors="replace")
                        logger.error("Ollama error %d: %s", resp.status_code, error_body)
                        raise httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}: {error_body}",
                            request=resp.request,
                            response=resp,
                        )
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
        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.warning("Ollama chat service unavailable (%s); falling back to mock model", exc)
            async for chunk in MockModelClient().stream(messages, session_id):
                yield chunk


# ---------------------------------------------------------------------------
# Fast RAG gate — a cheap LLM call that decides whether a turn needs retrieval.
# ---------------------------------------------------------------------------

_RAG_QUERY_SYSTEM = (
    "Jesteś modułem wyszukiwania dla asystenta spraw urzędowych Miasta Lublin (baza BIP). "
    "Na podstawie historii rozmowy i OSTATNIEGO pytania użytkownika ułóż JEDNO zwięzłe, "
    "samodzielne zapytanie wyszukiwania po polsku — temat i najważniejsze słowa kluczowe. "
    "Rozwiń odniesienia z kontekstu (np. 'a ile to kosztuje?' uzupełnij o to, czego dotyczy). "
    "Całkowicie ignoruj wiadomości-śmieci, testy i liczby bez znaczenia "
    "('ping', '800+', losowe znaki). "
    "Jeśli ostatnia wiadomość to powitanie, podziękowanie, test lub nie wymaga bazy wiedzy "
    "urzędowej — odpowiedz dokładnie: NONE. "
    "Zwróć WYŁĄCZNIE samo zapytanie albo NONE — bez cudzysłowów, etykiet i wyjaśnień. /no_think"
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S).strip()


def _format_recent(history: list[dict], max_turns: int) -> str:
    lines: list[str] = []
    for m in history[-max_turns:]:
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = "Użytkownik" if m.get("role") == "user" else "Asystent"
        lines.append(f"{role}: {content.strip()[:300]}")
    return "\n".join(lines)


async def _quick_complete(messages: list[dict], max_tokens: int, timeout: float) -> str:
    if settings.openrouter_api_key:
        body = {
            "model": settings.rag_gate_model or settings.openrouter_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"].get("content") or "").strip()

    if settings.chat_llm_provider == "ollama" and settings.chat_llm_base_url:
        body = {
            "model": settings.rag_gate_model or settings.chat_llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{settings.chat_llm_base_url.rstrip('/')}/api/chat", json=body
            )
            resp.raise_for_status()
            return (resp.json().get("message", {}).get("content") or "").strip()

    # No live provider (mock/demo) — signal "no rewrite available".
    return ""


async def build_rag_query(history: list[dict], question: str) -> str | None:
    """
    Turn the conversation + latest message into ONE clean, standalone search query
    for retrieval. This is both the gate and the query builder:

      - returns None  -> the turn needs no knowledge-base lookup (skip RAG)
      - returns str   -> the rewritten query to embed (junk turns dropped,
                         follow-ups expanded with context)

    Fails open to the raw question on any error so retrieval is never lost, but
    never falls back to concatenating raw history (that poisons the embedding).
    """
    q = question.strip()
    if not q:
        return None

    convo = _format_recent(history, settings.rag_query_context_turns * 2)
    user_block = (
        f"Historia rozmowy:\n{convo}\n\n" if convo else ""
    ) + f"Ostatnie pytanie użytkownika:\n{q}\n\nZapytanie wyszukiwania:"
    messages = [
        {"role": "system", "content": _RAG_QUERY_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    try:
        raw = await _quick_complete(messages, max_tokens=64, timeout=settings.rag_gate_timeout_s)
    except Exception as exc:
        logger.warning("RAG query rewrite failed (%s); using raw question", exc)
        return q

    cleaned = _strip_think(raw).strip().strip('"').strip()
    if not cleaned:
        return q  # no live rewriter / blank reply -> search the literal question
    if cleaned.upper().startswith("NONE"):
        logger.info("RAG query: skip (no lookup) for %r", q[:60])
        return None
    logger.info("RAG query rewrite: %r -> %r", q[:60], cleaned[:90])
    return cleaned


def get_model_client() -> MockModelClient | OpenAICompatibleModelClient | OllamaChatModelClient:
    if settings.openrouter_api_key:
        return OpenAICompatibleModelClient(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            send_app_headers=settings.openrouter_send_app_headers,
        )
    if settings.chat_llm_provider == "ollama" and settings.chat_llm_base_url:
        return OllamaChatModelClient()
    if settings.model_service_url:
        return OpenAICompatibleModelClient(
            base_url=settings.model_service_url,
            api_key=settings.model_service_api_key or "local",
            model=settings.model_service_model,
            send_app_headers=False,
        )
    return MockModelClient()

# Backend for Lublin AI Assistant — Chat Orchestration Layer

## Context

Hackathon project: an AI chatbot/assistant for Lublinians, built on top of the
existing **UrbanLab Lublin** template (FastAPI + PostGIS + DuckDB + React/deck.gl).

**Team split (important):** this backend is *not* building the AI model. Two
teammates are training/serving the model separately and will expose it as an
**HTTP service**. Their service owns **retrieval (RAG) + generation** — it takes
the conversation and returns the answer. This backend is the **orchestration
layer** between the frontend and their model.

Decisions locked in:
- Model interface: **HTTP service they host** (contract not finalized → build against
  a clean adapter + mock so the backend runs now).
- RAG ownership: **their model service** (backend sends question + history, not chunks).
- Persistence: **store sessions & messages in Postgres** (already running).
- Responses: **streaming via SSE** to the frontend.
- Language: **Polish-first** (pass-through; backend stays language-agnostic, mock replies in Polish).

### What the backend is responsible for
1. CRUD for chat **sessions** and **messages**, persisted in Postgres.
2. A **streaming chat endpoint** (SSE) the frontend consumes.
3. A **ModelClient adapter** that calls the teammates' HTTP model service (with a
   mock fallback when its URL is unset).
4. Loading recent history from the DB for multi-turn context, persisting both the
   user turn and the assistant turn (incl. any `sources` the model returns).

## Existing patterns to reuse
- ORM models: `backend/app/models/schemas.py` (`Event`) — `Base`, `Mapped`,
  `mapped_column`, `JSONB`, `server_default=func.now()`.
- DB session: `backend/app/database.py` — `AsyncSessionLocal`, `get_db`, `Base`.
- Router registration + table auto-create: `backend/app/main.py`
  (`Base.metadata.create_all` in `lifespan`, `app.include_router(...)`).
- Settings: `backend/app/config.py` (`pydantic-settings`, `.env`).
- Router conventions: `backend/app/routers/data.py` (`APIRouter`, `Depends(get_db)`, response_model).

## Data model (new) — `backend/app/models/chat.py`
Two tables, auto-created by the existing `lifespan` `create_all` (must be imported
so they register on `Base` — import the module in `main.py`).

- `chat_sessions`
  - `id` UUID PK, default `gen_random_uuid()` (PG16 built-in; non-enumerable, shareable)
  - `title` `str | None` (nullable; derived from first user message later)
  - `created_at`, `updated_at` (`DateTime(timezone=True)`, server defaults; bump `updated_at` on new message)
  - `meta` `JSONB | None`
- `chat_messages`
  - `id` int PK (follow `Event` pattern)
  - `session_id` UUID FK → `chat_sessions.id` (indexed, `ondelete="CASCADE"`)
  - `role` `str` — `"user" | "assistant" | "system"`
  - `content` `Text`
  - `sources` `JSONB | None` (citations the model returns)
  - `created_at`

Pydantic DTOs in the same file: `SessionCreate`, `SessionRead`, `MessageCreate`
(`{content: str}`), `MessageRead`, `ChatRequest`.

## Endpoints (new) — `backend/app/routers/chat.py`, prefix `/api/chat`
- `POST /sessions` → create session, return `SessionRead`.
- `GET /sessions` → list sessions (id, title, updated_at), newest first.
- `GET /sessions/{id}/messages` → full message history (`list[MessageRead]`).
- `POST /sessions/{id}/messages` → **SSE stream**. Body `{content}`.
  - Persist the user message.
  - Load recent history (e.g. last 20 messages) for context.
  - Call `ModelClient.stream(...)`, relay deltas as SSE, accumulate text.
  - On completion, persist the assistant message (+ `sources`); bump `updated_at`.
- `DELETE /sessions/{id}` → delete session + cascade messages.

Register in `main.py`:
`app.include_router(chat.router, prefix="/api/chat", tags=["chat"])` and
`from app.models import chat  # noqa: F401` so tables register.

### SSE contract (backend → frontend)
`StreamingResponse(gen(), media_type="text/event-stream")`, events:
```
event: delta    data: {"text": "..."}        # repeated, token/chunk deltas
event: sources  data: [{"title": "...", "url": "..."}]   # once, if model returns citations
event: done     data: {"message_id": 123}     # terminal
event: error    data: {"detail": "..."}       # on failure
```
**Gotcha:** a `Depends(get_db)` session is closed before the streaming body runs.
Inside the SSE generator, open a fresh `AsyncSessionLocal()` to persist the
assistant turn — do not reuse the request-scoped session.

## Model client adapter — `backend/app/services/model_client.py`
Async, `httpx`-based. Single method:
`async def stream(messages: list[dict], session_id: str) -> AsyncIterator[ChatChunk]`
where `ChatChunk` is either a text delta or a sources payload.

- **Real client** (when `MODEL_SERVICE_URL` is set): `POST {MODEL_SERVICE_URL}/chat`
  with `{"messages": [{"role","content"}...], "session_id": "..."}`, consume the
  streamed response (`resp.aiter_lines()` for SSE, or `aiter_text()` for chunked
  text), yield deltas; parse a trailing/`sources` event if present. Configurable
  timeout; optional bearer token (`MODEL_SERVICE_API_KEY`).
- **Mock client** (when `MODEL_SERVICE_URL` is empty): yields a canned **Polish**
  answer word-by-word with small `asyncio.sleep`, plus a sample `sources` payload.
  Lets the whole backend + frontend be demoed before the model is live.

This is the single seam to swap when teammates finalize their contract. Define the
contract explicitly in a docstring and share it with them (request shape above;
response = SSE `delta`/`sources` events, or newline-delimited JSON — adapter handles both).

## Config & deps
- `backend/app/config.py` — add:
  - `model_service_url: str = ""` (empty → mock)
  - `model_service_api_key: str = ""`
  - `model_timeout_s: float = 60.0`
  - `history_limit: int = 20`
- `backend/pyproject.toml` — add `httpx>=0.27` to main `dependencies` (currently dev-only).
- `.env.example` + `docker-compose.yml` (backend `environment:`) — add
  `MODEL_SERVICE_URL`, `MODEL_SERVICE_API_KEY`.

## OpenAPI contract export
FastAPI already serves the **live** spec at `GET /openapi.json` and Swagger UI at
`/docs`. To give teammates (frontend + model service) a stable, committable
contract, add a script that dumps the spec to a static file.

- **new** `backend/scripts/export_openapi.py` — imports `app.main.app`, calls
  `app.openapi()`, writes `backend/openapi.json` (pretty-printed). Works fully
  offline: importing the app does **not** open a DB connection (the engine is lazy;
  `lifespan`/`create_all` only run under uvicorn), so no Postgres needed to export.
- **edit** `Taskfile.yml` — add a `task openapi` target:
  `cd backend && python scripts/export_openapi.py`.
- Commit `backend/openapi.json`; regenerate via `task openapi` whenever endpoints
  change. Set a descriptive `title`/`version`/`description` on the `FastAPI(...)`
  app so the exported contract is self-documenting (rename from the template's
  "Hackathon API").
- The chat request/response models and SSE event shapes are documented via the
  Pydantic DTOs, so they surface in the spec automatically. (SSE bodies aren't
  fully expressible in OpenAPI — annotate the streaming endpoint's response with a
  description of the `delta`/`sources`/`done`/`error` events.)

## Files to create / modify
- **new** `backend/app/models/chat.py` — ORM models + Pydantic DTOs
- **new** `backend/app/routers/chat.py` — endpoints + SSE generator
- **new** `backend/app/services/model_client.py` — adapter + mock
- **new** `backend/scripts/export_openapi.py` — dump OpenAPI spec → `backend/openapi.json`
- **new** `backend/tests/test_chat.py` — session CRUD + streamed turn (mock model)
- **edit** `backend/app/main.py` — import chat models, include chat router, set app title/version/description
- **edit** `backend/app/config.py` — model-service settings
- **edit** `backend/pyproject.toml` — add `httpx`
- **edit** `Taskfile.yml` — `task openapi` target
- **edit** `.env.example`, `docker-compose.yml` — model-service env vars
- **generated** `backend/openapi.json` — committed API contract

## Verification
1. `task run` (or `docker compose up`) — Postgres + backend come up.
2. Open `http://localhost:8000/docs` — chat endpoints present.
3. End-to-end with the **mock** model (no `MODEL_SERVICE_URL`):
   ```bash
   SID=$(curl -s -XPOST localhost:8000/api/chat/sessions | jq -r .id)
   curl -N -XPOST localhost:8000/api/chat/sessions/$SID/messages \
     -H 'content-type: application/json' -d '{"content":"Jak załatwić dowód osobisty w Lublinie?"}'
   ```
   Expect streamed `event: delta` lines (Polish), then `sources`, then `done`.
4. `task db` → `SELECT * FROM chat_sessions; SELECT role, content FROM chat_messages;`
   — both turns persisted.
5. `cd backend && pytest` — `test_chat.py` green (uses mock model).
6. `task openapi` → writes `backend/openapi.json`; confirm chat paths/schemas
   present (`jq '.paths | keys' backend/openapi.json`). Also reachable live at
   `http://localhost:8000/openapi.json` and `http://localhost:8000/docs`.
7. When teammates' service is ready: set `MODEL_SERVICE_URL`, re-run step 3 against it.

## Out of scope (note for later)
- Frontend chat UI (separate task; SSE contract above is the integration point).
- Auth / rate limiting.
- The AI model + retrieval (owned by teammates).
- Title auto-generation, message editing, streaming cancellation — easy follow-ups.

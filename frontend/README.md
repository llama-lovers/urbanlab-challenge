# UrbanLab Document Assistant (frontend)

A chat UI ([assistant-ui](https://github.com/assistant-ui/assistant-ui) + Next.js) for asking
questions about your documents. The user asks a question (optionally attaching PDF / DOCX / CSV /
TXT / Markdown files); the **FastAPI backend** parses the documents, calls a **local Ollama**
model, and streams the answer back token by token. This app is UI only — all parsing and the LLM
call happen in the backend.

## Getting Started

1. **Run Ollama** locally and pull a model:

   ```bash
   ollama serve
   ollama pull llama3.1          # or set OLLAMA_MODEL in the backend
   ```

2. **Run the backend** (from the repo root) so it's reachable at http://localhost:8000:

   ```bash
   docker compose up backend postgres   # or run uvicorn locally
   ```

3. **Configure & run the frontend:**

   ```bash
   cp .env.example .env.local            # NEXT_PUBLIC_API_URL=http://localhost:8000
   npm install
   npm run dev
   ```

Open [http://localhost:3000](http://localhost:3000). Type a question, attach documents with the
**+** button, and the streamed answer appears in the thread.

## How it works

- `lib/chat-adapter.ts` — a custom assistant-ui `ChatModelAdapter` that POSTs the conversation +
  attached files to `POST {NEXT_PUBLIC_API_URL}/api/chat` and renders the streamed text.
- `lib/document-attachment-adapter.ts` — a pass-through attachment adapter; it keeps the raw
  `File` so the chat adapter can upload it (the backend does the parsing).
- `app/assistant.tsx` — wires both adapters into `useLocalRuntime`.

The backend endpoint lives in `backend/app/routers/chat.py` (parsing in
`backend/app/services/documents.py`, Ollama streaming in `backend/app/services/ollama.py`).

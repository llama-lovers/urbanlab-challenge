"""
RAG context retrieval for the chat pipeline.

RagService embeds the user question, searches Qdrant, and returns:
  - a formatted context string ready to be injected into the system prompt
  - a sources list ready to be emitted as an SSE `sources` event
"""
from __future__ import annotations

import logging

from app.config import settings
from app.services.document_ai import EmbeddingService, RerankerService
from app.services.qdrant_store import QdrantRagStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self) -> None:
        embedding_service = EmbeddingService()
        reranker_service = RerankerService()
        self._store = QdrantRagStore(embedding_service, reranker_service)

    async def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[str, list[dict]]:
        """
        Returns (context_text, sse_sources).
        Both are empty when Qdrant has no relevant results or is unreachable.
        """
        k = top_k or settings.rag_top_k
        logger.info("RAG retrieve: question=%.80r top_k=%d", question, k)
        matches, warnings = await self._store.search(question, top_k=k)

        if warnings:
            for w in warnings:
                logger.warning("RAG warning: %s", w)

        if not matches:
            logger.warning("RAG retrieve: no matches found for question %.80r", question)
            return "", []

        parts = []
        for i, m in enumerate(matches, 1):
            label = m.metadata.get("url") or m.metadata.get("source_id") or ""
            parts.append(f"[{i}] {label}\n{m.text}" if label else f"[{i}]\n{m.text}")

        context = "\n\n---\n\n".join(parts)
        logger.info("RAG context built: %d sources, %d chars", len(matches), len(context))

        sources = [
            {
                "title": m.metadata.get("title") or m.metadata.get("source_id") or f"Wynik {i}",
                "url": m.metadata.get("url", ""),
            }
            for i, m in enumerate(matches, 1)
        ]

        return context, sources


_instance: RagService | None = None


def get_rag_service() -> RagService:
    global _instance
    if _instance is None:
        _instance = RagService()
    return _instance

"""
Qdrant-backed vector store.

Expected payload per point (set by the pipeline or the assistant indexing endpoint):
    text        str  — chunk text
    source_id   str  — logical source name, e.g. "bip_uslugi"
    url         str  — canonical source URL (optional)
    title       str  — page or document title (optional)
    chunk_index int  — position within the source document (optional)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.services.document_ai import EmbeddingService, RerankerService, SearchMatch

logger = logging.getLogger(__name__)


def _to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantRagStore:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._reranker_service = reranker_service
        self._client: AsyncQdrantClient | None = None

    async def get_total_chunks(self) -> int:
        try:
            client = self._get_client()
            info = await client.get_collection(settings.qdrant_collection)
            return info.points_count or 0
        except Exception:
            return 0

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            logger.info("Connecting to Qdrant at %s, collection=%s", settings.qdrant_url, settings.qdrant_collection)
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=10,
            )
        return self._client

    async def _ensure_collection(self) -> None:
        client = self._get_client()
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}
        if settings.qdrant_collection not in existing:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

    async def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        client = self._get_client()
        await self._ensure_collection()

        points = [
            PointStruct(
                id=_to_uuid(chunk["id"]),
                vector=chunk["embedding"],
                payload={"text": chunk["text"], **chunk.get("metadata", {})},
            )
            for chunk in chunks
        ]
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        return len(points)

    async def search(
        self,
        question: str,
        top_k: int,
        source_id: str | None = None,
    ) -> tuple[list[SearchMatch], list[str]]:
        client = self._get_client()

        logger.debug("Embedding query for Qdrant search: %.80r", question)
        query_vector = self._embedding_service.embed_texts([question])[0]

        query_filter: Filter | None = None
        if source_id:
            query_filter = Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            )

        candidate_limit = (
            top_k * max(1, settings.reranker_candidate_multiplier)
            if self._reranker_service
            else top_k
        )

        logger.debug(
            "Querying Qdrant collection=%s limit=%d filter=%s",
            settings.qdrant_collection, candidate_limit, source_id,
        )
        try:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_vector,
                limit=candidate_limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant search failed (collection=%s url=%s): %s", settings.qdrant_collection, settings.qdrant_url, exc)
            return [], [f"Qdrant search failed: {exc}"]

        logger.info("Qdrant returned %d candidates for query %.60r", len(response.points), question)

        matches = [
            SearchMatch(
                id=str(r.id),
                text=r.payload.get("text", ""),
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k != "text"},
            )
            for r in response.points
        ]

        warnings: list[str] = []
        if self._reranker_service and matches:
            matches = self._reranker_service.rerank(question, matches, top_k=top_k)
            if self._reranker_service.warning:
                warnings.append(self._reranker_service.warning)
        else:
            matches = matches[:top_k]

        logger.info("RAG search done: %d matches after rerank/trim (top_k=%d)", len(matches), top_k)
        if matches:
            logger.debug("Top match score=%.4f text=%.80r", matches[0].score, matches[0].text)

        return matches, warnings

    def answer(self, question: str, matches: list[SearchMatch]) -> str:
        if not matches:
            return (
                "Nie mam jeszcze zindeksowanych dokumentów pasujących do pytania. "
                "Najpierw użyj endpointu /api/assistant/documents/index."
            )
        best = matches[0]
        preview = best.metadata.get("document_text_preview")
        if isinstance(preview, str) and preview:
            return (
                f"Najbardziej pasujący dokument po tytule: {best.text}. "
                f"Podgląd treści dokumentu: {preview[:700]}"
            )
        return f"Najbardziej pasujący dokument po tytule: {best.text[:900]}"

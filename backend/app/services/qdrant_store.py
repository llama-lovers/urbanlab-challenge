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

import asyncio
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
            logger.info(
                "Connecting to Qdrant at %s, collection=%s",
                settings.qdrant_url,
                settings.qdrant_collection,
            )
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
            settings.qdrant_collection,
            candidate_limit,
            source_id,
        )
        try:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_vector,
                limit=candidate_limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=settings.rag_min_retrieval_score or None,
            )
        except Exception as exc:
            logger.error(
                "Qdrant search failed (collection=%s url=%s): %s",
                settings.qdrant_collection,
                settings.qdrant_url,
                exc,
            )
            return [], [f"Qdrant search failed: {exc}"]

        logger.info(
            "Qdrant returned %d candidates (min_score=%.3f) for query %.60r",
            len(response.points),
            settings.rag_min_retrieval_score,
            question,
        )

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
        reranked = False
        if self._reranker_service and matches:
            matches = self._reranker_service.rerank(question, matches, top_k=top_k)
            if self._reranker_service.warning:
                warnings.append(self._reranker_service.warning)
            else:
                reranked = True
        else:
            matches = matches[:top_k]

        # Optional relevance gate on the cross-encoder score. Only meaningful when the
        # reranker actually ran (otherwise `.score` is the cosine, already gated by Qdrant).
        if reranked and settings.rag_min_rerank_score > 0 and matches:
            before = len(matches)
            matches = [m for m in matches if m.score >= settings.rag_min_rerank_score]
            if len(matches) != before:
                logger.info(
                    "Reranker gate dropped %d/%d matches below %.3f",
                    before - len(matches),
                    before,
                    settings.rag_min_rerank_score,
                )

        logger.info("RAG search done: %d matches after rerank/trim (top_k=%d)", len(matches), top_k)
        if matches:
            logger.debug("Top match score=%.4f text=%.80r", matches[0].score, matches[0].text)

        return matches, warnings

    async def expand_to_pages(self, matches: list[SearchMatch]) -> list[SearchMatch]:
        """
        Each match is a paragraph-level node. Expand every matched paragraph to its
        full parent page — pull all sibling chunks that share the page key — but keep
        each paragraph as its OWN match (ordered by chunk_index) so the model sees the
        whole page AND every paragraph carries its own source (url/title).

        Pages are processed best-match-first; paragraphs are de-duplicated by id.
        Matches without a page key are kept as-is.
        """
        if not matches:
            return matches

        client = self._get_client()
        page_key = settings.rag_page_key

        # Unique pages, best-match-first; remember the representative match per page.
        seen_pages: set[str] = set()
        plan: list[tuple[SearchMatch, str | None, str | None]] = []
        for m in matches:
            field = page_key if m.metadata.get(page_key) else "source_id"
            page_value = m.metadata.get(field)
            if not page_value:
                plan.append((m, None, None))
                continue
            if str(page_value) in seen_pages:
                continue
            seen_pages.add(str(page_value))
            plan.append((m, field, str(page_value)))

        # Fetch every page's chunks concurrently — sequential scrolls were the bottleneck.
        fetches = await asyncio.gather(
            *(
                self._fetch_page_chunks(client, field, value)
                if value is not None
                else asyncio.sleep(0, result=[])
                for _, field, value in plan
            )
        )

        expanded: list[SearchMatch] = []
        seen_ids: set[str] = set()
        for (rep, _field, _value), siblings in zip(plan, fetches, strict=False):
            if not siblings:
                if rep.id not in seen_ids:
                    seen_ids.add(rep.id)
                    expanded.append(rep)
                continue
            siblings.sort(key=lambda p: p.payload.get("chunk_index") or 0)
            for p in siblings:
                pid = str(p.id)
                text = p.payload.get("text", "")
                if pid in seen_ids or not text:
                    continue
                seen_ids.add(pid)
                expanded.append(
                    SearchMatch(
                        id=pid,
                        text=text,
                        score=rep.score,
                        metadata={k: v for k, v in p.payload.items() if k != "text"},
                    )
                )

        logger.info(
            "Parent expansion: %d matches -> %d paragraphs across %d pages",
            len(matches),
            len(expanded),
            len(seen_pages),
        )
        return expanded

    async def _fetch_page_chunks(self, client: AsyncQdrantClient, field: str, value: str) -> list:
        flt = Filter(must=[FieldCondition(key=field, match=MatchValue(value=value))])
        collected: list = []
        offset = None
        while len(collected) < settings.rag_max_page_chunks:
            points, offset = await client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=min(256, settings.rag_max_page_chunks - len(collected)),
                offset=offset,
                with_payload=True,
            )
            collected.extend(points)
            if offset is None:
                break
        return collected

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

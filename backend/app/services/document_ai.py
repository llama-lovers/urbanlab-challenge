from __future__ import annotations

import base64
import hashlib
import io
import math
import re
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import settings


@dataclass(slots=True)
class ExtractedDocument:
    text: str
    text_source: str
    needs_ocr: bool
    pages: int | None
    warnings: list[str]


@dataclass(slots=True)
class VisionResult:
    answer: str
    model: str
    warnings: list[str]


@dataclass(slots=True)
class SearchMatch:
    id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | None]


class DocumentProcessor:
    def extract_text(self, content: bytes, filename: str, content_type: str | None) -> ExtractedDocument:
        warnings: list[str] = []
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if self._is_pdf(content_type, suffix):
            text, pages, pdf_warnings = self._extract_pdf_text(content)
            warnings.extend(pdf_warnings)
            if text.strip():
                return ExtractedDocument(
                    text=text,
                    text_source="pdf_text_layer",
                    needs_ocr=False,
                    pages=pages,
                    warnings=warnings,
                )

            ocr_text, ocr_warnings = self._ocr_pdf(content)
            warnings.extend(ocr_warnings)
            return ExtractedDocument(
                text=ocr_text,
                text_source="ocr" if ocr_text.strip() else "unavailable",
                needs_ocr=True,
                pages=pages,
                warnings=warnings,
            )

        if self._is_image(content_type, suffix):
            ocr_text, ocr_warnings = self._ocr_image(content)
            warnings.extend(ocr_warnings)
            return ExtractedDocument(
                text=ocr_text,
                text_source="ocr" if ocr_text.strip() else "unavailable",
                needs_ocr=True,
                pages=1,
                warnings=warnings,
            )

        decoded = self._decode_text(content)
        return ExtractedDocument(
            text=decoded,
            text_source="plain_text" if decoded.strip() else "unavailable",
            needs_ocr=False,
            pages=None,
            warnings=warnings,
        )

    def _extract_pdf_text(self, content: bytes) -> tuple[str, int | None, list[str]]:
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", None, ["pypdf is not installed; PDF text layer extraction skipped"]

        try:
            reader = PdfReader(io.BytesIO(content))
            pages = len(reader.pages)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip(), pages, []
        except Exception as exc:  # pragma: no cover - depends on malformed PDFs
            return "", None, [f"PDF text extraction failed: {exc}"]

    def _ocr_pdf(self, content: bytes) -> tuple[str, list[str]]:
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            return "", ["pdf2image is not installed; scanned PDF OCR skipped"]

        warnings: list[str] = []
        try:
            images = convert_from_bytes(content)
        except Exception as exc:  # pragma: no cover - depends on system poppler
            return "", [f"PDF rasterization for OCR failed: {exc}"]

        texts: list[str] = []
        for index, image in enumerate(images, start=1):
            text, image_warnings = self._ocr_pil_image(image)
            warnings.extend(f"page {index}: {warning}" for warning in image_warnings)
            texts.append(text)
        return "\n\n".join(part for part in texts if part.strip()), warnings

    def _ocr_image(self, content: bytes) -> tuple[str, list[str]]:
        try:
            image = Image.open(io.BytesIO(content))
        except UnidentifiedImageError:
            return "", ["Uploaded file is not a valid image"]
        return self._ocr_pil_image(image)

    def _ocr_pil_image(self, image: Image.Image) -> tuple[str, list[str]]:
        try:
            import pytesseract
        except ImportError:
            return "", ["pytesseract is not installed; OCR skipped"]

        try:
            return pytesseract.image_to_string(image, lang=settings.ocr_language).strip(), []
        except Exception as exc:  # pragma: no cover - depends on system tesseract
            return "", [f"OCR failed: {exc}"]

    def _decode_text(self, content: bytes) -> str:
        for encoding in ("utf-8", "cp1250", "iso-8859-2"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return ""

    def _is_pdf(self, content_type: str | None, suffix: str) -> bool:
        return content_type == "application/pdf" or suffix == "pdf"

    def _is_image(self, content_type: str | None, suffix: str) -> bool:
        return (content_type or "").startswith("image/") or suffix in {"jpg", "jpeg", "png", "webp", "tif", "tiff"}


class EmbeddingService:
    def __init__(self) -> None:
        self._load_warning: str | None = None

    @property
    def model_name(self) -> str:
        return settings.embedding_model if settings.embedding_service_url else "hashing-fallback"

    @property
    def dimension(self) -> int:
        return settings.embedding_dimension

    def embed_chunks(
        self,
        text: str,
        source_id: str,
        chunk_size: int,
        overlap: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        chunks = self._chunk_text(text, chunk_size=max(100, chunk_size), overlap=max(0, overlap))
        warnings = [self._load_warning] if self._load_warning else []
        vectors = self.embed_texts(chunks)
        result = [
            {
                "id": f"{source_id}:{index}",
                "text": chunk,
                "embedding": vector,
                "metadata": {"source_id": source_id, "chunk_index": index},
            }
            for index, (chunk, vector) in enumerate(zip(chunks, vectors), start=1)
        ]
        return result, warnings

    def embed_document_title(self, filename: str, document_text: str) -> tuple[list[dict[str, Any]], list[str]]:
        title = self._document_title(filename, document_text)
        warnings = [self._load_warning] if self._load_warning else []
        vector = self.embed_texts([title])[0]
        chunk = {
            "id": f"{filename}:title",
            "text": title,
            "embedding": vector,
            "metadata": {
                "source_id": filename,
                "chunk_index": 1,
                "embedding_basis": "document_title",
                "title": title,
                "document_text_preview": " ".join(document_text.split())[:1200],
            },
        }
        return [chunk], warnings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if settings.embedding_service_url:
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        f"{settings.embedding_service_url.rstrip('/')}/embed",
                        json={"texts": texts, "normalize": True},
                    )
                    response.raise_for_status()
                self._load_warning = None
                return response.json()["embeddings"]
            except Exception as exc:  # pragma: no cover - depends on model service availability
                self._load_warning = f"Embedding service unavailable, using hashing fallback: {exc}"

        return [self._hash_embedding(text) for text in texts]

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        effective_overlap = min(overlap, chunk_size // 2)
        while start < len(normalized):
            previous_start = start
            end = min(len(normalized), start + chunk_size)
            if end < len(normalized):
                split_at = normalized.rfind(" ", start, end)
                if split_at > start + chunk_size // 2:
                    end = split_at
            chunks.append(normalized[start:end].strip())
            if end >= len(normalized):
                break
            start = max(end - effective_overlap, 0)
            if start <= previous_start:
                start = end
        return [chunk for chunk in chunks if chunk]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * settings.embedding_dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % settings.embedding_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _document_title(self, filename: str, document_text: str) -> str:
        filename_title = self._filename_title(filename)
        heading = self._first_heading(document_text)
        if heading and heading.lower() not in filename_title.lower():
            return f"{filename_title} - {heading}"
        return filename_title

    def _filename_title(self, filename: str) -> str:
        stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        title = re.sub(r"[-_]+", " ", stem)
        title = re.sub(r"\s+", " ", title).strip()
        return title or filename

    def _first_heading(self, document_text: str) -> str | None:
        for raw_line in document_text.splitlines():
            line = re.sub(r"[._]{4,}", " ", raw_line)
            line = re.sub(r"\s+", " ", line).strip(" :;-")
            if not 8 <= len(line) <= 180:
                continue
            lower = line.lower()
            if lower.startswith(("część ", "pola ", "wypełniaj ", "dane ", "adres ")):
                continue
            if sum(char.isalpha() for char in line) < 6:
                continue
            return line
        return None


class RerankerService:
    def __init__(self) -> None:
        self._load_warning: str | None = None

    @property
    def model_name(self) -> str:
        return settings.reranker_model if settings.reranker_enabled and settings.reranker_service_url else "disabled"

    @property
    def warning(self) -> str | None:
        return self._load_warning

    def rerank(self, question: str, matches: list[SearchMatch], top_k: int) -> list[SearchMatch]:
        if not settings.reranker_enabled or not settings.reranker_service_url or not matches:
            return matches[:top_k]

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{settings.reranker_service_url.rstrip('/')}/rerank",
                    json={"pairs": [{"query": question, "text": match.text} for match in matches]},
                )
                response.raise_for_status()
            scores = response.json()["scores"]
            self._load_warning = None
        except Exception as exc:  # pragma: no cover - depends on model service availability
            self._load_warning = f"Reranker service unavailable, using embedding similarity order: {exc}"
            return matches[:top_k]

        scored_matches = [
            SearchMatch(
                id=match.id,
                text=match.text,
                score=float(score),
                metadata={**match.metadata, "retrieval_score": match.score, "reranker_model": settings.reranker_model},
            )
            for match, score in zip(matches, scores)
        ]
        scored_matches.sort(key=lambda match: match.score, reverse=True)
        return scored_matches[:top_k]


class InMemoryRagStore:
    def __init__(self, embedding_service: EmbeddingService, reranker_service: RerankerService | None = None) -> None:
        self._embedding_service = embedding_service
        self._reranker_service = reranker_service
        self._chunks: dict[str, dict[str, Any]] = {}

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        for chunk in chunks:
            self._chunks[chunk["id"]] = chunk
        return len(chunks)

    def search(self, question: str, top_k: int, source_id: str | None = None) -> tuple[list[SearchMatch], list[str]]:
        if not self._chunks:
            return [], []

        query_vector = self._embedding_service.embed_texts([question])[0]
        matches: list[SearchMatch] = []
        for chunk in self._chunks.values():
            metadata = chunk.get("metadata", {})
            if source_id and metadata.get("source_id") != source_id:
                continue

            score = self._cosine_similarity(query_vector, chunk["embedding"])
            matches.append(
                SearchMatch(
                    id=chunk["id"],
                    text=chunk["text"],
                    score=score,
                    metadata=metadata,
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        candidate_limit = max(top_k, top_k * max(1, settings.reranker_candidate_multiplier))
        candidates = matches[:candidate_limit]

        warnings: list[str] = []
        if self._reranker_service is not None:
            candidates = self._reranker_service.rerank(question, candidates, top_k=max(1, top_k))
            if self._reranker_service.warning:
                warnings.append(self._reranker_service.warning)
        else:
            candidates = candidates[: max(1, top_k)]
        return candidates, warnings

    def answer(self, question: str, matches: list[SearchMatch]) -> str:
        if not matches:
            return (
                "Nie mam jeszcze zindeksowanych dokumentów pasujących do pytania. "
                "Najpierw użyj endpointu /api/assistant/documents/index."
            )

        best = matches[0]
        document_preview = best.metadata.get("document_text_preview")
        if isinstance(document_preview, str) and document_preview:
            return (
                f"Najbardziej pasujący dokument po tytule: {best.text}. "
                f"Podgląd treści dokumentu: {document_preview[:700]}"
            )
        return (
            "Najbardziej pasujący dokument po tytule: "
            f"{best.text[:900]}"
        )

    def _cosine_similarity(self, first: list[float], second: list[float]) -> float:
        dot = sum(a * b for a, b in zip(first, second))
        first_norm = math.sqrt(sum(value * value for value in first)) or 1.0
        second_norm = math.sqrt(sum(value * value for value in second)) or 1.0
        return dot / (first_norm * second_norm)


class VisionLLMService:
    async def analyze(self, content: bytes, filename: str, content_type: str | None, prompt: str) -> VisionResult:
        image_content, mime_type, warnings = self._prepare_image_input(content, filename, content_type)

        if settings.vision_llm_provider == "ollama":
            return await self._analyze_with_ollama(image_content, prompt, warnings)

        return await self._analyze_with_openai_compatible(image_content, mime_type, prompt, warnings)

    async def _analyze_with_ollama(
        self,
        image_content: bytes,
        prompt: str,
        warnings: list[str],
    ) -> VisionResult:
        base_url = settings.vision_llm_base_url or "http://host.docker.internal:11434"
        payload = {
            "model": settings.vision_llm_model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_content).decode("ascii")],
                }
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            return VisionResult(
                answer=(
                    "Local VisionLLM is not running. Start Ollama locally and pull a vision model, "
                    f"for example: ollama pull {settings.vision_llm_model}."
                ),
                model=settings.vision_llm_model,
                warnings=[*warnings, "Ollama is not reachable"],
            )
        except httpx.HTTPStatusError as exc:
            return VisionResult(
                answer=f"Ollama returned an error: {exc.response.text}",
                model=settings.vision_llm_model,
                warnings=[*warnings, f"Ollama request failed with HTTP {exc.response.status_code}"],
            )

        data = response.json()
        return VisionResult(
            answer=data.get("message", {}).get("content", ""),
            model=settings.vision_llm_model,
            warnings=warnings,
        )

    async def _analyze_with_openai_compatible(
        self,
        image_content: bytes,
        mime_type: str,
        prompt: str,
        warnings: list[str],
    ) -> VisionResult:
        api_key = settings.resolved_vision_llm_api_key
        base_url = settings.vision_llm_base_url or "https://api.openai.com/v1"
        if not api_key:
            return VisionResult(
                answer=(
                    "VisionLLM is not configured. Set OPENAI_API_KEY or VISION_LLM_API_KEY "
                    "to enable multimodal document analysis."
                ),
                model=settings.vision_llm_model,
                warnings=["VisionLLM API key is not configured"],
            )

        image_url = f"data:{mime_type};base64,{base64.b64encode(image_content).decode('ascii')}"
        payload = {
            "model": settings.vision_llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                return VisionResult(
                    answer=f"VisionLLM provider returned an error: {exc.response.text}",
                    model=settings.vision_llm_model,
                    warnings=[*warnings, f"VisionLLM request failed with HTTP {exc.response.status_code}"],
                )

        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return VisionResult(answer=answer, model=settings.vision_llm_model, warnings=warnings)

    def _prepare_image_input(
        self,
        content: bytes,
        filename: str,
        content_type: str | None,
    ) -> tuple[bytes, str, list[str]]:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if content_type == "application/pdf" or suffix == "pdf":
            try:
                from pdf2image import convert_from_bytes

                images = convert_from_bytes(content, first_page=1, last_page=1)
                if not images:
                    return content, "application/pdf", ["PDF preview conversion returned no pages"]
                output = io.BytesIO()
                images[0].convert("RGB").save(output, format="JPEG", quality=90)
                return output.getvalue(), "image/jpeg", ["PDF was converted to first-page image for VisionLLM"]
            except Exception as exc:  # pragma: no cover - depends on system poppler
                return content, "application/pdf", [f"PDF preview conversion failed: {exc}"]

        mime_type = content_type or self._guess_mime_type(filename)
        return content, mime_type, []

    def _guess_mime_type(self, filename: str) -> str:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(suffix, "application/octet-stream")

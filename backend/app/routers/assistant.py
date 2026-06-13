from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import (
    DocumentAnalysisResponse,
    RagAskRequest,
    RagAskResponse,
    RagChunk,
    RagEmbeddingRequest,
    RagEmbeddingResponse,
    RagIndexResponse,
    RagSearchMatch,
    VisionAnalysisResponse,
)
from app.services.document_ai import (
    DocumentProcessor,
    EmbeddingService,
    InMemoryRagStore,
    RerankerService,
    VisionLLMService,
)

router = APIRouter()

document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
reranker_service = RerankerService()
rag_store = InMemoryRagStore(embedding_service, reranker_service)
vision_service = VisionLLMService()


@router.post("/documents/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document = document_processor.extract_text(content, file.filename or "document", file.content_type)
    return DocumentAnalysisResponse(
        filename=file.filename or "document",
        content_type=file.content_type,
        text=document.text,
        text_source=document.text_source,
        needs_ocr=document.needs_ocr,
        pages=document.pages,
        warnings=document.warnings,
    )


@router.post("/rag/embeddings", response_model=RagEmbeddingResponse)
async def create_rag_embeddings(payload: RagEmbeddingRequest):
    chunks, warnings = embedding_service.embed_chunks(
        text=payload.text,
        source_id=payload.source_id,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )
    return RagEmbeddingResponse(
        source_id=payload.source_id,
        model=embedding_service.model_name,
        dimension=embedding_service.dimension,
        chunks=[RagChunk(**chunk) for chunk in chunks],
        warnings=warnings,
    )


@router.post("/documents/prepare-rag", response_model=RagEmbeddingResponse)
async def prepare_document_for_rag(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document = document_processor.extract_text(content, file.filename or "document", file.content_type)
    if not document.text.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No text could be extracted from the document",
                "warnings": document.warnings,
            },
    )

    source_id = file.filename or "document"
    chunks, warnings = embedding_service.embed_document_title(source_id, document.text)
    return RagEmbeddingResponse(
        source_id=source_id,
        model=embedding_service.model_name,
        dimension=embedding_service.dimension,
        chunks=[RagChunk(**chunk) for chunk in chunks],
        warnings=[*document.warnings, *warnings],
    )


@router.post("/documents/index", response_model=RagIndexResponse)
async def index_document_for_rag(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document = document_processor.extract_text(content, file.filename or "document", file.content_type)
    if not document.text.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No text could be extracted from the document",
                "warnings": document.warnings,
            },
        )

    source_id = file.filename or "document"
    chunks, warnings = embedding_service.embed_document_title(source_id, document.text)
    indexed_chunks = rag_store.index_chunks(chunks)
    return RagIndexResponse(
        source_id=source_id,
        indexed_chunks=indexed_chunks,
        total_chunks=rag_store.total_chunks,
        warnings=[*document.warnings, *warnings],
    )


@router.post("/ask", response_model=RagAskResponse)
async def ask_assistant(payload: RagAskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    matches, search_warnings = rag_store.search(
        question=payload.question,
        top_k=payload.top_k,
        source_id=payload.source_id,
    )
    answer = rag_store.answer(payload.question, matches)
    warnings = [*search_warnings]
    if not matches:
        warnings.append("No indexed chunks found. Upload a document with /api/assistant/documents/index first.")

    return RagAskResponse(
        question=payload.question,
        answer=answer,
        matches=[
            RagSearchMatch(
                id=match.id,
                text=match.text,
                score=match.score,
                metadata=match.metadata,
            )
            for match in matches
        ],
        warnings=warnings,
    )


@router.post("/vision/analyze", response_model=VisionAnalysisResponse)
async def analyze_with_vision_llm(
    file: UploadFile = File(...),
    prompt: str = Form(
        "Opisz dokument i wskaż informacje ważne dla sprawy urzędowej mieszkańca Lublina."
    ),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = await vision_service.analyze(
        content=content,
        filename=file.filename or "document",
        content_type=file.content_type,
        prompt=prompt,
    )
    return VisionAnalysisResponse(
        filename=file.filename or "document",
        model=result.model,
        answer=result.answer,
        warnings=result.warnings,
    )

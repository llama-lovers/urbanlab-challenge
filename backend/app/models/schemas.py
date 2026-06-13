from pydantic import BaseModel, Field


class DocumentAnalysisResponse(BaseModel):
    filename: str
    content_type: str | None = None
    text: str
    text_source: str
    needs_ocr: bool
    pages: int | None = None
    warnings: list[str] = Field(default_factory=list)


class RagChunk(BaseModel):
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagEmbeddingRequest(BaseModel):
    text: str
    source_id: str = "manual"
    chunk_size: int = 900
    overlap: int = 120


class RagEmbeddingResponse(BaseModel):
    source_id: str
    model: str
    dimension: int
    chunks: list[RagChunk]
    warnings: list[str] = Field(default_factory=list)


class RagIndexResponse(BaseModel):
    source_id: str
    indexed_chunks: int
    total_chunks: int
    warnings: list[str] = Field(default_factory=list)


class RagSearchMatch(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagAskRequest(BaseModel):
    question: str
    top_k: int = 3
    source_id: str | None = None


class RagAskResponse(BaseModel):
    question: str
    answer: str
    matches: list[RagSearchMatch]
    warnings: list[str] = Field(default_factory=list)


class VisionAnalysisResponse(BaseModel):
    filename: str
    model: str
    answer: str
    warnings: list[str] = Field(default_factory=list)

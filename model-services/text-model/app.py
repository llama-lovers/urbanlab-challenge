from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODEL_TASK: Literal["embedding", "reranker"] = os.getenv("MODEL_TASK", "embedding")  # type: ignore[assignment]
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "8192"))

app = FastAPI(title="UrbanLab Text Model Service", version="0.1.0")
model = None


class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


class RerankPair(BaseModel):
    query: str
    text: str


class RerankRequest(BaseModel):
    pairs: list[RerankPair]


class RerankResponse(BaseModel):
    model: str
    scores: list[float]


def get_model():
    global model
    if model is not None:
        return model

    if MODEL_TASK == "embedding":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        return model

    if MODEL_TASK == "reranker":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(MODEL_NAME, max_length=MODEL_MAX_LENGTH)
        return model

    raise RuntimeError(f"Unsupported MODEL_TASK: {MODEL_TASK}")


@app.on_event("startup")
def load_model_on_startup() -> None:
    get_model()


@app.get("/health")
def health():
    get_model()
    return {"status": "ok", "task": MODEL_TASK, "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest):
    if MODEL_TASK != "embedding":
        raise HTTPException(status_code=400, detail="This service is not configured for embeddings")

    embeddings = get_model().encode(payload.texts, normalize_embeddings=payload.normalize)
    vectors = [[float(value) for value in vector] for vector in embeddings]
    dimension = len(vectors[0]) if vectors else 0
    return EmbedResponse(model=MODEL_NAME, dimension=dimension, embeddings=vectors)


@app.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest):
    if MODEL_TASK != "reranker":
        raise HTTPException(status_code=400, detail="This service is not configured for reranking")

    pairs = [(pair.query, pair.text) for pair in payload.pairs]
    scores = get_model().predict(pairs)
    return RerankResponse(model=MODEL_NAME, scores=[float(score) for score in scores])

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hackathon"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    debug: bool = False

    ocr_language: str = "pol+eng"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    embedding_service_url: str | None = "http://embedding-minilm:8080"
    reranker_model: str = "sdadas/polish-reranker-roberta-v3"
    reranker_enabled: bool = True
    reranker_candidate_multiplier: int = 4
    reranker_service_url: str | None = "http://reranker-polish-roberta:8080"
    vision_llm_provider: str = "ollama"
    vision_llm_base_url: str | None = "http://host.docker.internal:11434"
    vision_llm_api_key: str | None = None
    vision_llm_model: str = "glm-ocr"
    openai_api_key: str | None = None

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Model service — leave empty to use the built-in mock client
    model_service_url: str = ""
    model_service_api_key: str = ""
    model_timeout_s: float = 60.0
    history_limit: int = 20

    # OpenRouter / OpenAI-compatible model
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_send_app_headers: bool = True

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "urbanlab"

    # RAG-augmented chat
    rag_top_k: int = 5
    rag_system_prompt: str = (
        "Jesteś pomocnym asystentem UrbanLab Lublin. "
        "Odpowiadaj wyłącznie na podstawie poniższego kontekstu. "
        "Jeśli odpowiedź nie wynika z kontekstu, napisz że nie wiesz.\n\n"
        "Kontekst:\n{context}"
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value

    @field_validator(
        "embedding_service_url",
        "reranker_service_url",
        "vision_llm_base_url",
        "vision_llm_api_key",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def resolved_vision_llm_api_key(self) -> str | None:
        return self.vision_llm_api_key or self.openai_api_key


settings = Settings()

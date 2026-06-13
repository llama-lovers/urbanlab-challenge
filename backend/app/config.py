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
    reranker_model: str = "sdadas/polish-reranker-roberta-v3"
    reranker_enabled: bool = True
    reranker_candidate_multiplier: int = 4
    vision_llm_base_url: str | None = None
    vision_llm_api_key: str | None = None
    vision_llm_model: str = "gpt-4o-mini"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Model service — leave empty to use the built-in mock client
    model_service_url: str = ""
    model_service_api_key: str = ""
    model_timeout_s: float = 60.0
    history_limit: int = 20

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value


settings = Settings()

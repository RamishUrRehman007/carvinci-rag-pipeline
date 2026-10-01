from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    app_name: str = "CARvinci RAG Pipeline"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    max_upload_size_mb: int = Field(default=50, gt=0)

    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    google_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.5-flash-lite"
    llm_thinking_level: Literal["low", "medium", "high"] = "low"


@lru_cache
def get_settings() -> Settings:
    return Settings()

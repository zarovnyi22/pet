from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# all-MiniLM-L6-v2 truncates input at 256 tokens, 2 of which are [CLS]/[SEP].
MAX_CHUNK_TOKENS = 254


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/reformulation"
    llm_provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40
    agent_max_iterations: int = 6
    agent_timeout_seconds: int = 60

    @model_validator(mode="after")
    def check_chunking(self) -> "Settings":
        if not 0 < self.chunk_size_tokens <= MAX_CHUNK_TOKENS:
            raise ValueError(f"CHUNK_SIZE_TOKENS must be in 1..{MAX_CHUNK_TOKENS}")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be in [0, CHUNK_SIZE_TOKENS)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

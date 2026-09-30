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
    # Empty = no fallback. Otherwise the other provider takes over a call after the primary's
    # retries on 503 are exhausted, or on 429 (FallbackLLMClient).
    llm_fallback_provider: Literal["", "gemini", "groq"] = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_model: str = "openai/gpt-oss-120b"
    # reasoning_effort for gpt-oss models on Groq. "low" measured: ~30% fewer tokens per
    # /reformulate run, same substitutions (README). Empty in .env = not sent (Groq's default).
    groq_reasoning_effort: Literal["", "low", "medium", "high"] = "low"
    llm_timeout_seconds: float = 30.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40
    agent_timeout_seconds: int = 60
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def check_chunking(self) -> "Settings":
        if not 0 < self.chunk_size_tokens <= MAX_CHUNK_TOKENS:
            raise ValueError(f"CHUNK_SIZE_TOKENS must be in 1..{MAX_CHUNK_TOKENS}")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be in [0, CHUNK_SIZE_TOKENS)")
        if self.llm_fallback_provider == self.llm_provider:
            raise ValueError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER (or be empty)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

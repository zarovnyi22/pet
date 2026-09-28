from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/reformulation"
    llm_provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    agent_max_iterations: int = 6
    agent_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()

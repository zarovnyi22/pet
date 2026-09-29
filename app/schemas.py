from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocType = Literal["ingredient_spec", "trial_report", "guideline"]


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1)
    doc_type: DocType
    content: str

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        return value


class DocumentOut(BaseModel):
    doc_id: str
    chunks_created: int

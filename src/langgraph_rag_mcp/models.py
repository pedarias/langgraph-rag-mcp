from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from langgraph_rag_mcp.settings import Settings


class Page(BaseModel):
    source: str
    title: str
    content: str = Field(min_length=1)


class IndexConfig(BaseModel):
    schema_version: Literal[1] = 1
    model_name: str
    model_revision: str
    chunk_size: int
    chunk_overlap: int
    index_url: str
    normalize_embeddings: Literal[True] = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "IndexConfig":
        return cls.model_validate(settings.model_dump())


class SnapshotPointer(BaseModel):
    generation: str = Field(pattern=r"^[a-f0-9]{32}$")


class Manifest(SnapshotPointer):
    config: IndexConfig
    created_at: datetime
    pages_hash: str
    pages: int = Field(ge=1)
    chunks: int = Field(ge=1)


class BuildResult(SnapshotPointer):
    pages: int
    chunks: int
    embedded_chunks: int


class SearchHit(BaseModel):
    id: str
    source: str
    title: str
    section: str
    content: str
    score: float = Field(description="Cosine similarity, higher is better; not a confidence score.")


class SearchResponse(BaseModel):
    query: str
    indexed_at: datetime
    results: list[SearchHit]


class PageResponse(BaseModel):
    source: str
    title: str
    content: str
    offset: int
    next_offset: int | None
    total_characters: int

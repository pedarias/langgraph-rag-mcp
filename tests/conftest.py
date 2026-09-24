from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from langgraph_rag_mcp.models import Page
from langgraph_rag_mcp.settings import Settings


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(word in text.lower()) for word in ("memory", "stream", "graph")] + [0.1]


class FakeTokenizer:
    model_max_length = 64

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(text.encode()) + ([0, 0] if add_special_tokens else [])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, chunk_size=64, chunk_overlap=8)


@pytest.fixture
def pages() -> list[Page]:
    return [
        Page(
            source="https://docs.langchain.com/oss/python/langgraph/memory",
            title="Memory",
            content="# Memory\n\nStore memory between graph runs.",
        ),
        Page(
            source="https://docs.langchain.com/oss/python/langgraph/streaming",
            title="Streaming",
            content="# Streaming\n\nStream tokens from a graph.",
        ),
    ]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

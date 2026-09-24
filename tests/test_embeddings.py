from pathlib import Path

import pytest
from conftest import FakeEmbeddings

from langgraph_rag_mcp.embeddings import CachedEmbeddings


def test_duplicate_texts_are_embedded_once_and_cache_survives_restart(tmp_path: Path) -> None:
    backend = FakeEmbeddings()
    cache = CachedEmbeddings(backend, tmp_path / "embeddings.sqlite3", "model-revision")
    first = cache.embed_documents(["memory", "memory", "stream"])
    assert backend.documents == ["memory", "stream"]
    assert first[0] == first[1]
    restarted = CachedEmbeddings(backend, cache.path, cache.namespace)
    assert restarted.embed_documents(["memory", "stream"]) == [first[0], first[2]]
    assert restarted.embedded_chunks == 0
    assert len(backend.documents) == 2


def test_invalid_embeddings_are_not_committed_to_cache(tmp_path: Path) -> None:
    class InvalidEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float("nan")] for _ in texts]

    cache = CachedEmbeddings(InvalidEmbeddings(), tmp_path / "embeddings.sqlite3", "model")
    with pytest.raises(ValueError, match="Invalid"):
        cache.embed_documents(["memory"])
    backend = FakeEmbeddings()
    recovered = CachedEmbeddings(backend, cache.path, cache.namespace)
    assert recovered.embed_documents(["memory"]) == [backend.embed_query("memory")]
    assert backend.documents == ["memory"]

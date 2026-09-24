import json

import pytest
from conftest import FakeEmbeddings, FakeTokenizer

from langgraph_rag_mcp.chunking import split_pages
from langgraph_rag_mcp.index import build_index, load_manifest
from langgraph_rag_mcp.models import Page
from langgraph_rag_mcp.retrieval import DocumentationIndex, IndexNotReadyError
from langgraph_rag_mcp.settings import Settings


def test_chunks_fit_the_embedding_tokenizer_and_preserve_metadata(settings: Settings) -> None:
    page = Page(
        source="https://example.test/page",
        title="Guide",
        content="# Guide\n\n## Memory\n\n" + "Memória persistente no grafo. " * 30,
    )
    chunks = split_pages([page], settings, FakeTokenizer())
    assert len(chunks) > 1
    assert all(len(FakeTokenizer().encode(chunk.page_content)) <= 64 for chunk in chunks)
    assert all(chunk.metadata["source"] == page.source for chunk in chunks)
    assert all(chunk.metadata["section"] == "Memory" for chunk in chunks)
    assert all(chunk.id for chunk in chunks)


def test_rejects_chunks_larger_than_model_window(settings: Settings, pages: list[Page]) -> None:
    settings = settings.model_copy(update={"chunk_size": 100})
    with pytest.raises(ValueError, match="tokenizer"):
        split_pages(pages, settings, FakeTokenizer())


def test_index_roundtrip_and_unchanged_build_reuses_embeddings(settings: Settings, pages: list[Page]) -> None:
    embeddings = FakeEmbeddings()
    first = build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    count = len(embeddings.documents)
    second = build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    assert first.generation == second.generation
    assert len(embeddings.documents) == count
    assert second.embedded_chunks == 0
    result = DocumentationIndex(settings, embeddings=embeddings).search("memory", k=10)
    assert len(result.results) == 2
    assert result.results[0].source == pages[0].source
    assert result.results[0].score > result.results[1].score


def test_changed_pages_only_embed_new_chunks_and_removed_pages_disappear(
    settings: Settings, pages: list[Page]
) -> None:
    embeddings = FakeEmbeddings()
    first = build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    count = len(embeddings.documents)
    service = DocumentationIndex(settings, embeddings=embeddings)
    service.search("memory")
    updated = pages[0].model_copy(update={"content": "# Memory\n\nNew persistent memory."})
    second = build_index(settings, pages=[updated], embeddings=embeddings, tokenizer=FakeTokenizer())
    assert first.generation != second.generation
    assert len(embeddings.documents) == count + 1
    result = service.search("memory")
    assert len(result.results) == 1
    assert "New persistent memory" in result.results[0].content


def test_failed_build_preserves_the_previous_snapshot(settings: Settings, pages: list[Page]) -> None:
    embeddings = FakeEmbeddings()
    first = build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())

    class BrokenEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        build_index(
            settings,
            pages=[pages[0].model_copy(update={"content": "Changed text"})],
            embeddings=BrokenEmbeddings(),
            tokenizer=FakeTokenizer(),
        )
    assert load_manifest(settings).generation == first.generation


def test_missing_and_incompatible_indices_fail_before_loading_models(
    settings: Settings, pages: list[Page]
) -> None:
    service = DocumentationIndex(settings)
    with pytest.raises(IndexNotReadyError, match="index"):
        service.search("memory")
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    changed = settings.model_copy(update={"model_name": "different-model"})
    with pytest.raises(IndexNotReadyError, match="incompatible"):
        DocumentationIndex(changed).search("memory")


def test_snapshot_path_cannot_escape_data_directory(settings: Settings) -> None:
    (settings.data_dir / "current.json").write_text(json.dumps({"generation": "../../outside"}))
    with pytest.raises(IndexNotReadyError):
        DocumentationIndex(settings).search("memory")


@pytest.mark.parametrize("query,k", [(" ", 3), ("x" * 4001, 3), ("memory", 0), ("memory", 21)])
def test_query_input_is_bounded(settings: Settings, query: str, k: int) -> None:
    with pytest.raises(ValueError):
        DocumentationIndex(settings).search(query, k=k)


def test_model_and_store_are_loaded_once_per_process(
    settings: Settings, pages: list[Page], monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    factory = Mock(return_value=embeddings)
    monkeypatch.setattr("langgraph_rag_mcp.retrieval.load_embeddings", factory)
    service = DocumentationIndex(settings)
    service.search("memory")
    store = service._store
    service.search("stream")
    assert service._store is store
    factory.assert_called_once()


def test_model_revision_change_invalidates_embedding_cache(settings: Settings, pages: list[Page]) -> None:
    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    count = len(embeddings.documents)
    changed = settings.model_copy(update={"model_revision": "different-revision"})
    result = build_index(changed, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    assert result.embedded_chunks == count
    assert len(embeddings.documents) == 2 * count


def test_empty_rebuild_does_not_replace_a_working_index(settings: Settings, pages: list[Page]) -> None:
    first = build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="empty"):
        build_index(settings, pages=[], embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    assert load_manifest(settings).generation == first.generation


def test_page_reads_are_local_and_paginated_without_loading_models(
    settings: Settings, pages: list[Page]
) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    service = DocumentationIndex(settings)
    with pytest.raises(ValueError, match="Unknown source"):
        service.read_page("https://untrusted.test/page")
    page = service.read_page(pages[0].source, offset=10, limit=20000)
    assert page.content == pages[0].content[10:]
    assert page.next_offset is None
    assert service._embeddings is None

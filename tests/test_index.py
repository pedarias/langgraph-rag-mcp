import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import FakeEmbeddings, FakeTokenizer

from langgraph_rag_mcp.chunking import split_pages
from langgraph_rag_mcp.index import build_index, load_manifest, snapshot_path
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


@pytest.mark.parametrize("artifact", ["pages.json", "vectors.parquet", "manifest.json"])
def test_unchanged_build_rejects_missing_snapshot_artifacts(
    settings: Settings, pages: list[Page], artifact: str
) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    snapshot = snapshot_path(settings, load_manifest(settings))
    pointer = (settings.data_dir / "current.json").read_bytes()
    (snapshot / artifact).unlink()
    with pytest.raises(IndexNotReadyError):
        build_index(settings, pages=pages)
    assert (settings.data_dir / "current.json").read_bytes() == pointer


@pytest.mark.parametrize(
    "artifact,content",
    [
        ("pages.json", b"{"),
        ("pages.json", b"[]"),
        ("pages.json", b'[{"source": "page", "title": "Page", "content": ""}]'),
        ("vectors.parquet", b"not parquet"),
    ],
)
def test_unchanged_build_rejects_invalid_snapshot_artifacts(
    settings: Settings, pages: list[Page], artifact: str, content: bytes
) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    snapshot = snapshot_path(settings, load_manifest(settings))
    pointer = (settings.data_dir / "current.json").read_bytes()
    (snapshot / artifact).write_bytes(content)
    with pytest.raises(IndexNotReadyError):
        build_index(settings, pages=pages)
    assert (settings.data_dir / "current.json").read_bytes() == pointer


def test_unchanged_build_rejects_vector_directory(settings: Settings, pages: list[Page]) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    path = snapshot_path(settings, load_manifest(settings)) / "vectors.parquet"
    table = pq.read_table(path)
    path.unlink()
    path.mkdir()
    pq.write_table(table, path / "part.parquet")
    with pytest.raises(IndexNotReadyError):
        build_index(settings, pages=pages)


def test_unchanged_build_rejects_modified_page_contents(settings: Settings, pages: list[Page]) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    snapshot = snapshot_path(settings, load_manifest(settings))
    altered = [page.model_copy(update={"content": "Changed content"}) for page in pages]
    (snapshot / "pages.json").write_text(json.dumps([page.model_dump() for page in altered]))
    with pytest.raises(IndexNotReadyError):
        build_index(settings, pages=pages)


@pytest.mark.parametrize(
    "damage", ["missing_column", "nonfinite", "empty_vectors", "metadata", "count", "source"]
)
def test_unchanged_build_rejects_invalid_vector_data(
    settings: Settings, pages: list[Page], damage: str
) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    path = snapshot_path(settings, load_manifest(settings)) / "vectors.parquet"
    data = pq.read_table(path).to_pydict()
    if damage == "missing_column":
        del data["embedding"]
    elif damage == "nonfinite":
        data["embedding"][0][0] = float("nan")
    elif damage == "empty_vectors":
        data["embedding"] = [[] for _ in data["embedding"]]
    elif damage == "metadata":
        data["metadata"][0] = "{}"
    elif damage == "count":
        data = {key: values[:-1] for key, values in data.items()}
    else:
        metadata = json.loads(data["metadata"][0])
        metadata["source"] = "https://unknown.test/page"
        data["metadata"][0] = json.dumps(metadata)
    pq.write_table(pa.table(data), path)
    with pytest.raises(IndexNotReadyError):
        build_index(settings, pages=pages)


def test_unchanged_build_validation_does_not_load_models(
    settings: Settings, pages: list[Page], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())

    def unexpected_model_load(*args: object) -> None:
        pytest.fail("An unchanged build must not load a model or tokenizer")

    monkeypatch.setattr("langgraph_rag_mcp.index.load_embeddings", unexpected_model_load)
    monkeypatch.setattr("langgraph_rag_mcp.index.load_tokenizer", unexpected_model_load)
    second = build_index(settings, pages=pages)
    assert second.generation == first.generation
    assert second.embedded_chunks == 0


@pytest.mark.parametrize("page_count", [1, 2])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_full_docs_limit_includes_sources_and_separators(
    settings: Settings, pages: list[Page], monkeypatch: pytest.MonkeyPatch, page_count: int, delta: int
) -> None:
    selected = pages[:page_count]
    overhead = sum(len(f"SOURCE: {page.source}\n\n") for page in selected) + 7 * (page_count - 1)
    content_length = 200000 - overhead - sum(len(page.content) for page in selected[1:]) + delta
    selected[0] = selected[0].model_copy(update={"content": "é" * content_length})
    service = DocumentationIndex(settings)
    service._pages = {page.source: page for page in selected}
    monkeypatch.setattr(service, "_refresh", lambda: None)
    if delta > 0:
        with pytest.raises(ValueError, match="Corpus exceeds"):
            service.full_docs()
    else:
        result = service.full_docs()
        assert result == "\n\n---\n\n".join(f"SOURCE: {page.source}\n\n{page.content}" for page in selected)
        assert len(result) == 200000 + delta


@pytest.mark.parametrize("page_count", [1, 2])
def test_full_docs_rejects_oversized_content_before_formatting(
    settings: Settings, pages: list[Page], monkeypatch: pytest.MonkeyPatch, page_count: int
) -> None:
    class UnformattableText(str):
        def __format__(self, format_spec: str) -> str:
            pytest.fail("Oversized page content must not be formatted into a response")

    selected = [page.model_copy(update={"content": "x" * 100001}) for page in pages[:page_count]]
    selected[-1] = selected[-1].model_copy(
        update={"content": UnformattableText("x" * (200001 if page_count == 1 else 100001))}
    )
    service = DocumentationIndex(settings)
    service._pages = {page.source: page for page in selected}
    monkeypatch.setattr(service, "_refresh", lambda: None)
    with pytest.raises(ValueError, match="Corpus exceeds"):
        service.full_docs()

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import FakeEmbeddings, FakeTokenizer
from pydantic import ValidationError

from langgraph_rag_mcp.cli import main
from langgraph_rag_mcp.evaluation import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseResult,
    evaluate,
    load_dataset,
    summarize,
)
from langgraph_rag_mcp.index import build_index
from langgraph_rag_mcp.models import Page, SearchHit
from langgraph_rag_mcp.retrieval import DocumentationIndex
from langgraph_rag_mcp.settings import Settings


def hit(source: str) -> SearchHit:
    return SearchHit(id=source, source=source, title="Title", section="Section", content="Text", score=0.5)


def case_result(language: str, sources: list[str], expected: str, latency: float = 10) -> CaseResult:
    return CaseResult(
        id=f"case-{language}",
        language=language,
        query="test",
        expected_sources=[expected],
        latency_ms=latency,
        results=[hit(source) for source in sources],
    )


def test_metrics_keep_chunk_ranks_and_count_duplicate_sources() -> None:
    results = [
        case_result(
            "en",
            ["https://docs.test/a", "https://docs.test/a", "https://docs.test/b"],
            "https://docs.test/b",
            10,
        ),
        case_result("pt", ["https://docs.test/c"], "https://docs.test/b", 30),
    ]
    metrics = summarize(results, [1, 3])
    assert metrics.cases == 2
    assert metrics.hit_rate[1] == 0
    assert metrics.hit_rate[3] == 0.5
    assert metrics.mrr[3] == pytest.approx(1 / 6, abs=1e-6)
    assert metrics.duplicate_source_fraction[3] == pytest.approx(1 / 6, abs=1e-6)
    assert metrics.latency_p50_ms == 20
    assert metrics.latency_p95_ms == 30


def test_any_expected_source_is_accepted_and_url_aliases_are_normalized() -> None:
    result = case_result("en", ["https://docs.test/b.md#section"], "https://docs.test/b/")
    result.expected_sources.append("https://docs.test/c")
    metrics = summarize([result], [1, 5])
    assert metrics.hit_rate == {1: 1.0, 5: 1.0}
    assert metrics.mrr[1] == 1.0


def test_no_hits_do_not_cause_division_by_zero() -> None:
    metrics = summarize([case_result("pt", [], "https://docs.test/a")], [5])
    assert metrics.hit_rate[5] == 0
    assert metrics.mrr[5] == 0
    assert metrics.duplicate_source_fraction[5] == 0


@pytest.mark.parametrize(
    "values",
    [
        {"cases": []},
        {
            "cases": [
                {"id": "one", "language": "en", "query": "  ", "expected_sources": ["https://docs.test/a"]}
            ]
        },
        {"cases": [{"id": "one", "language": "en", "query": "test", "expected_sources": []}]},
    ],
)
def test_invalid_datasets_are_rejected(values: dict) -> None:
    with pytest.raises(ValidationError):
        BenchmarkDataset(name="test", **values)


def test_duplicate_case_ids_are_rejected() -> None:
    case = BenchmarkCase(id="same", language="en", query="memory", expected_sources=["https://docs.test/a"])
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkDataset(name="test", cases=[case, case])


def test_default_dataset_has_paired_queries_and_stable_fingerprint() -> None:
    dataset = load_dataset()
    assert len(dataset.cases) == 20
    for topic in {case.id.rsplit("-", 1)[0] for case in dataset.cases}:
        pair = [case for case in dataset.cases if case.id.rsplit("-", 1)[0] == topic]
        assert {case.language for case in pair} == {"en", "pt"}
        assert pair[0].expected_sources == pair[1].expected_sources
    assert dataset.fingerprint == load_dataset().fingerprint


def test_evaluation_records_snapshot_and_language_metrics(settings: Settings, pages: list[Page]) -> None:
    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    index = DocumentationIndex(settings, embeddings=embeddings)
    dataset = BenchmarkDataset(
        name="small",
        cases=[
            BenchmarkCase(id="en", language="en", query="memory", expected_sources=[pages[0].source]),
            BenchmarkCase(id="pt", language="pt", query="stream", expected_sources=[pages[1].source]),
        ],
    )
    report = evaluate(index, dataset, max_k=3)
    assert report.index.generation == index.status().generation
    assert report.dataset_sha256 == dataset.fingerprint
    assert report.groups["all"].cases == 2
    assert report.groups["en"].hit_rate[1] == 1
    assert report.groups["pt"].hit_rate[1] == 1
    assert report.warmup_ms >= 0
    assert len(report.cases) == 2


def test_missing_ground_truth_source_fails_before_search(settings: Settings, pages: list[Page]) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    dataset = BenchmarkDataset(
        name="missing",
        cases=[
            BenchmarkCase(
                id="en", language="en", query="test", expected_sources=["https://docs.test/missing"]
            )
        ],
    )
    with pytest.raises(ValueError, match="absent"):
        evaluate(DocumentationIndex(settings), dataset)


def test_evaluation_rejects_snapshot_changes(
    settings: Settings, pages: list[Page], monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    index = DocumentationIndex(settings, embeddings=embeddings)
    original_search = index.search

    def changed_search(query: str, k: int = 3):
        return original_search(query, k).model_copy(update={"indexed_at": datetime(2000, 1, 1, tzinfo=UTC)})

    monkeypatch.setattr(index, "search", changed_search)
    dataset = BenchmarkDataset(
        name="changed",
        cases=[BenchmarkCase(id="en", language="en", query="memory", expected_sources=[pages[0].source])],
    )
    with pytest.raises(RuntimeError, match="changed"):
        evaluate(index, dataset)


def test_cli_writes_report_without_overwriting_existing_files(
    settings: Settings,
    pages: list[Page],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    monkeypatch.setattr("langgraph_rag_mcp.retrieval.load_embeddings", lambda settings: embeddings)
    monkeypatch.setenv("LANGGRAPH_MCP_CHUNK_SIZE", "64")
    monkeypatch.setenv("LANGGRAPH_MCP_CHUNK_OVERLAP", "8")
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "test",
                "cases": [
                    {
                        "id": "memory",
                        "language": "en",
                        "query": "memory",
                        "expected_sources": [pages[0].source],
                    }
                ],
            }
        )
    )
    output = tmp_path / "report.json"
    args = [
        "--data-dir",
        str(settings.data_dir),
        "evaluate",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    original = output.read_text()
    assert len(json.loads(original)["cases"]) == 1
    assert "cases" not in json.loads(capsys.readouterr().out)
    assert main(args) == 1
    assert output.read_text() == original
    assert "already exists" in capsys.readouterr().err

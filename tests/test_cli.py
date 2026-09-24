import json
from pathlib import Path

import pytest
from conftest import FakeEmbeddings, FakeTokenizer
from pydantic import ValidationError

from langgraph_rag_mcp.cli import main
from langgraph_rag_mcp.index import build_index
from langgraph_rag_mcp.models import Page
from langgraph_rag_mcp.settings import Settings


def test_status_is_json_without_model_dependencies(
    settings: Settings, pages: list[Page], capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(settings, pages=pages, embeddings=FakeEmbeddings(), tokenizer=FakeTokenizer())
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LANGGRAPH_MCP_CHUNK_SIZE", "64")
        patch.setenv("LANGGRAPH_MCP_CHUNK_OVERLAP", "8")
        assert main(["--data-dir", str(settings.data_dir), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["pages"] == 2


def test_missing_index_is_an_error_on_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path), "query", "memory"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "langgraph-rag index" in captured.err


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_size": 32, "chunk_overlap": 32},
        {"index_url": "http://example.test/llms.txt"},
        {"index_url": "https://user:password@example.test/llms.txt"},
    ],
)
def test_invalid_settings_are_rejected(values: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**values)

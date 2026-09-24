import json
import sys
from pathlib import Path

import pytest
from conftest import FakeEmbeddings, FakeTokenizer
from mcp import Client, StdioServerParameters

from langgraph_rag_mcp.index import build_index
from langgraph_rag_mcp.models import Page
from langgraph_rag_mcp.retrieval import DocumentationIndex
from langgraph_rag_mcp.server import create_server
from langgraph_rag_mcp.settings import Settings


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_mcp_search_and_paginated_page(
    settings: Settings, pages: list[Page], mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    embeddings = FakeEmbeddings()
    build_index(settings, pages=pages, embeddings=embeddings, tokenizer=FakeTokenizer())
    server = create_server(DocumentationIndex(settings, embeddings=embeddings))
    async with Client(server, mode=mode) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"langgraph_query_tool", "read_page"}
        result = await client.call_tool("langgraph_query_tool", {"query": "memory", "k": 2})
        assert not result.is_error
        assert result.structured_content["results"][0]["source"] == pages[0].source
        page = await client.call_tool("read_page", {"source": pages[0].source, "limit": 10})
        assert page.structured_content["content"] == pages[0].content[:10]
        assert page.structured_content["next_offset"] == 10
        status = await client.read_resource("docs://langgraph/status")
        assert json.loads(status.contents[0].text)["pages"] == 2
        full = await client.read_resource("docs://langgraph/full")
        assert pages[0].source in full.contents[0].text
    assert capsys.readouterr().out == ""


@pytest.mark.anyio
async def test_missing_index_returns_actionable_tool_error(settings: Settings) -> None:
    async with Client(create_server(DocumentationIndex(settings))) as client:
        result = await client.call_tool("langgraph_query_tool", {"query": "memory"})
        assert result.is_error
        assert "langgraph-rag index" in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_script", [False, True])
async def test_stdio_transport_starts_without_models_or_index(
    settings: Settings, legacy_script: bool
) -> None:
    args = (
        [str(Path(__file__).resolve().parents[1] / "langgraph-mcp.py")]
        if legacy_script
        else ["-m", "langgraph_rag_mcp", "serve"]
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=args,
        env={"LANGGRAPH_MCP_DATA_DIR": str(settings.data_dir)},
    )
    async with Client(parameters) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 2
        result = await client.call_tool("langgraph_query_tool", {"query": "memory"})
        assert result.is_error
        assert "langgraph-rag index" in result.content[0].text

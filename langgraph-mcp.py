import os

from langgraph_rag_mcp.models import SearchResponse
from langgraph_rag_mcp.retrieval import DocumentationIndex
from langgraph_rag_mcp.server import create_server
from langgraph_rag_mcp.settings import Settings

# Define common path to the repo locally
PATH = os.path.dirname(os.path.abspath(__file__))

# Create an MCP server
service = DocumentationIndex(Settings())
mcp = create_server(service)


# Add a tool to query the LangGraph documentation
def langgraph_query_tool(query: str, k: int = 3) -> SearchResponse:
    """Compatibility entry point for source-attributed documentation search."""
    return service.search(query, k)


# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
def get_all_langgraph_docs() -> str:
    """Compatibility entry point for the size-limited documentation resource."""

    # Local path to the LangGraph documentation
    return service.full_docs()


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")

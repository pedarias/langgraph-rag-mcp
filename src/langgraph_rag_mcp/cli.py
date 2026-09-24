import argparse
import logging
import sys
from pathlib import Path

import httpx
from filelock import Timeout

from langgraph_rag_mcp.evaluation import evaluate, load_dataset
from langgraph_rag_mcp.index import build_index
from langgraph_rag_mcp.retrieval import DocumentationIndex
from langgraph_rag_mcp.server import create_server
from langgraph_rag_mcp.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local LangGraph documentation index and MCP server")
    parser.add_argument("--data-dir", type=Path, help="Override LANGGRAPH_MCP_DATA_DIR")
    parser.add_argument("--index-url", help="HTTPS llms.txt source; overrides LANGGRAPH_MCP_INDEX_URL")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Fetch docs and atomically update the local index")
    commands.add_parser("serve", help="Run the read-only MCP server over stdio")
    commands.add_parser("status", help="Show index metadata without loading the embedding model")
    query_parser = commands.add_parser("query", help="Search the local documentation index")
    query_parser.add_argument("query")
    query_parser.add_argument("--k", type=int, default=3)
    evaluation_parser = commands.add_parser("evaluate", help="Measure retrieval against labeled queries")
    evaluation_parser.add_argument(
        "--dataset", type=Path, help="JSON dataset; defaults to bundled EN/PT cases"
    )
    evaluation_parser.add_argument("--k", type=int, default=5, help="Maximum retrieval cutoff (1–20)")
    evaluation_parser.add_argument(
        "--output", type=Path, help="Save full report to a new file; print a summary"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    overrides = {
        key: getattr(args, key) for key in ("data_dir", "index_url") if getattr(args, key) is not None
    }
    try:
        settings = Settings(**overrides)
        service = DocumentationIndex(settings)
        if args.command == "serve":
            create_server(service).run(transport="stdio")
        elif args.command == "index":
            print(build_index(settings).model_dump_json(indent=2))
        elif args.command == "query":
            print(service.search(args.query, args.k).model_dump_json(indent=2))
        elif args.command == "evaluate":
            if args.output is not None:
                if args.output.exists():
                    raise FileExistsError(f"Report already exists: {args.output}")
                if not args.output.parent.is_dir():
                    raise FileNotFoundError(f"Report directory does not exist: {args.output.parent}")
            report = evaluate(service, load_dataset(args.dataset), max_k=args.k)
            if args.output is not None:
                with args.output.open("x", encoding="utf-8") as output:
                    output.write(report.model_dump_json(indent=2) + "\n")
                print(report.model_dump_json(indent=2, exclude={"cases"}))
            else:
                print(report.model_dump_json(indent=2))
        else:
            print(service.status().model_dump_json(indent=2))
    except Timeout:
        print("Another indexing process is running for this data directory.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0

# LangGraph RAG MCP

A local documentation retrieval server for MCP-compatible coding assistants. It indexes LangGraph's official Markdown documentation, searches it with local embeddings, and returns excerpts with source URLs.

The server does **not** call an LLM or require an Anthropic/OpenAI API key. Your MCP host generates the final answer. LangGraph is the documentation source, not an orchestration dependency.

If you only need hosted, up-to-date documentation without maintaining a local index, LangChain also provides [official documentation MCP servers](https://docs.langchain.com/use-these-docs).

## Requirements

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/), or Docker with Docker Compose.
- Internet access for installation, documentation updates, and the initial Hugging Face model download.
- Enough disk space and RAM for CPU inference with `BAAI/bge-large-en-v1.5`. Model weights are downloaded separately from the Python packages and can be substantial.

After the model is cached and the index is built, retrieval is local. Set `HF_HUB_OFFLINE=1` for an explicitly offline runtime.

## Quick start

From the repository root:

```bash
uv sync --locked --extra embeddings
uv run --locked --extra embeddings langgraph-rag index
uv run --locked --extra embeddings langgraph-rag query "How do interrupts work?" --k 5
uv run --locked --extra embeddings langgraph-rag status
```

Run the MCP server:

```bash
uv run --locked --extra embeddings langgraph-rag serve
```

The server uses **stdio**: it waits for an MCP host, not browser requests. Logs go to stderr; stdout is reserved for protocol messages. Starting it and discovering tools do not download the model or require an existing index. Searching requires both.

Keep `--extra embeddings` on `uv run` commands that need the local model. Running `uv sync` without that extra intentionally installs only the lightweight development/test environment.

### Data and configuration

By default, data is stored at `$XDG_DATA_HOME/langgraph-rag-mcp`, falling back to `~/.local/share/langgraph-rag-mcp`. This is independent of the working directory.

For project-local data, use the same directory for indexing and serving:

```bash
uv run --locked --extra embeddings langgraph-rag --data-dir ./data index
uv run --locked --extra embeddings langgraph-rag --data-dir ./data serve
```

Global CLI options must precede the subcommand. Alternatively, export `LANGGRAPH_MCP_DATA_DIR` as an absolute path.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LANGGRAPH_MCP_DATA_DIR` | XDG data directory | Index snapshots and embedding cache |
| `LANGGRAPH_MCP_INDEX_URL` | `https://docs.langchain.com/oss/python/langgraph/llms.txt` | Documentation index |
| `LANGGRAPH_MCP_MODEL_NAME` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `LANGGRAPH_MCP_MODEL_REVISION` | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` | Pinned Hugging Face model revision |
| `LANGGRAPH_MCP_CHUNK_SIZE` | `450` | Maximum embedding-tokenizer tokens, including special tokens |
| `LANGGRAPH_MCP_CHUNK_OVERLAP` | `50` | Chunk overlap budget |
| `LANGGRAPH_MCP_MAX_PAGES` | `200` | Maximum number of discovered Markdown pages |
| `LANGGRAPH_MCP_REQUEST_TIMEOUT` | `30` | HTTP timeout in seconds |
| `HF_HOME` | Hugging Face default | Downloaded model/tokenizer cache |

Configuration comes from exported environment variables and CLI options; `.env` files are not loaded implicitly. `--index-url` overrides the source URL. For another model, set **both** its name and an appropriate revision, then rebuild the index. The pinned default revision belongs specifically to BGE.

## MCP integration

For hosts using the `mcpServers` configuration format, merge an entry like this into the host's settings:

```json
{
  "mcpServers": {
    "langgraph-docs": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/absolute/path/to/langgraph-rag-mcp",
        "--locked",
        "--extra", "embeddings",
        "langgraph-rag", "serve"
      ]
    }
  }
}
```

Use the absolute path to `uv` if it is not on your editor's PATH. For VS Code, use the `servers` key instead of `mcpServers` in its MCP configuration. If you override the data directory or model during indexing, pass the same settings through the host's `env` configuration.

### Tools and resources

- **`langgraph_query_tool(query, k=3)`**: returns `query`, `indexed_at`, and a `results` list. Each result includes an ID, title, section, source URL, content, and cosine similarity. `k` is bounded to 1–20 and queries to 4,000 characters. Similarity is not a confidence score.
- **`read_page(source, offset=0, limit=12000)`**: reads only a source already present in the local index. Pagination is in characters, with a maximum of 20,000 per request. It cannot fetch arbitrary URLs.
- **`docs://langgraph/status`**: index timestamp, counts, source and embedding configuration.
- **`docs://langgraph/full`**: compatibility resource for small corpora, limited to 200,000 characters. Prefer search and paginated reads for the complete LangGraph corpus.

Tools provide structured output as well as text content for host compatibility. Both tools are advertised as read-only. The embedding model and vector index are reused across requests; newly published snapshots are picked up automatically.

## Docker

Build the image and create the index before connecting your host:

```bash
docker compose build
docker compose run --rm --no-deps -T langgraph-mcp index
docker compose run --rm --no-deps -T langgraph-mcp query "How do checkpoints work?"
```

Then use the wrapper as the MCP host command:

```json
{
  "mcpServers": {
    "langgraph-docs": {
      "command": "/bin/bash",
      "args": ["/absolute/path/to/langgraph-rag-mcp/run-mcp-docker.sh"]
    }
  }
}
```

The wrapper resolves the project directory itself, does not allocate a TTY, and starts the server directly in a disposable container. An optional `--build` argument rebuilds the image first, sending build output to stderr. There is no background `tail` process, sleep, or `docker exec` lifecycle.

The image uses a locked CPU-only PyTorch installation on Linux, runs as UID 10001, and excludes `.env`, Git metadata, notebooks, tests and local data from the build context. Named volumes preserve the index (`index-data`) and downloaded models (`model-cache`). These Docker volumes are separate from a native installation's data directory.

## Indexing behavior

1. Discover Markdown pages from the configured `llms.txt`, including nested indexes within its directory scope.
2. Fetch pages with timeouts, HTTP status checks and a 2 MiB per-response limit. Redirects remain on the configured HTTPS origin; relocated Markdown pages retain Markdown retrieval. HTML/error pages are rejected.
3. Remove documentation-index/footer boilerplate and deduplicate identical page content.
4. Split by Markdown headings, then enforce the embedding tokenizer's token budget. Source, title and section metadata travel with each chunk.
5. Reuse embeddings from a local SQLite cache keyed by content and embedding configuration. Search vectors remain in **Parquet**, served by **scikit-learn** cosine nearest-neighbor search.
6. Publish a new immutable snapshot and atomically switch `current.json` only after the build succeeds. A failed build leaves the previous index active. Concurrent writers are prevented with a file lock.

Running `index` again refetches documentation to detect changes, but embeds only new/changed chunks. It does **not** implement conditional HTTP requests yet. If neither content nor configuration changed, the snapshot is reused. Pages removed from the source index disappear from the new search snapshot.

Old snapshots and cached embeddings are retained; automatic pruning is intentionally not implemented. A manifest records the model revision, chunking settings, source and index timestamp. Serving rejects configuration mismatches rather than silently mixing incompatible embeddings.

## Retrieval benchmark

The `evaluate` command measures retrieval with 20 manually authored questions: ten topics with equivalent English and Portuguese queries. The versioned dataset is bundled with the package at `src/langgraph_rag_mcp/data/benchmark.json`; no evaluation service or LLM API is involved.

With an existing Docker index, run:

```bash
docker compose run --rm --no-deps -T \
  -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e TOKENIZERS_PARALLELISM=false \
  langgraph-mcp evaluate --output /data/baseline.json
```

For a native installation:

```bash
uv run --locked --extra embeddings langgraph-rag evaluate
```

Without `--output`, stdout contains the complete JSON report. With it, the full report is saved and stdout contains a summary. The output directory must exist, and existing report files are never overwritten: choose a new filename for each comparison. Docker reports remain in the `index-data` volume.

Use `--dataset /path/to/questions.json` for another labeled dataset and `--k 10` to change the maximum retrieval cutoff (1–20). The dataset has `name`, `version`, `description` and a nonempty `cases` list. Each case requires a unique `id`, `language` (`en` or `pt`), `query`, and a nonempty list of `expected_sources`. All expected sources must exist in the current index; missing labels fail evaluation rather than silently counting as misses.

Reported metrics, both overall and per language:

- **Hit@k**: fraction of questions with at least one expected source among the first k chunks.
- **MRR@k**: mean reciprocal rank of the first expected source; zero for a miss.
- **Duplicate-source fraction@k**: mean `1 - unique sources / returned chunks`. Multiple chunks from one page may be useful, so this measures source diversity, not duplicate text or an error rate.
- **Warm-query latency p50/p95**: wall-clock search latency after one excluded warm-up query. The warm-up duration, including lazy model/index loading, is reported separately. p95 uses nearest-rank estimation.

The evaluator preserves the actual chunk ranking: it does not deduplicate results before scoring. Reports include all retrieved excerpts, cases missed at the maximum cutoff, the full index manifest, dataset and implementation SHA-256 fingerprints, package versions, and relevant thread/offline settings. Evaluation aborts if the index changes during the run.

This is a small diagnostic dataset, not a general accuracy claim or an exhaustive relevance judgment. Finding the expected page does not prove that the returned passage answers the question. The paired translations are correlated, and the suite does not assess generated answers, unsupported questions or abstention. Review failures before changing retrieval, and do not edit labels to improve a measured score. Change the dataset version when changing its questions or expected sources.

## Development and verification

The default development installation does not include PyTorch, model weights or an LLM SDK:

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv build
bash -n run-mcp-docker.sh
docker compose config --quiet
```

Tests use deterministic embeddings and mock HTTP responses, but exercise real Parquet persistence, scikit-learn retrieval, MCP structured responses, legacy client compatibility, and subprocess stdio transport. They need no credentials, network access or GPU.

CI runs these checks on Python 3.11, 3.12 and 3.13, and separately builds/smoke-tests the container. GitHub Actions are pinned to commit SHAs. Dependency resolution has a recorded `exclude-newer` cutoff in `pyproject.toml`; upgrades should be deliberate, reviewed, locked and tested rather than selecting unbounded latest versions.

Code is organized under `src/langgraph_rag_mcp`: configuration and schemas, source discovery, chunking, embedding cache, snapshot publication, vector storage, retrieval, MCP server and CLI.

## Migrating from the notebook version

- `rag-tool.ipynb` is preserved as a **historical experiment**, not the supported ingestion path. Its old dependencies, imports and Claude example were not migrated; do not run it to build the new index.
- `langgraph-mcp.py` remains a compatibility entry point after installing the package. Prefer `langgraph-rag serve` for new host configurations.
- Existing root-level `sklearn_vectorstore.parquet` and `llms_full.txt` files are not overwritten or imported. Build a new index with `langgraph-rag index`; the previous chunking and metadata do not satisfy the new index format.
- `requirements.txt` is a compatibility adapter for `pip install -r requirements.txt`; use `uv sync --locked --extra embeddings` for reproducible installations.
- No Anthropic API key is required. The obsolete `tail`/`docker exec` Compose setup has been replaced.

The default embedding model is English-oriented. Use the retrieval benchmark above to assess Portuguese-to-English searches on the current corpus; deterministic unit tests alone cannot establish semantic quality. Hybrid search, reranking and comparisons with lighter or multilingual embedding models remain separate future experiments.

## Resources

- [LangChain documentation integrations](https://docs.langchain.com/use-these-docs)
- [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/)
- [Original MCP From Scratch material](https://www.notion.so/MCP-From-Scratch-1c1dd782d50180fda61ece359beef88c)

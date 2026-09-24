import logging
from threading import RLock

from langchain_core.embeddings import Embeddings
from pydantic import TypeAdapter

from langgraph_rag_mcp.embeddings import load_embeddings
from langgraph_rag_mcp.index import IndexNotReadyError, load_manifest, snapshot_path
from langgraph_rag_mcp.models import (
    IndexConfig,
    Manifest,
    Page,
    PageResponse,
    SearchHit,
    SearchResponse,
)
from langgraph_rag_mcp.settings import Settings
from langgraph_rag_mcp.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class DocumentationIndex:
    def __init__(self, settings: Settings, *, embeddings: Embeddings | None = None) -> None:
        self.settings = settings
        self._embeddings = embeddings
        self._store: VectorStore | None = None
        self._manifest: Manifest | None = None
        self._pages: dict[str, Page] = {}
        self._lock = RLock()

    def _refresh(self) -> Manifest:
        manifest = load_manifest(self.settings)
        if manifest.config != IndexConfig.from_settings(self.settings):
            raise IndexNotReadyError(
                "Documentation index is incompatible with these settings. Run: langgraph-rag index"
            )
        if self._manifest is None or self._manifest.generation != manifest.generation:
            path = snapshot_path(self.settings, manifest)
            try:
                if not (path / "vectors.parquet").is_file():
                    raise FileNotFoundError("Missing vectors")
                pages = TypeAdapter(list[Page]).validate_json(
                    (path / "pages.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise IndexNotReadyError("Documentation snapshot is incomplete or invalid") from exc
            self._pages = {page.source: page for page in pages}
            self._store = None
            self._manifest = manifest
        return manifest

    def search(self, query: str, k: int = 3) -> SearchResponse:
        query = query.strip()
        if not query or len(query) > 4000:
            raise ValueError("query must contain between 1 and 4000 characters")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        with self._lock:
            manifest = self._refresh()
            if self._store is None:
                if self._embeddings is None:
                    self._embeddings = load_embeddings(self.settings)
                try:
                    self._store = VectorStore.load(
                        snapshot_path(self.settings, manifest) / "vectors.parquet", self._embeddings
                    )
                except (OSError, ValueError) as exc:
                    raise IndexNotReadyError("Documentation vectors could not be loaded") from exc
            matches = self._store.search(query, k=min(k, manifest.chunks))
            results = [
                SearchHit(
                    id=str(document.id),
                    source=document.metadata["source"],
                    title=document.metadata["title"],
                    section=document.metadata["section"],
                    content=document.page_content,
                    score=round(max(-1.0, min(1.0, 1.0 - distance)), 6),
                )
                for document, distance in matches
            ]
        logger.info("Retrieved %d documentation chunks", len(results))
        return SearchResponse(query=query, indexed_at=manifest.created_at, results=results)

    def read_page(self, source: str, offset: int = 0, limit: int = 12000) -> PageResponse:
        if offset < 0 or not 1 <= limit <= 20000:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 20000")
        with self._lock:
            self._refresh()
            page = self._pages.get(source)
            if page is None:
                raise ValueError("Unknown source; use a source URL returned by the search tool")
            end = offset + limit
            return PageResponse(
                source=page.source,
                title=page.title,
                content=page.content[offset:end],
                offset=offset,
                next_offset=end if end < len(page.content) else None,
                total_characters=len(page.content),
            )

    def status(self) -> Manifest:
        with self._lock:
            return self._refresh()

    def full_docs(self) -> str:
        with self._lock:
            self._refresh()
            content = "\n\n---\n\n".join(
                f"SOURCE: {page.source}\n\n{page.content}" for page in self._pages.values()
            )
        if len(content) > 200000:
            raise ValueError("Corpus exceeds the resource limit; use search and read_page instead")
        return content

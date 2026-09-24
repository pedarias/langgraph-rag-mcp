import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from filelock import FileLock
from langchain_core.embeddings import Embeddings

from langgraph_rag_mcp.chunking import Tokenizer, load_tokenizer, split_pages
from langgraph_rag_mcp.embeddings import CachedEmbeddings, load_embeddings
from langgraph_rag_mcp.models import BuildResult, IndexConfig, Manifest, Page, SnapshotPointer
from langgraph_rag_mcp.settings import Settings
from langgraph_rag_mcp.sources import DocumentationLoader
from langgraph_rag_mcp.vectorstore import VectorStore


class IndexNotReadyError(RuntimeError):
    pass


def snapshot_path(settings: Settings, pointer: SnapshotPointer) -> Path:
    return settings.data_dir / "snapshots" / pointer.generation


def load_manifest(settings: Settings) -> Manifest:
    try:
        pointer = SnapshotPointer.model_validate_json(
            (settings.data_dir / "current.json").read_text(encoding="utf-8")
        )
        manifest = Manifest.model_validate_json(
            (snapshot_path(settings, pointer) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.generation != pointer.generation:
            raise ValueError("Snapshot generation mismatch")
        return manifest
    except (OSError, ValueError) as exc:
        raise IndexNotReadyError(
            "Documentation index is missing or invalid. Run: langgraph-rag index"
        ) from exc


def build_index(
    settings: Settings,
    *,
    pages: list[Page] | None = None,
    embeddings: Embeddings | None = None,
    tokenizer: Tokenizer | None = None,
) -> BuildResult:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(settings.data_dir / "index.lock", timeout=0):
        if pages is None:
            pages = DocumentationLoader(
                settings.index_url,
                max_pages=settings.max_pages,
                timeout=settings.request_timeout,
            ).load()
        if not pages:
            raise ValueError("Cannot publish an empty documentation index")
        pages = sorted(pages, key=lambda page: page.source)
        serialized_pages = json.dumps([page.model_dump() for page in pages], ensure_ascii=False)
        pages_hash = hashlib.sha256(serialized_pages.encode()).hexdigest()
        config = IndexConfig.from_settings(settings)
        if (settings.data_dir / "current.json").exists():
            previous = load_manifest(settings)
            if previous.config == config and previous.pages_hash == pages_hash:
                if not (snapshot_path(settings, previous) / "vectors.parquet").is_file():
                    raise IndexNotReadyError("Index vectors are missing; restore the snapshot")
                return BuildResult(
                    generation=previous.generation,
                    pages=previous.pages,
                    chunks=previous.chunks,
                    embedded_chunks=0,
                )
        chunks = split_pages(pages, settings, tokenizer or load_tokenizer(settings))
        cached = CachedEmbeddings(
            embeddings or load_embeddings(settings),
            settings.data_dir / "embeddings.sqlite3",
            json.dumps([config.model_name, config.model_revision, config.normalize_embeddings]),
        )
        manifest = Manifest(
            generation=uuid4().hex,
            config=config,
            created_at=datetime.now(UTC),
            pages_hash=pages_hash,
            pages=len(pages),
            chunks=len(chunks),
        )
        (settings.data_dir / "snapshots").mkdir(exist_ok=True)
        with TemporaryDirectory(dir=settings.data_dir, prefix=".build-") as temporary:
            staging = Path(temporary) / manifest.generation
            staging.mkdir()
            VectorStore.build(chunks, cached).save(staging / "vectors.parquet")
            (staging / "pages.json").write_text(serialized_pages, encoding="utf-8")
            (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            staging.rename(snapshot_path(settings, manifest))
            pointer = Path(temporary) / "current.json"
            pointer.write_text(
                SnapshotPointer(generation=manifest.generation).model_dump_json(), encoding="utf-8"
            )
            os.replace(pointer, settings.data_dir / "current.json")
        return BuildResult(
            generation=manifest.generation,
            pages=manifest.pages,
            chunks=manifest.chunks,
            embedded_chunks=cached.embedded_chunks,
        )

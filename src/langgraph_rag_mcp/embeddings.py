import json
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from langgraph_rag_mcp.settings import Settings


def load_embeddings(settings: Settings) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install local embeddings with: uv sync --extra embeddings") from exc
    return HuggingFaceEmbeddings(
        model_name=settings.model_name,
        model_kwargs={"revision": settings.model_revision, "device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


class CachedEmbeddings(Embeddings):
    def __init__(self, backend: Embeddings, path: Path, namespace: str) -> None:
        self.backend = backend
        self.path = path
        self.namespace = namespace
        self.embedded_chunks = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [sha256(f"{self.namespace}\n{text}".encode()).hexdigest() for text in texts]
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )
            cached: dict[str, list[float]] = {}
            missing: dict[str, str] = {}
            for key, text in zip(keys, texts, strict=True):
                row = connection.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
                if row:
                    cached[key] = json.loads(row[0])
                else:
                    missing[key] = text
            if missing:
                vectors = self.backend.embed_documents(list(missing.values()))
                for key, vector in zip(missing, vectors, strict=True):
                    cached[key] = vector
                    connection.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector))
                    )
                self.embedded_chunks += len(missing)
            result = [cached[key] for key in keys]
            if result:
                values = np.asarray(result, dtype=np.float64)
                if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
                    raise ValueError("Invalid embedding vectors; cache update rolled back")
            return result

    def embed_query(self, text: str) -> list[float]:
        return self.backend.embed_query(text)

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.neighbors import NearestNeighbors


class VectorStore:
    def __init__(self, documents: list[Document], vectors: list[list[float]], embeddings: Embeddings) -> None:
        self.documents = documents
        self.vectors = np.asarray(vectors, dtype=np.float64)
        if (
            not documents
            or self.vectors.ndim != 2
            or len(documents) != len(self.vectors)
            or not np.isfinite(self.vectors).all()
        ):
            raise ValueError("Invalid documentation vectors")
        self.embeddings = embeddings
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute").fit(self.vectors)

    @classmethod
    def build(cls, documents: list[Document], embeddings: Embeddings) -> "VectorStore":
        return cls(
            documents,
            embeddings.embed_documents([document.page_content for document in documents]),
            embeddings,
        )

    def save(self, path: Path) -> None:
        table = pa.table(
            {
                "id": [document.id for document in self.documents],
                "content": [document.page_content for document in self.documents],
                "metadata": [json.dumps(document.metadata) for document in self.documents],
                "embedding": self.vectors.tolist(),
            }
        )
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path, embeddings: Embeddings) -> "VectorStore":
        data = pq.read_table(path).to_pydict()
        if set(data) != {"id", "content", "metadata", "embedding"}:
            raise ValueError("Invalid vector index columns")
        documents = [
            Document(id=identifier, page_content=content, metadata=json.loads(metadata))
            for identifier, content, metadata in zip(
                data["id"], data["content"], data["metadata"], strict=True
            )
        ]
        for document in documents:
            if not document.id or not {"source", "title", "section"} <= document.metadata.keys():
                raise ValueError("Invalid vector index metadata")
        return cls(documents, data["embedding"], embeddings)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        distances, indices = self.neighbors.kneighbors(
            [self.embeddings.embed_query(query)], n_neighbors=min(k, len(self.documents))
        )
        return [
            (self.documents[int(index)], float(distance))
            for index, distance in zip(indices[0], distances[0], strict=True)
        ]

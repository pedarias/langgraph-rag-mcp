from hashlib import sha256
from typing import Protocol

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langgraph_rag_mcp.models import Page
from langgraph_rag_mcp.settings import Settings


class Tokenizer(Protocol):
    model_max_length: int

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...


def load_tokenizer(settings: Settings) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install local embeddings with: uv sync --extra embeddings") from exc
    return AutoTokenizer.from_pretrained(
        settings.model_name, revision=settings.model_revision, trust_remote_code=False
    )


def split_pages(pages: list[Page], settings: Settings, tokenizer: Tokenizer) -> list[Document]:
    if settings.chunk_size > tokenizer.model_max_length:
        raise ValueError("chunk_size exceeds the embedding tokenizer window")

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "heading"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=token_count,
    )
    chunks: dict[str, Document] = {}
    for page in pages:
        for section in headers.split_text(page.content):
            section_name = " / ".join(
                section.metadata[key] for key in ("section", "subsection") if key in section.metadata
            )
            for text in splitter.split_text(section.page_content):
                if token_count(text) > settings.chunk_size:
                    raise ValueError("A chunk exceeds the embedding tokenizer window")
                chunk_id = sha256(f"{page.source}\n{section_name}\n{text}".encode()).hexdigest()
                chunks[chunk_id] = Document(
                    id=chunk_id,
                    page_content=text,
                    metadata={
                        "source": page.source,
                        "title": page.title,
                        "section": section_name,
                    },
                )
    if not chunks:
        raise ValueError("No indexable chunks found")
    return list(chunks.values())

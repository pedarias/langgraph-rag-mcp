import os
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGGRAPH_MCP_", extra="ignore")

    data_dir: Path = Field(
        default_factory=lambda: (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "langgraph-rag-mcp"
        )
    )
    index_url: str = "https://docs.langchain.com/oss/python/langgraph/llms.txt"
    model_name: str = "BAAI/bge-large-en-v1.5"
    model_revision: str = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    chunk_size: int = Field(default=450, ge=16)
    chunk_overlap: int = Field(default=50, ge=0)
    max_pages: int = Field(default=200, ge=1, le=2000)
    request_timeout: float = Field(default=30, gt=0, le=300)

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        url = urlsplit(self.index_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not url.path.endswith("/llms.txt")
        ):
            raise ValueError("index_url must be an HTTPS llms.txt URL without credentials or query")
        self.data_dir = self.data_dir.expanduser().resolve()
        return self

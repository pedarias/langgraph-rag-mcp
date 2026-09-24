import hashlib
import json
import math
import os
import platform
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langgraph_rag_mcp import __version__
from langgraph_rag_mcp.models import Manifest, SearchHit
from langgraph_rag_mcp.retrieval import DocumentationIndex


def canonical_source(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise ValueError("Expected a public HTTPS source URL without credentials or query parameters")
    path = parsed.path.rstrip("/").removesuffix(".md")
    return urlunsplit(parsed._replace(netloc=parsed.netloc.lower(), path=path, fragment=""))


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    language: Literal["en", "pt"]
    query: str = Field(min_length=1, max_length=4000)
    expected_sources: list[str] = Field(min_length=1)

    @field_validator("expected_sources")
    @classmethod
    def normalize_sources(cls, sources: list[str]) -> list[str]:
        normalized = [canonical_source(source) for source in sources]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Expected sources must be unique")
        return normalized


class BenchmarkDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    description: str = ""
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Benchmark case IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class CaseResult(BenchmarkCase):
    latency_ms: float = Field(ge=0)
    results: list[SearchHit]

    def first_relevant_rank(self, k: int) -> int | None:
        expected = {canonical_source(source) for source in self.expected_sources}
        return next(
            (
                rank
                for rank, hit in enumerate(self.results[:k], 1)
                if canonical_source(hit.source) in expected
            ),
            None,
        )


class Metrics(BaseModel):
    cases: int
    hit_rate: dict[int, float]
    mrr: dict[int, float]
    duplicate_source_fraction: dict[int, float]
    latency_p50_ms: float
    latency_p95_ms: float


class EvaluationReport(BaseModel):
    schema_version: Literal[1] = 1
    evaluated_at: datetime
    dataset_name: str
    dataset_version: int
    dataset_sha256: str
    implementation_sha256: str
    index: Manifest
    runtime: dict[str, str]
    warmup_ms: float
    max_k: int
    groups: dict[str, Metrics]
    misses_at_max_k: list[str]
    cases: list[CaseResult]


def load_dataset(path: Path | None = None) -> BenchmarkDataset:
    source = (
        path if path is not None else resources.files("langgraph_rag_mcp").joinpath("data/benchmark.json")
    )
    return BenchmarkDataset.model_validate_json(source.read_text(encoding="utf-8"))


def summarize(cases: list[CaseResult], cutoffs: list[int]) -> Metrics:
    if not cases or not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("Metrics require cases and positive cutoffs")
    hit_rate = {}
    mrr = {}
    duplicates = {}
    for k in cutoffs:
        ranks = [case.first_relevant_rank(k) for case in cases]
        hit_rate[k] = round(mean(rank is not None for rank in ranks), 6)
        mrr[k] = round(mean(1 / rank if rank is not None else 0 for rank in ranks), 6)
        fractions = []
        for case in cases:
            sources = [canonical_source(hit.source) for hit in case.results[:k]]
            fractions.append(1 - len(set(sources)) / len(sources) if sources else 0)
        duplicates[k] = round(mean(fractions), 6)
    latencies = sorted(case.latency_ms for case in cases)
    return Metrics(
        cases=len(cases),
        hit_rate=hit_rate,
        mrr=mrr,
        duplicate_source_fraction=duplicates,
        latency_p50_ms=round(median(latencies), 3),
        latency_p95_ms=round(latencies[math.ceil(0.95 * len(latencies)) - 1], 3),
    )


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for module in sorted(resources.files("langgraph_rag_mcp").iterdir(), key=lambda item: item.name):
        if module.name.endswith(".py"):
            digest.update(module.name.encode())
            digest.update(module.read_bytes())
    return digest.hexdigest()


def runtime_info() -> dict[str, str]:
    info = {
        "application": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in (
        "mcp",
        "langchain-huggingface",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "scikit-learn",
        "pyarrow",
    ):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "not installed"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE"):
        info[variable] = os.environ.get(variable, "unset")
    return info


def evaluate(index: DocumentationIndex, dataset: BenchmarkDataset, max_k: int = 5) -> EvaluationReport:
    if not 1 <= max_k <= 20:
        raise ValueError("max_k must be between 1 and 20")
    manifest = index.status()
    expected = {source for case in dataset.cases for source in case.expected_sources}
    for source in sorted(expected):
        try:
            index.read_page(source, limit=1)
        except ValueError as exc:
            raise ValueError(f"Benchmark source is absent from the current index: {source}") from exc
    start = perf_counter()
    warmup = index.search(dataset.cases[0].query, k=max_k)
    warmup_ms = (perf_counter() - start) * 1000
    if warmup.indexed_at != manifest.created_at:
        raise RuntimeError("Index changed during evaluation; rerun against a stable snapshot")
    cases = []
    for case in dataset.cases:
        start = perf_counter()
        response = index.search(case.query, k=max_k)
        latency_ms = (perf_counter() - start) * 1000
        if response.indexed_at != manifest.created_at:
            raise RuntimeError("Index changed during evaluation; rerun against a stable snapshot")
        cases.append(
            CaseResult(**case.model_dump(), latency_ms=round(latency_ms, 3), results=response.results)
        )
    if index.status().generation != manifest.generation:
        raise RuntimeError("Index changed during evaluation; rerun against a stable snapshot")
    cutoffs = sorted({k for k in (1, 3, 5, max_k) if k <= max_k})
    groups = {"all": summarize(cases, cutoffs)}
    for language in sorted({case.language for case in cases}):
        groups[language] = summarize([case for case in cases if case.language == language], cutoffs)
    return EvaluationReport(
        evaluated_at=datetime.now(UTC),
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=dataset.fingerprint,
        implementation_sha256=implementation_fingerprint(),
        index=manifest,
        runtime=runtime_info(),
        warmup_ms=round(warmup_ms, 3),
        max_k=max_k,
        groups=groups,
        misses_at_max_k=[case.id for case in cases if case.first_relevant_rank(max_k) is None],
        cases=cases,
    )

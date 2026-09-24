import hashlib
import logging
import re
from contextlib import nullcontext
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx

from langgraph_rag_mcp.models import Page

logger = logging.getLogger(__name__)
LINK = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^\s)]+)\)", re.MULTILINE)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def clean_markdown(text: str) -> str:
    text = re.sub(r"\A> ## Documentation Index\n(?:>[^\n]*\n)*\s*", "", text)
    text = text.split('<div class="source-links">', 1)[0].strip()
    return re.sub(r"\n\*\*\*\s*$", "", text).strip()


class DocumentationLoader:
    def __init__(
        self,
        index_url: str,
        *,
        client: httpx.Client | None = None,
        max_pages: int = 200,
        timeout: float = 30,
    ) -> None:
        self.index_url = index_url
        self.origin = urlsplit(index_url)
        self.prefix = self.origin.path.rsplit("/", 1)[0] + "/"
        self.client = client
        self.max_pages = max_pages
        self.timeout = timeout

    def _same_origin(self, url: str) -> bool:
        parsed = urlsplit(url)
        path = unquote(parsed.path)
        return (
            parsed.scheme == "https"
            and parsed.netloc == self.origin.netloc
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and ".." not in path.split("/")
            and "\\" not in path
        )

    def _in_scope(self, url: str) -> bool:
        return self._same_origin(url) and unquote(urlsplit(url).path).startswith(self.prefix)

    def _fetch(self, client: httpx.Client, url: str) -> tuple[str, str]:
        markdown_page = urlsplit(url).path.endswith(".md")
        for _ in range(6):
            if not self._same_origin(url):
                raise ValueError("Documentation URL redirects outside the configured source origin")
            with client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect:
                    target = urlsplit(urljoin(url, response.headers["location"]))
                    if markdown_page and not target.path.endswith(".md"):
                        target = target._replace(path=target.path.rstrip("/") + ".md")
                    url = urlunsplit(target._replace(fragment=""))
                    continue
                response.raise_for_status()
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_RESPONSE_BYTES:
                        raise ValueError("Documentation response exceeds the 2 MiB limit")
                text = chunks.decode("utf-8")
                if "text/html" in response.headers.get("content-type", "") or re.search(
                    r"<!doctype\s+html|<html\b", text[:1024], re.IGNORECASE
                ):
                    raise ValueError("Expected Markdown documentation, received HTML")
                return text, url
        raise ValueError("Too many documentation redirects")

    def load(self) -> list[Page]:
        context = (
            nullcontext(self.client)
            if self.client is not None
            else httpx.Client(
                timeout=self.timeout,
                transport=httpx.HTTPTransport(retries=2),
                headers={"User-Agent": "langgraph-rag-mcp/0.2"},
            )
        )
        with context as client:
            pending = [self.index_url]
            seen_indexes: set[str] = set()
            links: dict[str, str] = {}
            while pending:
                index = pending.pop()
                if index in seen_indexes:
                    continue
                seen_indexes.add(index)
                if len(seen_indexes) > 20:
                    raise ValueError("Documentation source exceeds the 20-index limit")
                text, final_url = self._fetch(client, index)
                for title, href in LINK.findall(text):
                    parsed = urlsplit(urljoin(final_url, href))
                    url = urlunsplit(parsed._replace(fragment=""))
                    if not self._in_scope(url):
                        continue
                    if parsed.path.endswith("/llms.txt"):
                        pending.append(url)
                    elif parsed.path.endswith(".md"):
                        links.setdefault(url, title)
                if len(links) > self.max_pages:
                    raise ValueError("Documentation source exceeds max_pages; no index was published")
            if not links:
                raise ValueError("No Markdown pages found in the documentation index")
            pages = []
            seen_content: set[str] = set()
            for url, title in sorted(links.items()):
                logger.info("Fetching %s", url)
                text, final_url = self._fetch(client, url)
                content = clean_markdown(text)
                if not content:
                    raise ValueError(f"Empty documentation page: {url}")
                digest = hashlib.sha256(content.encode()).hexdigest()
                if digest in seen_content:
                    continue
                seen_content.add(digest)
                heading = re.search(r"^# (.+)$", content, re.MULTILINE)
                pages.append(
                    Page(
                        source=final_url.removesuffix(".md"),
                        title=heading.group(1) if heading else title,
                        content=content,
                    )
                )
            return pages

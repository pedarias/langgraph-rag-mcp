import httpx
import pytest

from langgraph_rag_mcp.sources import DocumentationLoader

INDEX = "https://docs.example.test/langgraph/llms.txt"


def loader(responses: dict[str, httpx.Response]) -> DocumentationLoader:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) in responses
        return responses[str(request.url)]

    return DocumentationLoader(INDEX, client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_discovers_markdown_and_nested_indexes_without_leaving_scope() -> None:
    source = loader(
        {
            INDEX: httpx.Response(
                200,
                text="\n".join(
                    [
                        "- [Memory](memory.md)",
                        "- [Memory again](memory.md#usage)",
                        "- [More](guides/llms.txt)",
                        "- [External](https://elsewhere.test/page.md)",
                        "- [Other project](/other/page.md)",
                    ]
                ),
            ),
            "https://docs.example.test/langgraph/guides/llms.txt": httpx.Response(
                200, text="- [Stream](../stream.md)"
            ),
            "https://docs.example.test/langgraph/memory.md": httpx.Response(
                200, text="# Memory\n\nPersistent state."
            ),
            "https://docs.example.test/langgraph/stream.md": httpx.Response(
                200, text="# Stream\n\nToken streaming."
            ),
        }
    )
    pages = source.load()
    assert [page.title for page in pages] == ["Memory", "Stream"]
    assert pages[0].source == "https://docs.example.test/langgraph/memory"


def test_rejects_cross_origin_redirect_before_following_it() -> None:
    source = loader({INDEX: httpx.Response(302, headers={"location": "http://127.0.0.1/"})})
    with pytest.raises(ValueError, match="outside"):
        source.load()


def test_deduplicates_page_content_and_strips_boilerplate() -> None:
    content = (
        "> ## Documentation Index\n> Fetch the complete documentation index at: elsewhere\n\n"
        "# Memory\n\nPersistent state.\n\n***\n\n"
        '<div class="source-links">Edit this page</div>'
    )
    source = loader(
        {
            INDEX: httpx.Response(200, text="- [A](a.md)\n- [B](b.md)"),
            "https://docs.example.test/langgraph/a.md": httpx.Response(200, text=content),
            "https://docs.example.test/langgraph/b.md": httpx.Response(200, text=content),
        }
    )
    pages = source.load()
    assert len(pages) == 1
    assert pages[0].content == "# Memory\n\nPersistent state."


@pytest.mark.parametrize(
    "response",
    [httpx.Response(404), httpx.Response(200, text="<html>Sign in</html>")],
)
def test_does_not_index_error_or_html_pages(response: httpx.Response) -> None:
    source = loader(
        {
            INDEX: httpx.Response(200, text="- [Page](page.md)"),
            "https://docs.example.test/langgraph/page.md": response,
        }
    )
    with pytest.raises((ValueError, httpx.HTTPStatusError)):
        source.load()


@pytest.mark.parametrize(
    "index_url,target",
    [
        (INDEX, "https://DOCS.EXAMPLE.TEST/langgraph/page.md"),
        (INDEX, "https://docs.example.test:443/langgraph/page.md"),
        ("https://DOCS.EXAMPLE.TEST:443/langgraph/llms.txt", "https://docs.example.test/langgraph/page.md"),
        (
            "https://docs.example.test:8443/langgraph/llms.txt",
            "https://DOCS.EXAMPLE.TEST:8443/langgraph/page.md",
        ),
    ],
)
def test_equivalent_url_origins_are_accepted(index_url: str, target: str) -> None:
    assert DocumentationLoader(index_url)._in_scope(target)


@pytest.mark.parametrize("authority", ["DOCS.EXAMPLE.TEST", "docs.example.test:443", "DOCS.EXAMPLE.TEST:443"])
def test_discovers_and_follows_equivalent_url_origins(authority: str) -> None:
    source = loader(
        {
            INDEX: httpx.Response(200, text=f"- [Page](https://{authority}/langgraph/page.md)"),
            "https://docs.example.test/langgraph/page.md": httpx.Response(
                302, headers={"location": f"https://{authority}/langgraph/relocated"}
            ),
            "https://docs.example.test/langgraph/relocated.md": httpx.Response(
                200, text="# Page\n\nDocumentation."
            ),
        }
    )
    pages = source.load()
    assert len(pages) == 1
    assert pages[0].title == "Page"
    assert pages[0].source.endswith("/langgraph/relocated")


@pytest.mark.parametrize("port", ["443", "0", "invalid", "65536"])
def test_custom_origin_port_does_not_match_other_ports(port: str) -> None:
    source = DocumentationLoader("https://docs.example.test:8443/langgraph/llms.txt")
    assert not source._same_origin(f"https://docs.example.test:{port}/langgraph/page.md")


def test_empty_index_fails_instead_of_publishing_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="No Markdown"):
        loader({INDEX: httpx.Response(200, text="# Empty")}).load()


def test_follows_relocated_markdown_without_requesting_html() -> None:
    source = loader(
        {
            INDEX: httpx.Response(200, text="- [Changelog](changelog.md)"),
            "https://docs.example.test/langgraph/changelog.md": httpx.Response(
                307, headers={"location": "https://docs.example.test/releases/changelog"}
            ),
            "https://docs.example.test/releases/changelog.md": httpx.Response(
                200, text="# Changelog\n\nRelease notes."
            ),
        }
    )
    pages = source.load()
    assert pages[0].source == "https://docs.example.test/releases/changelog"
    assert pages[0].title == "Changelog"


def test_page_limit_aborts_before_fetching_partial_corpus() -> None:
    source = loader({INDEX: httpx.Response(200, text="- [A](a.md)\n- [B](b.md)")})
    source.max_pages = 1
    with pytest.raises(ValueError, match="max_pages"):
        source.load()


@pytest.mark.parametrize(
    "target",
    [
        "https://elsewhere.test/page",
        "http://docs.example.test/page",
        "https://user@docs.example.test/page",
        "https://@docs.example.test/page",
        "https://docs.example.test:8443/page",
        "https://docs.example.test:0/page",
        "https://docs.example.test:invalid/page",
        "https://docs.example.test:65536/page",
        "https://docs.example.test/page?query=1",
        "https://docs.example.test/%2e%2e/page",
        "https://docs.example.test/path%5cpage",
    ],
)
def test_page_redirects_cannot_escape_origin(target: str) -> None:
    source = loader(
        {
            INDEX: httpx.Response(200, text="- [Page](page.md)"),
            "https://docs.example.test/langgraph/page.md": httpx.Response(302, headers={"location": target}),
        }
    )
    with pytest.raises((ValueError, httpx.RemoteProtocolError), match="outside|Invalid URL"):
        source.load()

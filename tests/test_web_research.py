"""Fallback web configurable con backend simulado y trazabilidad."""

from typing import Any

from agents.web_research import ConfiguredWebSearchProvider, WebSearchConfig


class FakeBackend:
    def __init__(self, results: list[dict[str, str]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, max_results: int):
        self.calls.append((query, max_results))
        return self.results


def result(title: str, url: str, snippet: str = "Technical fragment") -> dict[str, str]:
    return {"title": title, "url": url, "snippet": snippet, "source": "simulated"}


def test_prioritizes_configured_technology_and_rag_official_domains() -> None:
    backend = FakeBackend(
        [
            result("General", "https://allowed.test/guide"),
            result("Official", "https://official.test/reference"),
            result("RAG official", "https://rag-official.test/api"),
        ]
    )
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(
            allowed_domains=("allowed.test", "official.test", "rag-official.test"),
            priority_domains=("official.test",),
            technology_domains={"runtime-x": ("allowed.test",)},
            max_results=3,
        ),
        backend=backend,
    )

    evidence = provider.search_context(
        "error handling",
        technologies=("runtime-x",),
        rag_metadata=(
            {
                "source_type": "official_documentation",
                "path_or_url": "https://rag-official.test/index",
                "tags": [],
            },
        ),
    )

    executed_query = backend.calls[0][0]
    assert "site:official.test" in executed_query
    assert "site:allowed.test" in executed_query
    assert "site:rag-official.test" in executed_query
    assert [item.reference for item in evidence] == [
        "https://allowed.test/guide",
        "https://official.test/reference",
        "https://rag-official.test/api",
    ]
    assert all(item.relevance == 0.8 for item in evidence)


def test_enforces_allowed_and_blocked_domains() -> None:
    backend = FakeBackend(
        [
            result("Allowed", "https://docs.allowed.test/item"),
            result("Blocked", "https://blocked.allowed.test/item"),
            result("Outside", "https://outside.test/item"),
        ]
    )
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(
            allowed_domains=("allowed.test",),
            blocked_domains=("blocked.allowed.test",),
        ),
        backend=backend,
    )

    evidence = provider.search("query")

    assert [item.reference for item in evidence] == ["https://docs.allowed.test/item"]


def test_limits_and_deduplicates_results_by_canonical_url() -> None:
    backend = FakeBackend(
        [
            result("First", "https://docs.test/path#section"),
            result("Duplicate", "https://docs.test/path"),
            result("Second", "https://docs.test/second"),
            result("Third", "https://docs.test/third"),
        ]
    )
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(max_results=2), backend=backend
    )

    evidence = provider.search("query", limit=8)
    audit = provider.search_audit()

    assert len(evidence) == 2
    assert audit is not None
    assert len(audit["found"]) == 2
    assert len(audit["used"]) == 2
    assert backend.calls == [("query", 2)]


def test_trace_records_query_url_title_fragment_found_and_used() -> None:
    backend = FakeBackend(
        [result("Official title", "https://docs.test/reference", "Relevant snippet")]
    )
    provider = ConfiguredWebSearchProvider(backend=backend)

    provider.search("specific query", limit=1)
    audit: dict[str, Any] = dict(provider.search_audit() or {})

    assert audit["query"] == "specific query"
    assert audit["executed_queries"] == ["specific query"]
    assert audit["found"][0] == {
        "url": "https://docs.test/reference",
        "title": "Official title",
        "snippet": "Relevant snippet",
        "domain": "docs.test",
        "priority": False,
    }
    assert audit["used"] == audit["found"]
    assert audit["conclusions"]

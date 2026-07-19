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


def test_config_from_dict_normalizes_all_dynamic_options() -> None:
    config = WebSearchConfig.from_dict(
        {
            "allowed_domains": ["WWW.Allowed.Test", "allowed.test"],
            "priority_domains": "https://priority.test/docs",
            "blocked_domains": ["blocked.test"],
            "max_results": 3,
            "technology_domains": {
                "Runtime-X": ["WWW.RUNTIME.TEST", "runtime.test"],
            },
        }
    )

    assert config.allowed_domains == ("allowed.test",)
    assert config.priority_domains == ("priority.test",)
    assert config.blocked_domains == ("blocked.test",)
    assert config.max_results == 3
    assert config.technology_domains == {"runtime-x": ("runtime.test",)}


def test_does_not_repeat_identical_priority_and_general_queries() -> None:
    backend = FakeBackend([])
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(
            allowed_domains=("docs.test",),
            priority_domains=("docs.test",),
        ),
        backend=backend,
    )

    provider.search("query")

    assert backend.calls == [("query (site:docs.test)", 5)]


def test_blocked_domains_override_allowed_and_priority_domains() -> None:
    backend = FakeBackend(
        [
            result("Allowed priority", "https://priority.test/guide"),
            result("Blocked priority", "https://blocked.priority.test/guide"),
        ]
    )
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(
            allowed_domains=("priority.test",),
            priority_domains=("priority.test", "blocked.priority.test"),
            blocked_domains=("blocked.priority.test",),
            max_results=2,
        ),
        backend=backend,
    )

    evidence = provider.search("query")

    assert "site:priority.test" in backend.calls[0][0]
    assert "site:blocked.priority.test" not in backend.calls[0][0]
    assert [item.reference for item in evidence] == ["https://priority.test/guide"]


def test_global_result_limit_applies_when_multiple_queries_are_combined() -> None:
    class SequencedBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def __call__(self, query: str, max_results: int):
            self.calls.append((query, max_results))
            if len(self.calls) == 1:
                return [result("Priority", "https://priority.test/one")]
            return [
                result("General one", "https://allowed.test/two"),
                result("General two", "https://allowed.test/three"),
                result("General three", "https://allowed.test/four"),
            ]

    backend = SequencedBackend()
    provider = ConfiguredWebSearchProvider(
        WebSearchConfig(
            allowed_domains=("priority.test", "allowed.test"),
            priority_domains=("priority.test",),
            max_results=3,
        ),
        backend=backend,
    )

    evidence = provider.search("query", limit=3)
    audit = provider.search_audit()

    assert len(backend.calls) == 2
    assert len(evidence) == 3
    assert audit is not None
    assert len(audit["found"]) == 3
    assert len(audit["used"]) == 3

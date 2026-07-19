"""Tests de Tavily completamente aislados de internet."""

from unittest.mock import Mock, patch

import pytest

from tools.web_tools import (
    MAX_SNIPPET_LENGTH,
    SEARCH_TIMEOUT_SECONDS,
    WebSearchConfigurationError,
    WebSearchNetworkError,
    WebSearchResponseError,
    WebSearchValidationError,
    web_search,
)


def test_returns_structured_results_without_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = Mock()
    client.search.return_value = {
        "answer": "campo ignorado",
        "results": [
            {
                "title": "Documentación",
                "url": "https://www.example.com/docs",
                "content": "Un   resumen\n breve.",
                "raw_content": "<html>página completa</html>",
                "score": 0.9,
            }
        ],
    }

    with patch("tools.web_tools.TavilyClient", return_value=client) as client_class:
        result = web_search("  Python actual  ")

    assert result == [{
        "title": "Documentación",
        "url": "https://www.example.com/docs",
        "snippet": "Un resumen breve.",
        "source": "example.com",
    }]
    assert "raw_content" not in result[0]
    client_class.assert_called_once_with(api_key="test-key")
    client.search.assert_called_once_with(
        query="Python actual",
        max_results=5,
        timeout=SEARCH_TIMEOUT_SECONDS,
        include_answer=False,
        include_raw_content=False,
    )


def test_limits_output_count_and_snippet_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = Mock()
    client.search.return_value = {
        "results": [
            {"title": f"R{i}", "url": f"https://site{i}.test", "content": "x" * 900}
            for i in range(4)
        ]
    }
    with patch("tools.web_tools.TavilyClient", return_value=client):
        result = web_search("consulta", max_results=2)

    assert len(result) == 2
    assert len(result[0]["snippet"]) == MAX_SNIPPET_LENGTH + 1
    assert result[0]["snippet"].endswith("…")


@pytest.mark.parametrize("max_results", [0, 11, True, 1.5])
def test_rejects_invalid_max_results(
    monkeypatch: pytest.MonkeyPatch, max_results: object
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    with patch("tools.web_tools.TavilyClient") as client_class:
        with pytest.raises(WebSearchValidationError, match="entre 1 y 10"):
            web_search("consulta", max_results=max_results)  # type: ignore[arg-type]

    client_class.assert_not_called()


def test_rejects_empty_query_before_creating_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    with patch("tools.web_tools.TavilyClient") as client_class:
        with pytest.raises(WebSearchValidationError, match="vacía"):
            web_search("   ")
    client_class.assert_not_called()


def test_missing_api_key_is_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(WebSearchConfigurationError, match="TAVILY_API_KEY"):
        web_search("consulta")


def test_network_error_is_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = Mock()
    client.search.side_effect = TimeoutError("timeout")
    with patch("tools.web_tools.TavilyClient", return_value=client):
        with pytest.raises(WebSearchNetworkError, match="Tavily"):
            web_search("consulta")


@pytest.mark.parametrize(
    "response",
    [None, {}, {"results": "invalid"}, {"results": [None]},
     {"results": [{"title": "Sin campos"}]}],
)
def test_invalid_provider_response_is_controlled(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = Mock()
    client.search.return_value = response
    with patch("tools.web_tools.TavilyClient", return_value=client):
        with pytest.raises(WebSearchResponseError):
            web_search("consulta")

"""Búsqueda web acotada mediante la API de Tavily."""

from __future__ import annotations

import os
from typing import Any, TypedDict
from urllib.parse import urlparse

from tavily import TavilyClient


DEFAULT_MAX_RESULTS = 5
MIN_RESULTS = 1
MAX_RESULTS = 10
SEARCH_TIMEOUT_SECONDS = 10.0
MAX_SNIPPET_LENGTH = 500


class WebSearchError(RuntimeError):
    """Error base controlado de la tool de búsqueda."""


class WebSearchConfigurationError(WebSearchError):
    """Falta configuración necesaria para consultar Tavily."""


class WebSearchValidationError(WebSearchError):
    """La consulta o sus límites son inválidos."""


class WebSearchNetworkError(WebSearchError):
    """Tavily no respondió correctamente a la solicitud."""


class WebSearchResponseError(WebSearchError):
    """Tavily devolvió datos con un formato inesperado."""


class WebSearchResult(TypedDict):
    """Resultado breve y normalizado expuesto al coding agent."""

    title: str
    url: str
    snippet: str
    source: str


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[WebSearchResult]:
    """Busca en la web sin devolver contenido completo de las páginas."""
    normalized_query = query.strip() if isinstance(query, str) else ""
    if not normalized_query:
        raise WebSearchValidationError("La consulta de búsqueda no puede estar vacía.")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not MIN_RESULTS <= max_results <= MAX_RESULTS
    ):
        raise WebSearchValidationError(
            f"max_results debe estar entre {MIN_RESULTS} y {MAX_RESULTS}."
        )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchConfigurationError(
            "Falta la variable de entorno TAVILY_API_KEY."
        )

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=normalized_query,
            max_results=max_results,
            timeout=SEARCH_TIMEOUT_SECONDS,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as error:
        raise WebSearchNetworkError(f"Error al consultar Tavily: {error}") from error

    return _normalize_response(response, max_results)


def _normalize_response(
    response: Any, max_results: int
) -> list[WebSearchResult]:
    """Valida la respuesta de Tavily y descarta campos voluminosos."""
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise WebSearchResponseError("Tavily devolvió una respuesta inválida.")

    normalized: list[WebSearchResult] = []
    for item in response["results"][:max_results]:
        if not isinstance(item, dict):
            raise WebSearchResponseError("Tavily devolvió un resultado inválido.")
        title = item.get("title")
        url = item.get("url")
        content = item.get("content")
        if not all(isinstance(value, str) for value in (title, url, content)):
            raise WebSearchResponseError("Un resultado de Tavily tiene campos inválidos.")

        snippet = " ".join(content.split())
        if len(snippet) > MAX_SNIPPET_LENGTH:
            snippet = f"{snippet[:MAX_SNIPPET_LENGTH]}…"
        hostname = urlparse(url).hostname
        if not hostname:
            raise WebSearchResponseError("Tavily devolvió una URL inválida.")
        source = hostname.removeprefix("www.")
        normalized.append(
            {"title": title, "url": url, "snippet": snippet, "source": source}
        )

    return normalized

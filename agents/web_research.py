"""Fallback web configurable que adapta la búsqueda existente para Researcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from time import perf_counter

from core.research_ports import EvidenceFragment, WebSearchProvider
from core.validation import normalize_domain
from tools.web_tools import WebSearchResult, web_search
from core.observability import NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation


SearchBackend = Callable[[str, int], Sequence[WebSearchResult]]
_domain = normalize_domain


@dataclass(frozen=True)
class WebSearchConfig:
    allowed_domains: tuple[str, ...] = ()
    priority_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    max_results: int = 5
    technology_domains: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_results <= 10:
            raise ValueError("max_results debe estar entre 1 y 10.")
        for field in ("allowed_domains", "priority_domains", "blocked_domains"):
            object.__setattr__(
                self, field, tuple(dict.fromkeys(_domain(item) for item in getattr(self, field)))
            )
        normalized = {
            str(technology).casefold(): tuple(dict.fromkeys(_domain(item) for item in domains))
            for technology, domains in (self.technology_domains or {}).items()
        }
        object.__setattr__(self, "technology_domains", normalized)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WebSearchConfig:
        def values(name: str) -> tuple[str, ...]:
            raw = data.get(name, ())
            return (raw,) if isinstance(raw, str) else tuple(raw)

        technology_domains = {
            str(key): ((value,) if isinstance(value, str) else tuple(value))
            for key, value in data.get("technology_domains", {}).items()
        }
        return cls(
            allowed_domains=values("allowed_domains"),
            priority_domains=values("priority_domains"),
            blocked_domains=values("blocked_domains"),
            max_results=int(data.get("max_results", 5)),
            technology_domains=technology_domains,
        )


@dataclass(frozen=True)
class WebResultRecord:
    url: str
    title: str
    snippet: str
    domain: str
    priority: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class WebSearchTrace:
    query: str
    executed_queries: tuple[str, ...]
    found: tuple[WebResultRecord, ...]
    used: tuple[WebResultRecord, ...]
    conclusions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "executed_queries": list(self.executed_queries),
            "found": [item.to_dict() for item in self.found],
            "used": [item.to_dict() for item in self.used],
            "conclusions": list(self.conclusions),
        }


class ConfiguredWebSearchProvider(WebSearchProvider):
    """Prioriza y filtra resultados sin conocer dominios tecnológicos concretos."""

    def __init__(
        self,
        config: WebSearchConfig | None = None,
        *,
        backend: SearchBackend = web_search,
        observability: ObservabilityClient | None = None,
    ) -> None:
        self.config = config or WebSearchConfig()
        self.backend = backend
        self.last_trace: WebSearchTrace | None = None
        self.observability = observability or NoOpObservabilityClient()

    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        return self.search_context(query, limit=limit)

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        technologies: Sequence[str] = (),
        rag_metadata: Sequence[Mapping[str, Any]] = (),
    ) -> Sequence[EvidenceFragment]:
        started = perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query no puede estar vacía.")
        effective_limit = min(max(1, limit), self.config.max_results)
        priorities = self._priority_domains(technologies, rag_metadata)
        executed: list[str] = []
        raw_results: list[WebSearchResult] = []
        if priorities:
            prioritized_query = self._scoped_query(query, priorities)
            executed.append(prioritized_query)
            raw_results.extend(self.backend(prioritized_query, self.config.max_results))
        broad_domains = self.config.allowed_domains
        valid_priority_count = len(self._unique_records(raw_results, priorities))
        if valid_priority_count < effective_limit or not priorities:
            broad_query = self._scoped_query(query, broad_domains) if broad_domains else query
            if broad_query not in executed:
                executed.append(broad_query)
                raw_results.extend(self.backend(broad_query, self.config.max_results))

        unique = self._unique_records(raw_results, priorities)
        found = tuple(
            sorted(unique.values(), key=lambda item: (not item.priority, item.domain, item.url))
        )[: self.config.max_results]
        used = found[:effective_limit]
        conclusions = (
            f"Se encontraron {len(found)} resultados web válidos.",
            f"Se utilizaron {len(used)} resultados tras filtros y deduplicación.",
        )
        self.last_trace = WebSearchTrace(query.strip(), tuple(executed), found, used, conclusions)
        emit_observation(
            self.observability,
            ObservabilityEvent(
                "web", "web-search",
                payload={"query": query.strip(),
                         "priority_domains": priorities,
                         "allowed_domains": self.config.allowed_domains,
                         "blocked_domains": self.config.blocked_domains,
                         "result_count": len(found),
                         "used_results": [item.url for item in used],
                         "fallback_reason": "insufficient_priority_results" if len(executed) > 1 else None,
                         "available": True},
                latency_ms=(perf_counter() - started) * 1000,
            ),
        )
        return tuple(
            EvidenceFragment("web", item.url, f"{item.title}\n{item.snippet}", 0.8 if item.priority else 0.6)
            for item in used
        )

    def search_audit(self) -> Mapping[str, Any] | None:
        return self.last_trace.to_dict() if self.last_trace else None

    def _priority_domains(
        self,
        technologies: Sequence[str],
        rag_metadata: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        domains = list(self.config.priority_domains)
        mapping = self.config.technology_domains or {}
        for technology in technologies:
            domains.extend(mapping.get(technology.casefold(), ()))
        for metadata in rag_metadata:
            tags = {str(tag).casefold() for tag in metadata.get("tags", ())}
            source_type = str(metadata.get("source_type", "")).casefold()
            if "official" not in tags and "official" not in source_type:
                continue
            location = str(metadata.get("path_or_url", ""))
            hostname = urlsplit(location).hostname
            if hostname:
                domains.append(_domain(hostname))
        allowed = self.config.allowed_domains
        return tuple(
            dict.fromkeys(
                domain for domain in domains
                if not self._matches_any(domain, self.config.blocked_domains)
                and (not allowed or self._matches_any(domain, allowed))
            )
        )

    def _record(
        self, result: Mapping[str, Any], priorities: Sequence[str]
    ) -> WebResultRecord | None:
        try:
            url = str(result["url"]).strip()
            title = str(result["title"]).strip()
            snippet = str(result["snippet"]).strip()
            hostname = urlsplit(url).hostname
        except (KeyError, TypeError):
            return None
        if not url or not title or not snippet or not hostname:
            return None
        domain = _domain(hostname)
        if self._matches_any(domain, self.config.blocked_domains):
            return None
        if self.config.allowed_domains and not self._matches_any(domain, self.config.allowed_domains):
            return None
        return WebResultRecord(
            url, title, snippet, domain, self._matches_any(domain, priorities)
        )

    def _unique_records(
        self, results: Sequence[Mapping[str, Any]], priorities: Sequence[str]
    ) -> dict[str, WebResultRecord]:
        unique: dict[str, WebResultRecord] = {}
        for result in results:
            record = self._record(result, priorities)
            if record is not None:
                unique.setdefault(self._canonical_url(record.url), record)
        return unique

    @staticmethod
    def _scoped_query(query: str, domains: Sequence[str]) -> str:
        if not domains:
            return query.strip()
        scope = " OR ".join(f"site:{domain}" for domain in domains)
        return f"{query.strip()} ({scope})"

    @staticmethod
    def _matches_any(domain: str, configured: Sequence[str]) -> bool:
        return any(domain == item or domain.endswith(f".{item}") for item in configured)

    @staticmethod
    def _canonical_url(url: str) -> str:
        parsed = urlsplit(url)
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, ""))

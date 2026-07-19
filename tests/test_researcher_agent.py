"""Tests del pipeline de Researcher con proveedores totalmente simulados."""

import json
from collections.abc import Sequence
from typing import Any

from agents.researcher import (
    EvidenceFragment,
    EvidenceSufficiencyEvaluator,
    KnowledgeRetriever,
    ProjectMemoryProvider,
    ResearcherAgent,
    SufficiencyAssessment,
    WebSearchProvider,
)
from core.models import LLMResponse, LLMUsage, Message
from core.task_state import ErrorRecord, SourceReference, SubagentResult, TaskState


class FakeLLM:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.tools: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.messages = list(messages)
        self.tools = list(tools)
        payload = {
            "summary": "La evidencia recuperada describe la solución técnica.",
            "findings": ["La memoria y la documentación coinciden."],
            "recommendations": ["Aplicar la solución respetando la configuración."],
            "sources": [],
            "files_relevant": ["src/service.py"],
            "blockers": [],
            "confidence": 0.1,
        }
        text = json.dumps(payload)
        return LLMResponse(
            assistant_message=Message("assistant", text),
            text=text,
            tool_calls=[],
            model="fake-researcher",
            usage=LLMUsage(1, 1, 2),
            latency_ms=1.0,
        )


class FakeMemory(ProjectMemoryProvider):
    def __init__(self, events: list[str], fragments: Sequence[EvidenceFragment]) -> None:
        self.events = events
        self.fragments = tuple(fragments)
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        self.events.append("memory")
        self.queries.append(query)
        return self.fragments


class FakeRAG(KnowledgeRetriever):
    def __init__(self, events: list[str], fragments: Sequence[EvidenceFragment]) -> None:
        self.events = events
        self.fragments = tuple(fragments)
        self.queries: list[str] = []

    def retrieve(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        self.events.append("rag")
        self.queries.append(query)
        return self.fragments


class TracedRAG(FakeRAG):
    def __init__(self, events: list[str], fragments: Sequence[EvidenceFragment]) -> None:
        super().__init__(events, fragments)
        self.filters: dict[str, Sequence[str] | str] = {}

    def retrieve_filtered(self, query: str, *, filters=None, limit: int = 5):
        self.filters = dict(filters or {})
        return self.retrieve(query, limit=limit)

    def retrieval_audit(self):
        return {
            "query": self.queries[-1],
            "filters": self.filters,
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "score": 0.91,
                    "metadata": {
                        "document_id": "doc-1",
                        "path_or_url": "https://official.test/api",
                        "source_type": "official_documentation",
                        "tags": ["official"],
                    },
                }
            ],
            "used_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "score": 0.91,
                    "metadata": {
                        "document_id": "doc-1",
                        "path_or_url": "https://official.test/api",
                        "source_type": "official_documentation",
                        "tags": ["official"],
                    },
                }
            ],
            "scores": {"chunk-1": 0.91},
            "documents": ["doc-1"],
            "conclusions": ["Evidencia suficiente."],
            "sufficiency": {"sufficient": True, "confidence": 0.91, "reasons": []},
        }


class FakeWeb(WebSearchProvider):
    def __init__(self, events: list[str], fragments: Sequence[EvidenceFragment]) -> None:
        self.events = events
        self.fragments = tuple(fragments)
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        self.events.append("web")
        self.queries.append(query)
        return self.fragments


class ContextualWeb(FakeWeb):
    def __init__(self, events: list[str], fragments: Sequence[EvidenceFragment]) -> None:
        super().__init__(events, fragments)
        self.technologies: tuple[str, ...] = ()
        self.rag_metadata: tuple[dict[str, Any], ...] = ()

    def search_context(
        self, query: str, *, limit: int = 5, technologies=(), rag_metadata=()
    ):
        self.technologies = tuple(technologies)
        self.rag_metadata = tuple(dict(item) for item in rag_metadata)
        return self.search(query, limit=limit)

    def search_audit(self):
        fragment = self.fragments[0]
        return {
            "query": self.queries[-1],
            "executed_queries": [self.queries[-1]],
            "found": [
                {
                    "url": fragment.reference,
                    "title": "Official result",
                    "snippet": fragment.content,
                    "domain": "official.test",
                    "priority": True,
                }
            ],
            "used": [
                {
                    "url": fragment.reference,
                    "title": "Official result",
                    "snippet": fragment.content,
                    "domain": "official.test",
                    "priority": True,
                }
            ],
            "conclusions": ["Se utilizó documentación prioritaria."],
        }


class ScriptedEvaluator(EvidenceSufficiencyEvaluator):
    def __init__(self, events: list[str], sufficient_without_web: bool) -> None:
        self.events = events
        self.sufficient_without_web = sufficient_without_web

    def evaluate(
        self, query: str, fragments: Sequence[EvidenceFragment]
    ) -> SufficiencyAssessment:
        self.events.append("evaluate")
        has_web = any(fragment.origin == "web" for fragment in fragments)
        sufficient = self.sufficient_without_web or has_web
        return SufficiencyAssessment(
            sufficient=sufficient,
            confidence=0.9 if sufficient else 0.35,
            missing_information=() if sufficient else ("Falta documentación externa.",),
        )


def fragment(origin: str, reference: str, content: str) -> EvidenceFragment:
    return EvidenceFragment(origin, reference, content)  # type: ignore[arg-type]


def build_researcher(
    events: list[str], *, sufficient: bool
) -> tuple[ResearcherAgent, FakeMemory, FakeRAG, FakeWeb, FakeLLM]:
    memory = FakeMemory(
        events,
        [fragment("project_memory", "decision-12", "Se eligió una API estable.")],
    )
    rag = FakeRAG(
        events,
        [fragment("rag", "docs/api.md#errors", "La API devuelve errores tipados.")],
    )
    web = FakeWeb(
        events,
        [fragment("web", "https://example.test/reference", "Referencia oficial externa.")],
    )
    llm = FakeLLM()
    researcher = ResearcherAgent(
        llm_client=llm,
        project_memory=memory,
        knowledge_retriever=rag,
        web_search=web,
        sufficiency_evaluator=ScriptedEvaluator(events, sufficient),
    )
    return researcher, memory, rag, web, llm


def populated_state() -> TaskState:
    state = TaskState.create("Corregir manejo de errores", task_id="research-task")
    state.add_repository_finding("language=Python; evidencia: pyproject.toml.")
    state.add_repository_finding("framework=FastAPI; evidencia: pyproject.toml.")
    state.add_repository_finding("dependency=httpx; evidencia: pyproject.toml.")
    state.add_source(
        SourceReference("repository", "pyproject.toml", "Configuración y dependencias")
    )
    state.add_subagent_result(
        SubagentResult(
            "explorer",
            "Explorar",
            "completed",
            summary="Exploración lista",
            files_relevant=("src/service.py", "tests/test_service.py"),
        )
    )
    state.record_error(
        ErrorRecord("AssertionError en test_error", "testing", "pytest", True)
    )
    return state


def test_memory_is_queried_before_rag_and_web_is_skipped_when_sufficient() -> None:
    events: list[str] = []
    researcher, _, _, web, _ = build_researcher(events, sufficient=True)

    result = researcher.run("Investigar contrato de errores", populated_state())

    assert events == ["memory", "rag", "evaluate"]
    assert result.web_needed is False
    assert result.web_used is False
    assert web.queries == []
    assert [query.provider for query in result.queries_performed] == [
        "project_memory", "rag"
    ]


def test_rag_precedes_web_and_web_is_used_only_as_fallback() -> None:
    events: list[str] = []
    researcher, _, _, web, _ = build_researcher(events, sufficient=False)

    result = researcher.run("Investigar contrato de errores", populated_state())

    assert events == ["memory", "rag", "evaluate", "web", "evaluate"]
    assert result.web_needed is True
    assert result.web_used is True
    assert len(web.queries) == 1
    assert [query.provider for query in result.queries_performed] == [
        "project_memory", "rag", "web"
    ]


def test_web_fallback_receives_detected_technologies_and_official_rag_metadata() -> None:
    events: list[str] = []
    memory = FakeMemory(events, [])
    rag = TracedRAG(events, [])
    web = ContextualWeb(
        events,
        [fragment("web", "https://official.test/reference", "Official fragment")],
    )
    state = populated_state()
    researcher = ResearcherAgent(
        llm_client=FakeLLM(),
        project_memory=memory,
        knowledge_retriever=rag,
        web_search=web,
        sufficiency_evaluator=ScriptedEvaluator(events, False),
    )

    researcher.run("Investigar fallback", state)

    assert web.technologies == ("Python", "FastAPI")
    assert web.rag_metadata[0]["path_or_url"] == "https://official.test/api"
    trace = next(item for item in state.observations if item.startswith("WEB trace: "))
    payload = json.loads(trace.removeprefix("WEB trace: "))
    assert payload["found"][0]["title"] == "Official result"
    assert payload["used"][0]["snippet"] == "Official fragment"


def test_preserves_source_traceability_in_result_and_shared_state() -> None:
    events: list[str] = []
    researcher, _, _, _, _ = build_researcher(events, sufficient=False)
    state = populated_state()

    result = researcher.run("Reunir evidencia", state)

    assert {source.origin for source in result.sources_recovered} == {
        "repository", "project_memory", "rag", "web"
    }
    assert {fragment.origin for fragment in result.fragments_used} == {
        "repository", "project_memory", "rag", "web"
    }
    assert {source.origin for source in state.sources} >= {
        "repository", "project_memory", "rag", "web", "inference"
    }
    assert state.subagent_results[-1] == result.subagent_result
    assert result.technical_summary.startswith("La evidencia recuperada")
    assert result.confidence == 0.9


def test_query_adapts_to_explorer_technologies_files_errors_and_configuration() -> None:
    events: list[str] = []
    researcher, memory, rag, _, llm = build_researcher(events, sufficient=True)
    state = populated_state()

    researcher.run("Buscar guía de compatibilidad", state)

    query = memory.queries[0]
    assert rag.queries == [query]
    assert "Corregir manejo de errores" in query
    assert "Python" in query
    assert "FastAPI" in query
    assert "httpx" in query
    assert "src/service.py" in query
    assert "tests/test_service.py" in query
    assert "AssertionError en test_error" in query
    assert "pyproject.toml" in query
    sent_context = json.loads(llm.messages[1].content)["context"]
    assert any("project_memory" in fact for fact in sent_context["facts"])
    assert llm.tools == []


def test_passes_explorer_filters_and_registers_rag_trace_in_task_state() -> None:
    events: list[str] = []
    memory = FakeMemory(events, [fragment("project_memory", "decision", "Memory")])
    rag = TracedRAG(events, [fragment("rag", "rag://doc-1/0", "RAG evidence")])
    state = populated_state()
    researcher = ResearcherAgent(
        llm_client=FakeLLM(),
        project_memory=memory,
        knowledge_retriever=rag,
        sufficiency_evaluator=ScriptedEvaluator(events, True),
    )

    researcher.run("Investigar", state)

    assert rag.filters == {"language": ("Python",), "framework": ("FastAPI",)}
    trace = next(item for item in state.observations if item.startswith("RAG trace: "))
    payload = json.loads(trace.removeprefix("RAG trace: "))
    assert payload["retrieved"][0]["chunk_id"] == "chunk-1"
    assert payload["used"][0]["score"] == 0.91
    assert payload["documents"] == ["doc-1"]
    assert payload["conclusions"] == ["Evidencia suficiente."]


def test_reports_missing_information_when_web_is_needed_but_unavailable() -> None:
    events: list[str] = []
    memory = FakeMemory(events, [])
    rag = FakeRAG(events, [])
    researcher = ResearcherAgent(
        llm_client=FakeLLM(),
        project_memory=memory,
        knowledge_retriever=rag,
        web_search=None,
        sufficiency_evaluator=ScriptedEvaluator(events, False),
    )

    result = researcher.run("Investigar un dato ausente", populated_state())

    assert events == ["memory", "rag", "evaluate"]
    assert result.web_needed is True
    assert result.web_used is False
    assert "Falta documentación externa." in result.missing_information
    assert "La búsqueda web no está configurada." in result.missing_information
    assert result.subagent_result.status == "blocked"

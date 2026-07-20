"""Compactación del plan y consultas específicas del caso de análisis."""

import json
from collections.abc import Sequence

from agents.orchestrator import LLMPlanGenerator
from agents.researcher import ResearcherAgent
from core.models import LLMResponse, LLMUsage, Message
from core.profiles import ProjectProfile
from core.research_ports import KnowledgeRetriever, ProjectMemoryProvider
from core.task_state import SourceReference, SubagentResult, TaskState
from rag.models import SourceConfig


class CapturingLLM:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def complete(self, messages: Sequence[Message], tools=()) -> LLMResponse:
        self.messages = list(messages)
        return LLMResponse(
            Message("assistant", "1. Inspeccionar evidencia de sólo lectura."),
            "1. Inspeccionar evidencia de sólo lectura.", [], "fake",
            LLMUsage(1, 1, 2), 1.0,
        )


class EmptyMemory(ProjectMemoryProvider):
    def search(self, query: str, *, limit: int = 5):
        return ()


class EmptyRag(KnowledgeRetriever):
    def retrieve(self, query: str, *, limit: int = 5):
        return ()


def large_state() -> TaskState:
    request = "Analizá PrintScript y generá arquitectura y módulos sin modificar archivos."
    state = TaskState.create(request, task_id="large")
    state.add_observation("Tarea clasificada como analysis: sólo lectura.")
    state.add_repository_finding(
        "modules=cli, lexer, parser, interpreter; evidencia: settings.gradle.kts."
    )
    state.add_repository_finding(
        "dependency=picocli; evidencia: cli/build.gradle."
    )
    state.add_repository_finding("código fuente: " + ", ".join(
        f"module/src/main/File{index}.kt" for index in range(500)
    ))
    state.add_subagent_result(
        SubagentResult(
            "explorer", "explorar", "completed", summary="Arquitectura confirmada.",
            files_relevant=tuple(f"module/File{index}.kt" for index in range(100)),
        )
    )
    for index in range(100):
        state.add_source(SourceReference("repository", f"module/File{index}.kt"))
    return state


def test_plan_context_is_bounded_and_request_appears_once() -> None:
    llm = CapturingLLM()
    generator = LLMPlanGenerator(llm)
    state = large_state()

    generator.generate(state)
    payload_text = llm.messages[1].content
    payload = json.loads(payload_text)

    assert len(payload_text) <= generator.MAX_CONTEXT_CHARACTERS
    assert payload_text.count(state.original_request) == 1
    assert len(payload["relevant_files"]) <= generator.MAX_RELEVANT_FILES
    assert len(payload["sources"]) <= generator.MAX_SOURCES
    assert "File499.kt" not in payload_text
    assert payload["confirmed_modules"] == ["cli", "lexer", "parser", "interpreter"]
    assert payload["main_dependencies"] == ["picocli"]
    assert payload["security_constraints"]


def test_large_repository_still_generates_plan_with_fake() -> None:
    result = LLMPlanGenerator(CapturingLLM()).generate(large_state())
    assert result.startswith("1. Inspeccionar")


def test_printscript_analysis_uses_short_specific_queries_and_source_filter() -> None:
    profile = ProjectProfile(
        name="PrintScript",
        rag_sources=(
            SourceConfig(
                "printscript-language-spec", "local", "documentation",
                "docs/printscript-language-spec.md",
            ),
        ),
    )
    researcher = ResearcherAgent(
        llm_client=CapturingLLM(), project_memory=EmptyMemory(),
        knowledge_retriever=EmptyRag(), profile=profile,
    )
    state = TaskState.create("Analizar arquitectura y módulos de PrintScript")
    state.add_repository_finding("language=Kotlin; evidencia: build.gradle.kts.")
    state.add_repository_finding("build_system=Gradle; evidencia: settings.gradle.kts.")

    queries = researcher.build_rag_queries(state.original_request, state)

    assert "PrintScript language specification" in queries
    assert "PrintScript lexer parser interpreter flow" in queries
    assert "Kotlin Gradle multi-module architecture" in queries
    assert all(len(query) < 100 for query in queries)
    assert researcher._build_rag_filters(
        state, None, query="PrintScript language specification"
    ) == {"source": ("printscript-language-spec",)}

"""Subagente genérico de investigación con memoria, RAG y fallback web."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from agents.base import AgentContext, AgentExecutionError, AgentInput, BaseAgent
from core.llm_client import LLMClient
from core.research_ports import (
    EvidenceFragment,
    KnowledgeRetriever,
    ProjectMemoryProvider,
    WebSearchProvider,
)
from core.settings import AgentSettings
from core.task_state import SourceOrigin, SourceReference, SubagentResult, TaskState
from core.profiles import ProjectProfile
from tools.registry import ToolRegistry

__all__ = [
    "EvidenceFragment",
    "KnowledgeRetriever",
    "ProjectMemoryProvider",
    "WebSearchProvider",
    "EvidenceSufficiencyEvaluator",
    "ThresholdSufficiencyEvaluator",
    "ProviderName",
    "ResearchQuery",
    "SufficiencyAssessment",
    "ResearcherResult",
    "ResearcherAgent",
]


RESEARCHER_SYSTEM_PROMPT = """Sos Researcher, un investigador técnico genérico.
Sintetizá exclusivamente la evidencia recuperada y diferenciá repositorio, memoria
del proyecto, RAG, web e inferencias. Indicá contradicciones, incertidumbres y datos
faltantes. No inventes APIs, versiones ni comportamientos. No escribas archivos, no
modifiques código y no solicites tools: los proveedores ya fueron consultados por el
pipeline en el orden memoria, RAG y web sólo como fallback."""


ProviderName = Literal["project_memory", "rag", "web"]


@dataclass(frozen=True)
class ResearchQuery:
    provider: ProviderName
    text: str


@dataclass(frozen=True)
class SufficiencyAssessment:
    """Decisión explícita sobre cobertura, confianza y huecos de evidencia."""

    sufficient: bool
    confidence: float
    missing_information: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sufficient, bool):
            raise ValueError("sufficient debe ser booleano.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence debe estar entre 0 y 1.")
        object.__setattr__(self, "confidence", float(self.confidence))
        if not all(isinstance(item, str) and item.strip() for item in self.missing_information):
            raise ValueError("missing_information debe contener textos no vacíos.")
        object.__setattr__(
            self,
            "missing_information",
            tuple(item.strip() for item in self.missing_information),
        )


class EvidenceSufficiencyEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, query: str, fragments: Sequence[EvidenceFragment]
    ) -> SufficiencyAssessment:
        """Evalúa si la evidencia disponible permite responder responsablemente."""


class ThresholdSufficiencyEvaluator(EvidenceSufficiencyEvaluator):
    """Evaluador neutral basado en cantidad, diversidad de origen y relevancia."""

    def evaluate(
        self, query: str, fragments: Sequence[EvidenceFragment]
    ) -> SufficiencyAssessment:
        useful = [fragment for fragment in fragments if fragment.relevance >= 0.5]
        origins = {fragment.origin for fragment in useful}
        confidence = min(1.0, len(useful) * 0.2 + len(origins) * 0.15)
        sufficient = len(useful) >= 2 and len(origins) >= 2
        missing = () if sufficient else ("Falta evidencia suficiente y diversa.",)
        return SufficiencyAssessment(sufficient, confidence, missing)


@dataclass(frozen=True)
class ResearcherResult:
    """Salida completa del pipeline y su representación común persistida."""

    queries_performed: tuple[ResearchQuery, ...]
    sources_recovered: tuple[SourceReference, ...]
    fragments_used: tuple[EvidenceFragment, ...]
    technical_summary: str
    confidence: float
    missing_information: tuple[str, ...]
    web_needed: bool
    web_used: bool
    subagent_result: SubagentResult


class ResearcherAgent(BaseAgent):
    """Investiga en orden fijo y usa web sólo tras evidencia insuficiente."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        project_memory: ProjectMemoryProvider,
        knowledge_retriever: KnowledgeRetriever,
        web_search: WebSearchProvider | None = None,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator | None = None,
        name: str = "researcher",
        profile: ProjectProfile | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role="Technical Researcher",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            allowed_tools=(),
            llm_client=llm_client,
        )
        self.project_memory = project_memory
        self.knowledge_retriever = knowledge_retriever
        self.web_search = web_search
        self.sufficiency_evaluator = (
            sufficiency_evaluator or ThresholdSufficiencyEvaluator()
        )
        self.profile = profile or ProjectProfile()

    @classmethod
    def from_settings(
        cls,
        *,
        llm_client: LLMClient,
        project_memory: ProjectMemoryProvider,
        knowledge_retriever: KnowledgeRetriever,
        settings: AgentSettings,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator | None = None,
        name: str = "researcher",
        profile: ProjectProfile | None = None,
    ) -> "ResearcherAgent":
        """Compone Researcher con el fallback web habilitado por configuración."""
        provider: WebSearchProvider | None = None
        if settings.web_search_enabled and os.getenv("TAVILY_API_KEY"):
            from agents.web_research import ConfiguredWebSearchProvider, WebSearchConfig

            web_config = dict(settings.web_search_config or {})
            web_config.setdefault(
                "priority_domains", list((profile or ProjectProfile()).priority_web_domains)
            )
            provider = ConfiguredWebSearchProvider(WebSearchConfig.from_dict(web_config))
        return cls(
            llm_client=llm_client,
            project_memory=project_memory,
            knowledge_retriever=knowledge_retriever,
            web_search=provider,
            sufficiency_evaluator=sufficiency_evaluator,
            name=name,
            profile=profile,
        )

    def specialization_prompt(self) -> str:
        return (
            "La entrada contiene consultas y fragmentos ya recuperados. Devolvé un "
            "resumen técnico; toda afirmación debe poder vincularse a esos fragmentos."
        )

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
    ) -> ResearcherResult:
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state debe ser una instancia de TaskState.")
        queries: list[ResearchQuery] = []
        rag_audit: Mapping[str, Any] | None = None
        web_audit: Mapping[str, Any] | None = None
        with self._error_guard(task_state, action="investigar"):
            query = self.build_research_query(instruction, task_state, context)

            queries.append(ResearchQuery("project_memory", query))
            memory = self._validate_fragments(
                self.project_memory.search(query, limit=5), "project_memory"
            )

            queries.append(ResearchQuery("rag", query))
            rag = self._validate_fragments(
                self.knowledge_retriever.retrieve_filtered(
                    query,
                    filters=self._build_rag_filters(task_state, context),
                    limit=5,
                ),
                "rag",
            )
            rag_audit = self.knowledge_retriever.retrieval_audit()

            repository = self._repository_fragments(task_state, context)
            fragments = [*repository, *memory, *rag]
            initial_assessment = self.sufficiency_evaluator.evaluate(query, fragments)
            web_needed = not initial_assessment.sufficient
            web_used = False

            if web_needed and self.web_search is not None:
                queries.append(ResearchQuery("web", query))
                web = self._validate_fragments(
                    self.web_search.search_context(
                        query,
                        limit=5,
                        technologies=self._detected_technologies(task_state),
                        rag_metadata=self._rag_metadata(rag_audit),
                    ),
                    "web",
                )
                web_audit = self.web_search.search_audit()
                fragments.extend(web)
                web_used = True
                final_assessment = self.sufficiency_evaluator.evaluate(query, fragments)
            else:
                final_assessment = initial_assessment

            missing = final_assessment.missing_information
            if web_needed and self.web_search is None:
                missing = (*missing, "La búsqueda web no está configurada.")
                task_state.add_observation(
                    "WEB trace: "
                    + json.dumps(
                        {
                            "query": query,
                            "executed_queries": [],
                            "found": [],
                            "used": [],
                            "conclusions": ["La búsqueda web no está disponible."],
                        },
                        ensure_ascii=False,
                    )
                )

            subagent_result = self._synthesize(
                instruction,
                task_state,
                context,
                queries,
                fragments,
                final_assessment,
                missing,
            )

        sources = tuple(fragment.to_source() for fragment in fragments)
        for source in sources:
            task_state.add_source(source)
        task_state.add_source(
            SourceReference(
                "inference",
                f"researcher:{task_state.task_id}",
                subagent_result.summary,
            )
        )
        task_state.add_subagent_result(subagent_result)
        for item in missing:
            task_state.add_warning(item)
        task_state.add_observation(
            f"Researcher usó web: {'sí' if web_used else 'no'}; "
            f"confianza: {final_assessment.confidence:.2f}."
        )
        if rag_audit:
            task_state.add_observation(
                "RAG trace: "
                + json.dumps(self._compact_rag_audit(rag_audit), ensure_ascii=False)
            )
        if web_audit:
            task_state.add_observation(
                "WEB trace: "
                + json.dumps(self._compact_web_audit(web_audit), ensure_ascii=False)
            )
        return ResearcherResult(
            queries_performed=tuple(queries),
            sources_recovered=sources,
            fragments_used=tuple(fragments),
            technical_summary=subagent_result.summary or subagent_result.result or "",
            confidence=final_assessment.confidence,
            missing_information=tuple(missing),
            web_needed=web_needed,
            web_used=web_used,
            subagent_result=subagent_result,
        )

    def build_research_query(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
    ) -> str:
        """Combina sólo señales disponibles, con etiquetas neutrales y trazables."""
        relevant_files = tuple(
            dict.fromkeys(
                file
                for result in task_state.subagent_results
                for file in result.files_relevant
            )
        )
        repository_sources = tuple(
            source.reference for source in task_state.sources if source.origin == "repository"
        )
        errors = tuple(error.message for error in task_state.errors)
        parts = [
            f"Pedido original: {task_state.original_request}",
            f"Instrucción de investigación: {instruction}",
        ]
        confirmed = self._detected_technologies(task_state)
        if confirmed:
            parts.append("Tecnologías confirmadas por Explorer: " + ", ".join(confirmed))
        if self.profile.expected_technologies:
            parts.append(
                "Hipótesis secundarias del perfil: "
                + ", ".join(self.profile.expected_technologies)
            )
        if self.profile.search_tags:
            parts.append("Tags de búsqueda del perfil: " + ", ".join(self.profile.search_tags))
            task_state.add_observation(
                "Tags usados en consultas: " + ", ".join(self.profile.search_tags)
            )
        if self.profile.rag_sources:
            references = tuple(source.location for source in self.profile.rag_sources)
            parts.append("Fuentes RAG efectivas configuradas: " + ", ".join(references))
            task_state.add_observation(
                "Fuentes RAG efectivas consideradas: " + ", ".join(references)
            )
        if task_state.repository_findings:
            parts.append(
                "Tecnologías, dependencias y configuración detectadas por Explorer: "
                + " | ".join(task_state.repository_findings)
            )
        if relevant_files:
            parts.append("Archivos relevantes: " + ", ".join(relevant_files))
        if repository_sources:
            parts.append("Evidencia/configuración del repositorio: " + ", ".join(repository_sources))
        if errors:
            parts.append("Errores observados: " + " | ".join(errors))
        if context is not None and context.facts:
            parts.append("Contexto seleccionado: " + " | ".join(context.facts))
        return "\n".join(parts)

    def _build_rag_filters(
        self, state: TaskState, context: AgentContext | None
    ) -> dict[str, tuple[str, ...]]:
        filters: dict[str, list[str]] = {}
        supported = {"technology", "language", "framework", "source_type", "module", "tags"}
        for finding in state.repository_findings:
            prefix, separator, remainder = finding.partition("=")
            key = prefix.strip().casefold()
            if separator and key in supported:
                value = remainder.split(";", 1)[0].strip()
                if value:
                    filters.setdefault(key, []).append(value)
        if context is not None:
            for fact in context.facts:
                if not fact.startswith("rag_filter:"):
                    continue
                expression = fact.removeprefix("rag_filter:")
                key, separator, value = expression.partition("=")
                key = key.strip().casefold()
                if separator and key in supported and value.strip():
                    filters.setdefault(key, []).append(value.strip())
        if self.profile.search_tags:
            filters.setdefault("tags", []).extend(self.profile.search_tags)
        return {key: tuple(dict.fromkeys(values)) for key, values in filters.items()}

    @staticmethod
    def _compact_rag_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
        def summarize(items: object) -> list[dict[str, Any]]:
            if not isinstance(items, list):
                return []
            return [
                {
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "document_id": item.get("metadata", {}).get("document_id"),
                    "path_or_url": item.get("metadata", {}).get("path_or_url"),
                }
                for item in items
                if isinstance(item, dict)
            ]

        return {
            "query": audit.get("query"),
            "filters": audit.get("filters", {}),
            "retrieved": summarize(audit.get("retrieved_chunks")),
            "used": summarize(audit.get("used_chunks")),
            "scores": audit.get("scores", {}),
            "documents": audit.get("documents", []),
            "conclusions": audit.get("conclusions", []),
            "sufficiency": audit.get("sufficiency", {}),
        }

    @staticmethod
    def _detected_technologies(state: TaskState) -> tuple[str, ...]:
        categories = {
            "technology", "language", "framework", "test_framework",
            "tool", "build_system",
        }
        values = []
        for finding in state.repository_findings:
            prefix, separator, remainder = finding.partition("=")
            if separator and prefix.strip().casefold() in categories:
                value = remainder.split(";", 1)[0].strip()
                if value:
                    values.append(value)
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _rag_metadata(
        audit: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        if not audit or not isinstance(audit.get("used_chunks"), list):
            return ()
        return tuple(
            item["metadata"]
            for item in audit["used_chunks"]
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
        )

    @staticmethod
    def _compact_web_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
        def results(name: str) -> list[dict[str, Any]]:
            values = audit.get(name)
            if not isinstance(values, list):
                return []
            return [
                {
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "domain": item.get("domain"),
                    "priority": item.get("priority"),
                }
                for item in values
                if isinstance(item, dict)
            ]

        return {
            "query": audit.get("query"),
            "executed_queries": audit.get("executed_queries", []),
            "found": results("found"),
            "used": results("used"),
            "conclusions": audit.get("conclusions", []),
        }

    def _synthesize(
        self,
        instruction: str,
        state: TaskState,
        context: AgentContext | None,
        queries: Sequence[ResearchQuery],
        fragments: Sequence[EvidenceFragment],
        assessment: SufficiencyAssessment,
        missing: Sequence[str],
    ) -> SubagentResult:
        facts = (
            f"Consultas: {json.dumps([query.__dict__ for query in queries], ensure_ascii=False)}",
            f"Fragmentos: {json.dumps([item.to_dict() for item in fragments], ensure_ascii=False)}",
            f"Confianza evaluada: {assessment.confidence}",
            f"Información faltante: {json.dumps(list(missing), ensure_ascii=False)}",
        )
        selected = context or AgentContext()
        agent_input = AgentInput(
            instruction=instruction,
            task_id=state.task_id,
            context=AgentContext(
                facts=(*selected.facts, *facts),
                sources=selected.sources,
                files=selected.files,
                constraints=(
                    *selected.constraints,
                    "No agregar afirmaciones sin respaldo en los fragmentos.",
                ),
            ),
        )
        def _reject_tool_calls(response):
            if response.tool_calls:
                raise AgentExecutionError("Researcher no puede solicitar tools.")

        base_result = self._complete_and_parse(agent_input, (), _reject_tool_calls)
        sources = tuple(fragment.to_source() for fragment in fragments)
        return SubagentResult(
            subagent_id=base_result.subagent_id,
            task=base_result.task,
            status="completed" if assessment.sufficient else "blocked",
            result=base_result.result,
            error=base_result.error,
            summary=base_result.summary,
            findings=base_result.findings,
            recommendations=base_result.recommendations,
            requested_tool_calls=(),
            sources=sources,
            files_relevant=base_result.files_relevant,
            blockers=tuple(missing),
            confidence=assessment.confidence,
        )

    @staticmethod
    def _validate_fragments(
        fragments: Sequence[EvidenceFragment], expected_origin: SourceOrigin
    ) -> tuple[EvidenceFragment, ...]:
        result = tuple(fragments)
        if not all(isinstance(item, EvidenceFragment) for item in result):
            raise AgentExecutionError("Un proveedor devolvió fragmentos inválidos.")
        if any(item.origin != expected_origin for item in result):
            raise AgentExecutionError(
                f"El proveedor {expected_origin} devolvió un origen incorrecto."
            )
        return result

    @staticmethod
    def _repository_fragments(
        state: TaskState, context: AgentContext | None
    ) -> tuple[EvidenceFragment, ...]:
        sources = [source for source in state.sources if source.origin == "repository"]
        if context is not None:
            sources.extend(source for source in context.sources if source.origin == "repository")
        return tuple(
            EvidenceFragment(
                "repository",
                source.reference,
                source.summary or f"Evidencia del repositorio: {source.reference}",
            )
            for source in sources
        )

"""Selección relevante, progresiva y acotada de contexto para subagentes."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from agents.base import AgentContext
from core.task_state import SourceReference, SubagentResult, TaskState


class ContextManager(ABC):
    """Contrato para seleccionar contexto sin exponer el estado compartido."""

    @abstractmethod
    def select(
        self,
        instruction: str,
        task_state: TaskState,
        requested: AgentContext | None = None,
    ) -> AgentContext:
        """Devuelve el subconjunto autorizado para una ejecución."""


@dataclass(frozen=True)
class _RankedText:
    score: int
    order: int
    text: str


class StateContextManager(ContextManager):
    """Construye contexto genérico sin copiar historial ni repositorio completos."""

    _TECHNOLOGY_KEYS = frozenset(
        {"technology", "language", "framework", "test_framework", "tool", "build_system"}
    )

    def __init__(
        self,
        *,
        max_context_chars: int = 8_000,
        max_item_chars: int = 600,
        max_results: int = 5,
        max_sources: int = 6,
        max_files: int = 12,
    ) -> None:
        if max_context_chars < 256:
            raise ValueError("max_context_chars debe ser al menos 256.")
        if max_item_chars < 40:
            raise ValueError("max_item_chars debe ser al menos 40.")
        if min(max_results, max_sources, max_files) < 1:
            raise ValueError("Los límites de colecciones deben ser positivos.")
        self.max_context_chars = max_context_chars
        self.max_item_chars = max_item_chars
        self.max_results = max_results
        self.max_sources = max_sources
        self.max_files = max_files

    def select(
        self,
        instruction: str,
        task_state: TaskState,
        requested: AgentContext | None = None,
    ) -> AgentContext:
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state debe ser TaskState.")
        query = " ".join(
            filter(
                None,
                (
                    instruction,
                    task_state.original_request,
                    task_state.approved_plan or task_state.proposed_plan or "",
                    task_state.current_phase,
                ),
            )
        )
        tokens = self._tokens(query)
        selected = requested or AgentContext()

        facts = [
            f"Pedido: {self._truncate(task_state.original_request)}",
            f"Plan: {self._truncate(task_state.approved_plan or task_state.proposed_plan or 'Sin plan aprobado.')}",
            f"Fase: {task_state.current_phase}",
        ]
        facts.extend(self._repository_context(task_state, tokens))
        facts.extend(self._result_context(task_state.subagent_results, tokens))
        facts.extend(
            f"HECHO — contexto solicitado: {self._truncate(value)}"
            for value in self._unique(selected.facts)
        )

        sources = self._sources(task_state, selected, tokens)
        for source in sources:
            if source.origin == "inference":
                facts.append(
                    "INFERENCIA — "
                    + self._truncate(source.summary or source.reference)
                )

        files = self._files(task_state.subagent_results, selected.files, tokens)
        constraints = self._constraints(task_state, selected, tokens)
        context = AgentContext(
            facts=tuple(self._unique(facts)),
            sources=sources,
            files=files,
            constraints=constraints,
        )
        return self._fit_budget(context)

    def _repository_context(self, state: TaskState, tokens: set[str]) -> list[str]:
        architecture: list[_RankedText] = []
        technologies: list[_RankedText] = []
        for order, finding in enumerate(state.repository_findings):
            key, separator, raw = finding.partition("=")
            if not separator:
                continue
            value = raw.split(";", 1)[0].strip()
            normalized = key.strip().casefold()
            ranked = _RankedText(self._score(finding, tokens), order, value)
            if normalized == "architecture":
                architecture.append(ranked)
            elif normalized in self._TECHNOLOGY_KEYS:
                technologies.append(ranked)
        facts: list[str] = []
        if architecture:
            item = self._ranked(architecture)[0]
            facts.append(f"Arquitectura: {self._truncate(item.text)}")
        for item in self._ranked(technologies)[:6]:
            facts.append(f"Tecnología: {self._truncate(item.text)}")
        return facts

    def _result_context(
        self, results: tuple[SubagentResult, ...], tokens: set[str]
    ) -> list[str]:
        ranked: list[tuple[int, int, SubagentResult]] = []
        for order, result in enumerate(results):
            text = " ".join(
                filter(None, (result.task, result.summary, *result.findings, *result.recommendations))
            )
            score = self._score(text, tokens)
            if score:
                ranked.append((score, order, result))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        detailed = sorted(ranked[: self.max_results], key=lambda item: item[1])
        older = ranked[self.max_results :]
        facts: list[str] = []
        if older:
            statuses = ", ".join(
                f"{item[2].subagent_id}:{item[2].status}" for item in sorted(older, key=lambda item: item[1])
            )
            facts.append(f"HECHO — Resumen progresivo de resultados anteriores: {self._truncate(statuses)}")
        for _, _, result in detailed:
            summary = result.summary or result.result or result.status
            facts.append(
                f"HECHO — Resultado {result.subagent_id} ({result.status}): "
                f"{self._truncate(summary)}"
            )
        return facts

    def _sources(
        self, state: TaskState, requested: AgentContext, tokens: set[str]
    ) -> tuple[SourceReference, ...]:
        candidates = self._unique_objects((*requested.sources, *state.sources))
        ranked = sorted(
            (
                (self._score(f"{source.reference} {source.summary or ''}", tokens), index, source)
                for index, source in enumerate(candidates)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return tuple(item[2] for item in ranked if item[0] > 0)[: self.max_sources]

    def _files(
        self,
        results: tuple[SubagentResult, ...],
        requested: tuple[str, ...],
        tokens: set[str],
    ) -> tuple[str, ...]:
        discovered = self._unique(
            path for result in results for path in result.files_relevant
        )
        allowed = set(discovered)
        if requested:
            candidates = self._unique(path for path in requested if path in allowed)
        else:
            candidates = discovered
        result_context: dict[str, str] = {}
        for result in results:
            context = " ".join(filter(None, (result.task, result.summary, *result.findings)))
            for path in result.files_relevant:
                result_context[path] = f"{result_context.get(path, '')} {context}"
        direct_scores = {
            path: self._score(path.replace("/", " ").replace("_", " "), tokens)
            for path in candidates
        }
        use_direct = any(direct_scores.values())
        ranked = sorted(
            (
                (
                    direct_scores[path]
                    if use_direct
                    else self._score(result_context.get(path, ""), tokens),
                    index,
                    path,
                )
                for index, path in enumerate(candidates)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        positive = [item[2] for item in ranked if item[0] > 0]
        fallback = [item[2] for item in sorted(ranked, key=lambda item: item[1])]
        return tuple((positive or fallback)[: self.max_files])

    def _constraints(
        self, state: TaskState, requested: AgentContext, tokens: set[str]
    ) -> tuple[str, ...]:
        decisions: list[str] = []
        for finding in state.repository_findings:
            key, separator, value = finding.partition("=")
            if separator and key.strip().casefold() == "decision":
                decisions.append(f"Decisión persistente: {self._truncate(value)}")
        for source in state.sources:
            if source.origin == "project_memory" and "decision" in source.reference.casefold():
                decisions.append(
                    f"Decisión persistente: {self._truncate(source.summary or source.reference)}"
                )
        errors = sorted(
            state.errors,
            key=lambda error: (
                error.phase != state.current_phase,
                -self._score(error.message, tokens),
            ),
        )
        active = [
            f"Error activo: {self._truncate(error.message)}"
            for error in errors
            if error.phase == state.current_phase or not error.recoverable
        ]
        explicit = [
            f"Restricción: {self._truncate(value)}"
            for value in self._unique(requested.constraints)
        ]
        return tuple(self._unique((*decisions, *active, *explicit)))

    def _fit_budget(self, context: AgentContext) -> AgentContext:
        facts = list(context.facts)
        sources = list(context.sources)
        files = list(context.files)
        constraints = list(context.constraints)

        def build() -> AgentContext:
            return AgentContext(tuple(facts), tuple(sources), tuple(files), tuple(constraints))

        while len(str(build().to_dict())) > self.max_context_chars:
            optional_fact = next(
                (index for index in range(len(facts) - 1, 2, -1) if "Decisión persistente:" not in facts[index]),
                None,
            )
            if optional_fact is not None:
                facts.pop(optional_fact)
            elif sources:
                sources.pop()
            elif files:
                files.pop()
            elif any(not item.startswith("Decisión persistente:") for item in constraints):
                index = next(
                    index for index in range(len(constraints) - 1, -1, -1)
                    if not constraints[index].startswith("Decisión persistente:")
                )
                constraints.pop(index)
            else:
                break
        return build()

    def _truncate(self, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= self.max_item_chars:
            return normalized
        marker = " …[truncado]"
        limit = self.max_item_chars - len(marker)
        prefix = normalized[:limit]
        boundary = prefix.rfind(" ")
        if boundary >= max(1, limit // 2):
            prefix = prefix[:boundary]
        return prefix.rstrip(" ,;:") + marker

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"\w{2,}", text.casefold())}

    @classmethod
    def _score(cls, text: str, tokens: set[str]) -> int:
        return len(cls._tokens(text).intersection(tokens))

    @staticmethod
    def _ranked(items: list[_RankedText]) -> list[_RankedText]:
        return sorted(items, key=lambda item: (-item.score, item.order))

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @staticmethod
    def _unique_objects(values: Iterable[SourceReference]) -> list[SourceReference]:
        return list(dict.fromkeys(values))

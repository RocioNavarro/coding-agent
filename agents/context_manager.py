"""Selección explícita y acotada de contexto para subagentes mutadores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agents.base import AgentContext
from core.task_state import TaskState


class ContextManager(ABC):
    """Contrato para seleccionar contexto sin exponer todo el estado compartido."""

    @abstractmethod
    def select(
        self,
        instruction: str,
        task_state: TaskState,
        requested: AgentContext | None = None,
    ) -> AgentContext:
        """Devuelve el subconjunto autorizado para una ejecución."""


class StateContextManager(ContextManager):
    """Limita archivos al alcance informado previamente por Explorer."""

    def select(
        self,
        instruction: str,
        task_state: TaskState,
        requested: AgentContext | None = None,
    ) -> AgentContext:
        explorer_files = tuple(
            dict.fromkeys(
                path
                for result in task_state.subagent_results
                if result.subagent_id == "explorer"
                for path in result.files_relevant
            )
        )
        selected = requested or AgentContext()
        files = (
            tuple(path for path in selected.files if path in explorer_files)
            if selected.files
            else explorer_files
        )
        return AgentContext(
            facts=selected.facts,
            sources=selected.sources,
            files=files,
            constraints=selected.constraints,
        )

"""Orquestación explícita de subagentes mediante Python normal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal, Protocol, Sequence

from agents.base import AgentExecutionError
from agents.implementer import ImplementerResult
from agents.project_memory import (
    MemoryCorruptionError,
    ProjectMemory,
    ProjectMemoryError,
)
from agents.researcher import ResearcherResult
from agents.reviewer import ReviewerResult
from agents.tester import TesterResult
from core.llm_client import LLMClient
from core.models import Message, PlanReview
from core.task_state import ErrorRecord, SourceReference, SubagentResult, TaskState


TaskKind = Literal["analysis", "change"]
OrchestrationStatus = Literal[
    "completed", "rejected", "blocked", "max_iterations"
]
PlanReviewer = Callable[[str], PlanReview]


@dataclass(frozen=True)
class TaskAnalysis:
    kind: TaskKind
    research_required: bool
    rationale: str


class TaskAnalyzer(Protocol):
    def analyze(self, request: str) -> TaskAnalysis:
        """Clasifica el efecto de la tarea sin inferir tecnologías."""


class PlanGenerator(Protocol):
    def generate(
        self,
        state: TaskState,
        *,
        feedback: Sequence[str] = (),
    ) -> str:
        """Genera un plan usando exclusivamente el estado y feedback provistos."""


class ExplorerRunner(Protocol):
    def run(self, instruction: str, task_state: TaskState, *args: object, **kwargs: object) -> SubagentResult:
        ...


class ResearcherRunner(Protocol):
    def run(self, instruction: str, task_state: TaskState, *args: object, **kwargs: object) -> ResearcherResult:
        ...


class ImplementerRunner(Protocol):
    def run(self, instruction: str, task_state: TaskState, *args: object, **kwargs: object) -> ImplementerResult:
        ...


class TesterRunner(Protocol):
    def run(self, instruction: str, task_state: TaskState, *args: object, **kwargs: object) -> TesterResult:
        ...


class ReviewerRunner(Protocol):
    def run(self, instruction: str, task_state: TaskState, *args: object, **kwargs: object) -> ReviewerResult:
        ...


class ResultPresenter(Protocol):
    def present(self, state: TaskState) -> str:
        """Construye la salida visible sin tomar decisiones de coordinación."""


class TextResultPresenter:
    """Presentación textual de resultados, checks, archivos y fuentes."""

    def present(self, state: TaskState) -> str:
        agents = "\n".join(
            f"- {result.subagent_id}: {result.status} — "
            f"{result.summary or result.result or 'sin resumen'}"
            for result in state.subagent_results
        ) or "- Ninguno."
        files = "\n".join(f"- {path}" for path in state.files_modified) or "- Ninguno."
        checks = "\n".join(
            f"- {call.arguments.get('command', call.tool_name)}: "
            f"{'ok' if call.success else 'falló'}"
            for call in state.tool_calls
            if call.tool_name == "validation_command"
        ) or "- No se ejecutaron checks."
        sources = "\n".join(
            f"- [{'inferido' if source.origin == 'inference' else 'utilizado'}:"
            f"{source.origin}] {source.reference}"
            for source in state.sources
        ) or "- Sin fuentes adicionales."
        rag_traces = "\n".join(
            f"- {observation.removeprefix('RAG trace: ')}"
            for observation in state.observations
            if observation.startswith("RAG trace: ")
        ) or "- Sin recuperación RAG registrada."
        return (
            f"{state.final_result}\n\nResultados:\n{agents}\n\n"
            f"Archivos modificados:\n{files}\n\nValidaciones:\n{checks}\n\n"
            f"Fuentes:\n{sources}\n\nTrazabilidad RAG (recuperado/utilizado):\n"
            f"{rag_traces}"
        )


class LLMTaskAnalyzer:
    """Clasificador neutral que sólo decide si la tarea puede modificar archivos."""

    _PROMPT = (
        "Clasificá el pedido sin asumir lenguaje, framework ni estructura. Respondé "
        "sólo JSON con kind (analysis o change), research_required (boolean) y "
        "rationale (texto). Una tarea es change si solicita crear, modificar o eliminar "
        "artefactos; research_required indica conocimiento técnico adicional."
    )

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def analyze(self, request: str) -> TaskAnalysis:
        response = self.llm_client.complete(
            [Message("system", self._PROMPT), Message("user", request)], ()
        )
        if response.tool_calls:
            raise AgentExecutionError("El analizador de tareas no puede usar tools.")
        try:
            payload = json.loads(response.text)
            kind = payload["kind"]
            research = payload["research_required"]
            rationale = payload["rationale"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise AgentExecutionError("Clasificación de tarea inválida.") from error
        if kind not in {"analysis", "change"}:
            raise AgentExecutionError("kind debe ser analysis o change.")
        if not isinstance(research, bool):
            raise AgentExecutionError("research_required debe ser booleano.")
        if not isinstance(rationale, str) or not rationale.strip():
            raise AgentExecutionError("rationale no puede estar vacío.")
        return TaskAnalysis(kind, research, rationale.strip())


class LLMPlanGenerator:
    """Planificador sin tools que recibe evidencia acotada del estado compartido."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        state: TaskState,
        *,
        feedback: Sequence[str] = (),
    ) -> str:
        evidence = {
            "request": state.original_request,
            "repository_findings": list(state.repository_findings),
            "relevant_files": list(
                dict.fromkeys(
                    path
                    for result in state.subagent_results
                    for path in result.files_relevant
                )
            ),
            "sources": [source.to_dict() for source in state.sources],
            "feedback": list(feedback),
        }
        messages = [
            Message(
                "system",
                "Generá un plan numerado, concreto y verificable usando sólo la "
                "evidencia provista. No asumas tecnologías ni ejecutes acciones.",
            ),
            Message("user", json.dumps(evidence, ensure_ascii=False)),
        ]
        response = self.llm_client.complete(messages, ())
        plan = response.text.strip()
        if response.tool_calls or not plan:
            raise AgentExecutionError("El planificador devolvió un plan inválido.")
        return plan


@dataclass(frozen=True)
class OrchestrationResult:
    status: OrchestrationStatus
    task_state: TaskState
    final_response: str
    iterations: int
    selected_agents: tuple[str, ...]


class MainAgent:
    """Coordina agentes especializados sin conocer el proyecto inspeccionado."""

    def __init__(
        self,
        *,
        task_analyzer: TaskAnalyzer,
        plan_generator: PlanGenerator,
        explorer: ExplorerRunner,
        researcher: ResearcherRunner | None,
        implementer: ImplementerRunner | None,
        tester: TesterRunner | None,
        reviewer: ReviewerRunner | None,
        project_memory: ProjectMemory | None = None,
        presenter: ResultPresenter | None = None,
        max_iterations: int = 3,
        minimum_evidence_confidence: float = 0.5,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations debe ser al menos 1.")
        if not 0 <= minimum_evidence_confidence <= 1:
            raise ValueError("minimum_evidence_confidence debe estar entre 0 y 1.")
        self.task_analyzer = task_analyzer
        self.plan_generator = plan_generator
        self.explorer = explorer
        self.researcher = researcher
        self.implementer = implementer
        self.tester = tester
        self.reviewer = reviewer
        self.project_memory = project_memory
        self.presenter = presenter or TextResultPresenter()
        self.max_iterations = max_iterations
        self.minimum_evidence_confidence = minimum_evidence_confidence
        self._memory_available = True

    def run(
        self,
        request: str,
        review_plan: PlanReviewer,
        *,
        task_id: str | None = None,
    ) -> OrchestrationResult:
        state = TaskState.create(request, task_id=task_id)
        selected: list[str] = []
        self._memory_available = True
        try:
            if self.project_memory is not None:
                self.project_memory.load()
            state.set_status("running")
            state.set_phase("analysis")
            analysis = self.task_analyzer.analyze(request)
            state.add_observation(
                f"Tarea clasificada como {analysis.kind}: {analysis.rationale}"
            )

            state.set_phase("exploration")
            self.explorer.run(request, state)
            selected.append("explorer")

            needs_research = analysis.research_required or analysis.kind == "change"
            if needs_research:
                if self.researcher is None:
                    return self._blocked(state, selected, 0, "Researcher no está configurado.")
                state.set_phase("research")
                research = self.researcher.run(request, state)
                selected.append("researcher")
                if (
                    research.subagent_result.status != "completed"
                    or research.confidence < self.minimum_evidence_confidence
                    or not research.sources_recovered
                ):
                    return self._blocked(
                        state, selected, 0, "La evidencia técnica es insuficiente."
                    )

            feedback: tuple[str, ...] = ()
            for iteration in range(1, self.max_iterations + 1):
                state.set_phase("planning")
                plan = self.plan_generator.generate(state, feedback=feedback)
                state.propose_plan(plan)
                decision: PlanReview = review_plan(plan)
                if decision.action == "reject":
                    state.set_status("rejected")
                    state.set_final_result("El usuario rechazó el plan; no se realizaron cambios.")
                    return self._result("rejected", state, iteration, selected)
                if decision.action == "modify":
                    if not decision.modification:
                        raise AgentExecutionError("La modificación del plan está vacía.")
                    feedback = (decision.modification,)
                    state.add_observation(
                        f"Modificación solicitada al plan: {decision.modification}"
                    )
                    continue
                if decision.action != "approve":
                    raise AgentExecutionError("Decisión de plan inválida.")
                state.approve_plan(plan)

                if analysis.kind == "analysis":
                    state.set_status("completed")
                    state.set_phase("finalization")
                    state.set_final_result(self._analysis_summary(state))
                    return self._result("completed", state, iteration, selected)

                missing = [
                    name
                    for name, agent in (
                        ("Implementer", self.implementer),
                        ("Tester", self.tester),
                        ("Reviewer", self.reviewer),
                    )
                    if agent is None
                ]
                if missing:
                    return self._blocked(
                        state, selected, iteration,
                        f"Faltan agentes requeridos: {', '.join(missing)}.",
                    )

                state.set_phase("implementation")
                assert self.implementer is not None
                implementation = self.implementer.run(
                    request, state, mode="apply_changes"
                )
                selected.append("implementer")
                if not implementation.files_modified:
                    return self._blocked(
                        state, selected, iteration,
                        "Implementer no produjo cambios verificables.",
                    )

                state.set_phase("testing")
                assert self.tester is not None
                testing = self.tester.run("Validar los cambios aplicados", state)
                selected.append("tester")

                state.set_phase("review")
                assert self.reviewer is not None
                review = self.reviewer.run("Revisar el resultado completo", state)
                selected.append("reviewer")
                if review.decision == "approved" and testing.status == "passed":
                    state.set_status("completed")
                    state.set_phase("finalization")
                    state.set_final_result(review.summary)
                    return self._result("completed", state, iteration, selected)
                if testing.status in {"blocked", "unavailable", "skipped"}:
                    return self._blocked(
                        state, selected, iteration,
                        f"La validación no pudo completarse: {testing.summary}",
                    )
                if review.decision in {"blocked", "insufficient_evidence"}:
                    return self._blocked(state, selected, iteration, review.summary)

                feedback = tuple(review.required_changes) or (review.summary,)
                if testing.status == "failed":
                    feedback = (*feedback, testing.summary)
                state.add_observation("Reviewer solicitó replanificar el trabajo.")

            state.set_status("max_iterations")
            state.set_phase("stopped")
            state.set_final_result(
                f"Se alcanzó el máximo de {self.max_iterations} iteraciones."
            )
            return self._result("max_iterations", state, self.max_iterations, selected)
        except MemoryCorruptionError as error:
            self._memory_available = False
            state.record_error(
                ErrorRecord(str(error), state.current_phase, "project_memory", False)
            )
            return self._blocked(state, selected, 0, str(error))
        except ProjectMemoryError as error:
            self._memory_available = False
            state.record_error(
                ErrorRecord(str(error), state.current_phase, "project_memory", True)
            )
            return self._blocked(state, selected, 0, str(error))
        except (PermissionError, AgentExecutionError) as error:
            state.record_error(
                ErrorRecord(str(error), state.current_phase, "main_agent", True)
            )
            return self._blocked(state, selected, 0, str(error))
        except Exception as error:
            state.record_error(
                ErrorRecord(str(error), state.current_phase, "main_agent", False)
            )
            return self._blocked(
                state, selected, 0, f"La coordinación no pudo continuar: {error}"
            )

    @staticmethod
    def _analysis_summary(state: TaskState) -> str:
        findings = "\n".join(f"- {item}" for item in state.repository_findings)
        return "Análisis completado sin cambios.\n" + (findings or "Sin hallazgos.")

    def _blocked(
        self,
        state: TaskState,
        selected: Sequence[str],
        iterations: int,
        reason: str,
    ) -> OrchestrationResult:
        state.set_status("blocked")
        state.set_phase("stopped")
        state.add_warning(reason)
        state.set_final_result(reason)
        return self._result("blocked", state, iterations, selected)

    def _result(
        self,
        status: OrchestrationStatus,
        state: TaskState,
        iterations: int,
        selected: Sequence[str],
    ) -> OrchestrationResult:
        if self.project_memory is not None and self._memory_available:
            try:
                self.project_memory.load()
                self.project_memory.save_task_summary(state)
                self.project_memory.save()
            except ProjectMemoryError as error:
                self._memory_available = False
                state.record_error(
                    ErrorRecord(str(error), state.current_phase, "project_memory", True)
                )
                state.add_warning("No se pudo persistir el resumen de la tarea.")
        final = self.presenter.present(state)
        return OrchestrationResult(
            status, state, final, iterations, tuple(dict.fromkeys(selected))
        )

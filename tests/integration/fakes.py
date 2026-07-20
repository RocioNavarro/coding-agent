"""Dobles deterministas reutilizables por escenarios integrales."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents.researcher import EvidenceFragment
from agents.tester import CommandOutcome
from core.models import LLMResponse, LLMUsage, Message, PlanReview
from core.models import EvidenceAssessment
from core.observability import ObservabilityEvent
from agents.orchestrator import TaskAnalysis
from agents.researcher import SufficiencyAssessment
from core.task_state import SubagentResult, TaskState


ScriptValue = str | Mapping[str, Any] | LLMResponse | Exception


class StaticTaskAnalyzer:
    def __init__(self, kind: str, *, research_required: bool = False) -> None:
        self.result = TaskAnalysis(kind, research_required, "deterministic integration classification")
        self.calls: list[str] = []

    def analyze(self, request: str) -> TaskAnalysis:
        self.calls.append(request)
        return self.result


class ScriptedPlanner:
    def __init__(self, plans: Sequence[str]) -> None:
        self._plans = deque(plans)
        self.calls: list[tuple[str, ...]] = []

    def generate(self, state: TaskState, *, feedback=()) -> str:
        self.calls.append(tuple(feedback))
        if not self._plans:
            raise AssertionError("No quedan planes configurados.")
        return self._plans.popleft()


class FixedSufficiencyEvaluator:
    def __init__(self, *, sufficient: bool = True, confidence: float = 0.9) -> None:
        self.sufficient = sufficient
        self.confidence = confidence
        self.calls: list[tuple[str, tuple[EvidenceFragment, ...]]] = []

    def evaluate(self, query: str, fragments: Sequence[EvidenceFragment]):
        selected = tuple(fragments)
        self.calls.append((query, selected))
        return SufficiencyAssessment(
            self.sufficient, self.confidence,
            () if self.sufficient else ("insufficient deterministic evidence",),
        )


class ForbiddenRunner:
    """Spy que falla si un escenario alcanza un componente prohibido."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def assess_evidence(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(f"{self.name} no debía evaluar evidencia.")

    def run(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(f"{self.name} no debía ejecutarse.")


@dataclass(frozen=True)
class FakeImplementationResult:
    files_modified: tuple[str, ...]


class StateRecordingImplementer:
    """Implementer fake sin I/O, útil cuando el escenario ejercita otra fase."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.modes: list[str] = []

    def assess_evidence(self, instruction, state, *, validation_available):
        return EvidenceAssessment(
            "sufficient", (self.path,), (), (), "proceed", 1.0
        )

    def run(self, instruction, state, *args, **kwargs):
        mode = kwargs.get("mode", "propose_only")
        self.modes.append(mode)
        if mode == "apply_changes":
            state.record_file_modified(self.path)
            state.add_subagent_result(SubagentResult(
                "implementer", instruction, "completed",
                summary="Deterministic implementation state recorded.",
                files_relevant=(self.path,), confidence=1.0,
            ))
            return FakeImplementationResult((self.path,))
        return FakeImplementationResult(())


class ScriptedLLM:
    """LLM fake con guiones secuenciales globales o separados por fase."""

    def __init__(
        self,
        responses: Sequence[ScriptValue] = (),
        *,
        by_phase: Mapping[str, Sequence[ScriptValue]] | None = None,
    ) -> None:
        self._responses = deque(responses)
        self._by_phase = {name: deque(items) for name, items in (by_phase or {}).items()}
        self.phase = "default"
        self.prompts: list[tuple[str, tuple[Message, ...]]] = []
        self.tools: list[tuple[dict[str, Any], ...]] = []

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def complete(self, messages, tools=()) -> LLMResponse:
        self.prompts.append((self.phase, tuple(messages)))
        self.tools.append(tuple(tools))
        queue = self._by_phase.get(self.phase, self._responses)
        if not queue:
            raise AssertionError(f"No hay respuesta fake para la fase '{self.phase}'.")
        value = queue.popleft()
        if isinstance(value, Exception):
            raise value
        if isinstance(value, LLMResponse):
            return value
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, Mapping) else value
        return LLMResponse(
            Message("assistant", text), text, [], "fake-model",
            LLMUsage(3, 2, 5), 1.0,
        )


class FakeObservability:
    """Cliente fake con jerarquía, aperturas/cierres y fallo opcional."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[ObservabilityEvent] = []
        self.open_traces: set[str] = set()
        self.closed_traces: set[str] = set()
        self.flushed = False

    def record(self, event: ObservabilityEvent) -> None:
        if self.fail:
            raise RuntimeError("observability unavailable")
        self.events.append(event)
        if event.event_type == "task" and event.event_id:
            self.open_traces.add(event.event_id)
        if event.event_type in {"result", "error"} and event.parent_event_id:
            self.open_traces.discard(event.parent_event_id)
            self.closed_traces.add(event.parent_event_id)

    def flush(self) -> None:
        if self.fail:
            raise RuntimeError("observability flush unavailable")
        self.flushed = True

    def children_of(self, event_id: str) -> tuple[ObservabilityEvent, ...]:
        return tuple(event for event in self.events if event.parent_event_id == event_id)


class ScriptedPlanReview:
    """Callback de plan con decisiones únicas o secuenciales."""

    def __init__(self, decisions: Sequence[PlanReview]) -> None:
        self._decisions = deque(decisions)
        self.plans: list[str] = []

    @classmethod
    def approve(cls) -> "ScriptedPlanReview":
        return cls((PlanReview("approve"),))

    @classmethod
    def reject(cls) -> "ScriptedPlanReview":
        return cls((PlanReview("reject"),))

    @classmethod
    def modify(cls, instruction: str) -> "ScriptedPlanReview":
        return cls((PlanReview("modify", instruction), PlanReview("approve")))

    def __call__(self, plan: str) -> PlanReview:
        self.plans.append(plan)
        if not self._decisions:
            raise AssertionError("No quedan decisiones de plan configuradas.")
        return self._decisions.popleft()


class ScriptedCommandRunner:
    """ValidationExecutor fake con resultados diferentes por intento."""

    def __init__(self, outcomes: Mapping[str, Sequence[CommandOutcome]]) -> None:
        self._outcomes = {command: deque(items) for command, items in outcomes.items()}
        self.calls: list[tuple[str, float]] = []

    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        self.calls.append((command, timeout_seconds))
        queue = self._outcomes.get(command)
        if not queue:
            raise AssertionError(f"No hay resultado configurado para '{command}'.")
        return queue.popleft()


@dataclass(frozen=True)
class FakeReviewerResult:
    decision: str
    summary: str
    required_changes: tuple[str, ...] = ()


class ScriptedReviewer:
    """Reviewer fake compatible con MainAgent y replanificación."""

    def __init__(self, decisions: Sequence[str]) -> None:
        self._decisions = deque(decisions)
        self.calls: list[str] = []

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeReviewerResult:
        self.calls.append(instruction)
        if not self._decisions:
            raise AssertionError("No quedan decisiones de Reviewer.")
        decision = self._decisions.popleft()
        required = ("Cambiar estrategia con nueva evidencia.",) if decision == "changes_requested" else ()
        state.add_subagent_result(
            SubagentResult("reviewer", instruction, decision, summary=f"Reviewer {decision}.")
        )
        return FakeReviewerResult(decision, f"Reviewer {decision}.", required)


class FakeRAG:
    def __init__(self, fragments: Sequence[EvidenceFragment] = (), *, observability=None) -> None:
        self.fragments = tuple(fragments)
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []
        self.observability = observability

    def retrieve(self, query: str, *, limit: int = 5):
        return self.retrieve_filtered(query, filters=None, limit=limit)

    def retrieve_filtered(self, query: str, *, filters=None, limit: int = 5):
        self.calls.append((query, dict(filters or {}), limit))
        if self.observability is not None:
            self.observability.record(ObservabilityEvent(
                "rag", "fake-rag-retrieval",
                payload={"query": query, "retrieved": len(self.fragments[:limit])},
            ))
        return self.fragments[:limit]

    def retrieval_audit(self):
        chunks = [
            {"chunk_id": f"chunk-{index}", "score": fragment.relevance,
             "metadata": {"document_id": fragment.reference,
                          "path_or_url": fragment.reference}}
            for index, fragment in enumerate(self.fragments, 1)
        ]
        return {"query": self.calls[-1][0], "filters": self.calls[-1][1],
                "retrieved_chunks": chunks, "used_chunks": chunks, 
                "scores": {item["chunk_id"]: item["score"] for item in chunks},
                "documents": [item.reference for item in self.fragments],
                "conclusions": ["deterministic relevant chunks"],
                "sufficiency": {"sufficient": bool(self.fragments), "confidence": 1.0 if self.fragments else 0.0}}


class FakeMemory:
    def __init__(self, fragments: Sequence[EvidenceFragment] = (), *, observability=None) -> None:
        self.fragments = tuple(fragments)
        self.queries: list[tuple[str, int]] = []
        self.writes: list[dict[str, Any]] = []
        self.observability = observability

    def search(self, query: str, *, limit: int = 5):
        self.queries.append((query, limit))
        if self.observability is not None:
            self.observability.record(ObservabilityEvent(
                "agent", "fake-memory-query",
                payload={"query": query, "result_count": len(self.fragments[:limit])},
            ))
        return self.fragments[:limit]

    def save_task_summary(self, state: TaskState) -> None:
        self.writes.append({"task_id": state.task_id, "status": state.current_status})


class FakeWeb:
    def __init__(self, fragments: Sequence[EvidenceFragment] = (), *, observability=None) -> None:
        self.fragments = tuple(fragments)
        self.calls: list[tuple[str, int]] = []
        self.observability = observability

    def search(self, query: str, *, limit: int = 5):
        self.calls.append((query, limit))
        if self.observability is not None:
            self.observability.record(ObservabilityEvent(
                "web", "fake-web-search", payload={"query": query},
            ))
        return self.fragments[:limit]

    def search_context(self, query: str, *, limit: int = 5, **kwargs):
        return self.search(query, limit=limit)

    def search_audit(self):
        return {"query": self.calls[-1][0], "executed_queries": [self.calls[-1][0]],
                "found": [], "used": [], "conclusions": []}

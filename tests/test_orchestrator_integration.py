"""Integración del orquestador con subagentes simulados y estado real."""

from dataclasses import dataclass
from types import SimpleNamespace

from agents.orchestrator import MainAgent, TaskAnalysis
from agents.project_memory import ProjectMemory
from core.models import EvidenceAssessment, PlanReview
from core.task_state import SourceReference, SubagentResult, TaskState
from core.observability import ObservabilityEvent
from core.progress import ProgressAssessment
from core.planned_operations import (
    PlannedOperation, PlannedOperationResult, PolicyPreflight,
)
from core.settings import AgentSettings
from security.policy_engine import AgentToolPermissions, PolicyContext, PolicyEngine


class FakeObservability:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def record(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None


class FakeAnalyzer:
    def __init__(self, kind: str = "change", research_required: bool = False) -> None:
        self.result = TaskAnalysis(kind, research_required, "clasificación simulada")

    def analyze(self, request: str) -> TaskAnalysis:
        return self.result


class FakePlanner:
    def __init__(self) -> None:
        self.feedback: list[tuple[str, ...]] = []

    def generate(self, state: TaskState, *, feedback=()) -> str:
        self.feedback.append(tuple(feedback))
        return f"Plan {len(self.feedback)} para {state.original_request}"


class FakeExplorer:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> SubagentResult:
        self.calls += 1
        state.add_repository_finding(
            "Arquitectura e impacto detectados; evidencia: src/app.txt."
        )
        state.add_source(SourceReference("repository", "src/app.txt", "archivo relevante"))
        result = SubagentResult(
            "explorer", instruction, "completed", summary="Repositorio explorado.",
            files_relevant=("src/app.txt",), confidence=0.9,
        )
        state.add_subagent_result(result)
        return result


@dataclass
class FakeResearchResult:
    subagent_result: SubagentResult
    confidence: float
    sources_recovered: tuple[SourceReference, ...]


class FakeResearcher:
    def __init__(self, *, sufficient: bool = True) -> None:
        self.sufficient = sufficient
        self.calls = 0

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeResearchResult:
        self.calls += 1
        sources = (
            (SourceReference("rag", "knowledge://evidence", "evidencia"),)
            if self.sufficient else ()
        )
        result = SubagentResult(
            "researcher", instruction,
            "completed" if self.sufficient else "blocked",
            summary="Investigación completada.", sources=sources,
            confidence=0.9 if self.sufficient else 0.1,
        )
        for source in sources:
            state.add_source(source)
        state.add_subagent_result(result)
        return FakeResearchResult(result, result.confidence or 0.0, sources)


@dataclass
class FakeImplementationResult:
    files_modified: tuple[str, ...]


class FakeImplementer:
    def __init__(self, evidence_status: str = "sufficient") -> None:
        self.calls = 0
        self.assessment_calls = 0
        self.evidence_status = evidence_status

    def assess_evidence(
        self, instruction: str, state: TaskState, *, validation_available: bool
    ) -> EvidenceAssessment:
        self.assessment_calls += 1
        if self.evidence_status == "sufficient":
            return EvidenceAssessment("sufficient", ("src/app.txt",), (), (), "proceed", 1.0)
        if self.evidence_status == "partial":
            return EvidenceAssessment(
                "partial", (), ("impacto del cambio",), (),
                "gather_more_evidence", 0.5,
            )
        return EvidenceAssessment(
            "insufficient", (), ("permisos de modificación",),
            ("No hay permisos suficientes.",), "request_help", 1.0,
        )

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeImplementationResult:
        self.calls += 1
        state.record_file_modified("src/app.txt")
        state.add_subagent_result(
            SubagentResult(
                "implementer", instruction, "completed", summary="Cambio aplicado.",
                files_relevant=("src/app.txt",), confidence=0.9,
            )
        )
        return FakeImplementationResult(("src/app.txt",))


@dataclass
class FakeTesterResult:
    status: str
    summary: str
    progress_assessments: tuple[ProgressAssessment, ...] = ()


class FakeTester:
    def __init__(
        self,
        statuses: tuple[str, ...] = ("passed",),
        progress_assessments: tuple[ProgressAssessment, ...] = (),
    ) -> None:
        self.statuses = statuses
        self.progress_assessments = progress_assessments
        self.calls = 0

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeTesterResult:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        summary = f"Tester {status}."
        state.add_subagent_result(
            SubagentResult("tester", instruction, status, summary=summary, confidence=1.0)
        )
        return FakeTesterResult(status, summary, self.progress_assessments)


@dataclass
class FakeReviewResult:
    decision: str
    summary: str
    required_changes: tuple[str, ...]


class FakeReviewer:
    def __init__(self, decisions: tuple[str, ...] = ("approved",)) -> None:
        self.decisions = decisions
        self.calls = 0

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeReviewResult:
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        required = ("Corregir el cambio observado.",) if decision == "changes_requested" else ()
        summary = f"Reviewer {decision}."
        state.add_subagent_result(
            SubagentResult("reviewer", instruction, decision, summary=summary)
        )
        return FakeReviewResult(decision, summary, required)


def build_agent(
    *,
    kind: str = "change",
    researcher: FakeResearcher | None = None,
    tester: FakeTester | None = None,
    reviewer: FakeReviewer | None = None,
    max_iterations: int = 3,
    evidence_status: str = "sufficient",
    review_analysis_tasks: bool = False,
) -> tuple[MainAgent, FakePlanner, FakeImplementer, FakeTester, FakeReviewer]:
    planner = FakePlanner()
    implementer = FakeImplementer(evidence_status)
    selected_tester = tester or FakeTester()
    selected_reviewer = reviewer or FakeReviewer()
    agent = MainAgent(
        task_analyzer=FakeAnalyzer(kind),
        plan_generator=planner,
        explorer=FakeExplorer(),
        researcher=researcher or FakeResearcher(),
        implementer=implementer,
        tester=selected_tester,
        reviewer=selected_reviewer,
        max_iterations=max_iterations,
        review_analysis_tasks=review_analysis_tasks,
    )
    return agent, planner, implementer, selected_tester, selected_reviewer


def approve(_: str) -> PlanReview:
    return PlanReview("approve")


def attach_preflight(
    agent: MainAgent,
    tmp_path,
    *,
    allowed_tools: frozenset[str] | None = None,
    approval_tools: frozenset[str] = frozenset(),
) -> None:
    workspace = tmp_path / "preflight-workspace"
    workspace.mkdir(exist_ok=True)
    agent.policy_preflight = PolicyPreflight(
        PolicyEngine(),
        PolicyContext(
            agent="main", workspace=workspace,
            permissions=AgentToolPermissions(allowed_tools, approval_tools),
            settings=AgentSettings(supervision_enabled=False),
        ),
    )


def test_analysis_without_changes_selects_only_needed_agents() -> None:
    agent, _, implementer, tester, reviewer = build_agent(kind="analysis")

    result = agent.run("Explicar la arquitectura", approve, task_id="analysis")

    assert result.status == "completed"
    assert result.selected_agents == ("explorer",)
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.files_modified == ()
    assert "Análisis completado sin cambios" in result.final_response


def test_analysis_can_optionally_run_reviewer_without_implementation() -> None:
    agent, _, implementer, tester, reviewer = build_agent(
        kind="analysis", review_analysis_tasks=True
    )

    result = agent.run("Explicar la arquitectura", approve, task_id="review-analysis")

    assert result.status == "completed"
    assert result.selected_agents == ("explorer", "reviewer")
    assert reviewer.calls == 1
    assert implementer.calls == tester.calls == 0
    assert result.task_state.files_modified == ()


def test_change_runs_full_pipeline_after_approval() -> None:
    agent, _, implementer, tester, reviewer = build_agent()

    result = agent.run("Modificar el comportamiento", approve)

    assert result.status == "completed"
    assert result.selected_agents == (
        "explorer", "researcher", "implementer", "tester", "reviewer"
    )
    assert implementer.calls == tester.calls == reviewer.calls == 1
    assert result.task_state.approved_plan == "Plan 1 para Modificar el comportamiento"
    assert "[utilizado:rag] knowledge://evidence" in result.final_response


def test_orchestrator_records_root_unique_agents_and_result() -> None:
    agent, _, _, _, _ = build_agent()
    observed = FakeObservability()
    agent.observability = observed

    result = agent.run("Modificar observado", approve, task_id="observed-task")

    assert result.status == "completed"
    assert [event.event_type for event in observed.events].count("task") == 1
    assert [event.event_type for event in observed.events].count("result") == 1
    names = [
        event.name for event in observed.events
        if event.event_type == "agent" and event.name != "evidence-sufficiency-policy"
    ]
    assert names == ["explorer", "researcher", "implementer", "tester", "reviewer"]
    assert all(event.parent_event_id == "task:observed-task" for event in observed.events[1:])


def test_evidence_assessment_is_observable_once_per_evaluation() -> None:
    agent, _, implementer, _, _ = build_agent()
    observed = FakeObservability()
    agent.observability = observed

    result = agent.run("Modificar con evidencia observable", approve, task_id="evidence-event")

    events = [
        event for event in observed.events
        if event.name == "evidence-sufficiency-policy"
    ]
    assert result.status == "completed"
    assert len(events) == implementer.assessment_calls == 1
    assert events[0].payload == {
        "status": "sufficient",
        "blockers": (),
        "missing_information": (),
        "risks": (),
        "recommended_action": "proceed",
        "confidence": 1.0,
        "task_id": "evidence-event",
        "plan": "Plan 1 para Modificar con evidencia observable",
    }


def test_progress_recommendation_reaches_planner() -> None:
    assessment = ProgressAssessment(
        True, "command_error", "retry_with_new_strategy",
        "Cambiar estrategia de validación.", 2,
    )
    tester = FakeTester(("failed", "passed"), (assessment,))
    reviewer = FakeReviewer(("changes_requested", "approved"))
    agent, planner, _, _, _ = build_agent(tester=tester, reviewer=reviewer)

    result = agent.run("Corregir validación", approve)

    assert result.status == "completed"
    assert "Cambiar estrategia de validación." in planner.feedback[1]
    assert any(
        "ProgressMonitor recomendó retry_with_new_strategy" in item
        for item in result.task_state.observations
    )


def test_progress_stop_blocks_before_reviewer_and_repetition() -> None:
    assessment = ProgressAssessment(
        True, "no_new_evidence", "stop", "Sin nueva evidencia.", 3
    )
    tester = FakeTester(("failed",), (assessment,))
    reviewer = FakeReviewer()
    agent, planner, implementer, _, _ = build_agent(
        tester=tester, reviewer=reviewer, max_iterations=3
    )

    result = agent.run("Detener si no progresa", approve)

    assert result.status == "blocked"
    assert result.iterations == 1
    assert reviewer.calls == 0
    assert implementer.calls == 1
    assert planner.feedback == [()]
    assert result.final_response.startswith("Sin nueva evidencia.")


def test_preflight_deny_blocks_before_implementer_and_records_state(tmp_path) -> None:
    agent, _, implementer, tester, reviewer = build_agent()
    observed = FakeObservability()
    agent.observability = observed
    attach_preflight(agent, tmp_path, allowed_tools=frozenset())

    result = agent.run("Modificar con política", approve, task_id="preflight-deny")

    assert result.status == "blocked"
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.planned_operations[0]["operation_type"] == "modify_file"
    assert result.task_state.policy_preflight[0]["outcome"] == "deny"
    assert any(event.name == "policy-preflight" for event in observed.events)


def test_preflight_requires_exact_approval_before_continuing(tmp_path) -> None:
    agent, _, implementer, _, _ = build_agent()
    attach_preflight(
        agent, tmp_path, approval_tools=frozenset({"write_file"})
    )
    requested = []

    result = agent.run(
        "Modificar aprobado", approve,
        approve_operation=lambda operation: requested.append(operation) or True,
    )

    assert result.status == "completed"
    assert implementer.calls == 1
    assert len(requested) == 1
    assert result.task_state.policy_approvals == (requested[0].fingerprint,)
    assert result.task_state.policy_preflight[0]["outcome"] == "allow"


def test_preflight_without_structured_intent_blocks_before_implementer(tmp_path) -> None:
    class EmptyProvider:
        def provide(self, approved_plan, state, explorer_results, proposal=None):
            return PlannedOperationResult(
                missing_information=("Falta intención estructurada.",), confidence=0.0
            )

    agent, _, implementer, tester, reviewer = build_agent()
    agent.operation_provider = EmptyProvider()
    attach_preflight(agent, tmp_path)

    result = agent.run("Modificar sin intención", approve)

    assert result.status == "blocked"
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert "insufficient_structured_intent" in result.final_response


def test_observability_error_does_not_change_preflight_denial(tmp_path) -> None:
    class FailingObservability:
        def record(self, event):
            raise RuntimeError("provider failed")

        def flush(self):
            return None

    agent, _, implementer, _, _ = build_agent()
    agent.observability = FailingObservability()
    attach_preflight(agent, tmp_path, allowed_tools=frozenset())

    result = agent.run("Modificar bloqueado", approve)

    assert result.status == "blocked"
    assert implementer.calls == 0


def test_propose_only_can_supply_missing_structured_intent_once(tmp_path) -> None:
    class ProposalAwareProvider:
        def provide(self, approved_plan, state, explorer_results, proposal=None):
            if proposal is None:
                return PlannedOperationResult(
                    missing_information=("Falta propuesta estructurada.",)
                )
            return PlannedOperationResult(
                (
                    PlannedOperation(
                        "proposal-1", "modify_file", "implementer_propose_only",
                        "src/app.txt", {"path": "src/app.txt", "new_text": "changed"},
                        "plan-version",
                    ),
                ),
                confidence=1.0,
                provenance=("implementer_propose_only",),
            )

    class ProposalImplementer(FakeImplementer):
        def __init__(self):
            super().__init__()
            self.modes = []

        def run(self, instruction, state, *args, **kwargs):
            mode = kwargs.get("mode")
            self.modes.append(mode)
            if mode == "propose_only":
                return SimpleNamespace(changes=(SimpleNamespace(
                    path="src/app.txt", old_text="old", new_text="changed",
                    explanation="localized",
                ),))
            return super().run(instruction, state, *args, **kwargs)

    agent, _, _, _, _ = build_agent()
    implementer = ProposalImplementer()
    agent.implementer = implementer
    agent.operation_provider = ProposalAwareProvider()
    agent.use_propose_only_for_intent = True
    agent.max_intent_attempts = 1
    attach_preflight(agent, tmp_path)

    result = agent.run("Modificar con propuesta", approve)

    assert result.status == "completed"
    assert implementer.modes == ["propose_only", "apply_changes"]


def test_rejected_plan_stops_before_changes() -> None:
    agent, _, implementer, tester, reviewer = build_agent()

    result = agent.run("Modificar", lambda _: PlanReview("reject"))

    assert result.status == "rejected"
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.files_modified == ()


def test_plan_modification_generates_new_plan_before_execution() -> None:
    agent, planner, implementer, _, _ = build_agent()
    decisions = iter((PlanReview("modify", "Limitar el alcance"), PlanReview("approve")))

    result = agent.run("Modificar", lambda _: next(decisions))

    assert result.status == "completed"
    assert planner.feedback == [(), ("Limitar el alcance",)]
    assert result.task_state.approved_plan == "Plan 2 para Modificar"
    assert implementer.calls == 1


def test_failed_tester_result_is_forwarded_to_replanning() -> None:
    tester = FakeTester(("failed", "passed"))
    reviewer = FakeReviewer(("changes_requested", "approved"))
    agent, planner, _, _, _ = build_agent(tester=tester, reviewer=reviewer)

    result = agent.run("Corregir", approve)

    assert result.status == "completed"
    assert tester.calls == 2
    assert "Tester failed." in planner.feedback[1]


def test_reviewer_requested_changes_trigger_another_iteration() -> None:
    reviewer = FakeReviewer(("changes_requested", "approved"))
    agent, planner, implementer, _, _ = build_agent(reviewer=reviewer)

    result = agent.run("Cambiar", approve)

    assert result.status == "completed"
    assert implementer.calls == 2
    assert planner.feedback[1] == ("Corregir el cambio observado.",)


def test_stops_at_maximum_iterations() -> None:
    reviewer = FakeReviewer(("changes_requested",))
    agent, _, implementer, _, _ = build_agent(
        reviewer=reviewer, max_iterations=2
    )

    result = agent.run("Cambiar", approve)

    assert result.status == "max_iterations"
    assert result.iterations == 2
    assert implementer.calls == 2
    assert result.task_state.current_phase == "stopped"


def test_stops_when_research_evidence_is_insufficient() -> None:
    researcher = FakeResearcher(sufficient=False)
    agent, planner, implementer, tester, reviewer = build_agent(researcher=researcher)

    result = agent.run("Modificar con evidencia", approve)

    assert result.status == "blocked"
    assert planner.feedback == []
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert "evidencia técnica es insuficiente" in result.final_response


def test_partial_evidence_reexplores_and_never_reaches_implementation() -> None:
    agent, _, implementer, tester, reviewer = build_agent(evidence_status="partial")

    result = agent.run("Modificar con evidencia parcial", approve)

    assert result.status == "blocked"
    assert implementer.assessment_calls == 2
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.evidence_assessment is not None
    assert result.task_state.evidence_assessment.status == "partial"
    assert "gather_more_evidence" in result.final_response


def test_insufficient_evidence_preserves_structured_blockers() -> None:
    agent, _, implementer, tester, reviewer = build_agent(evidence_status="insufficient")

    result = agent.run("Modificar sin permisos", approve)

    assert result.status == "blocked"
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.files_modified == ()
    assert '"missing_information": ["permisos de modificación"]' in result.final_response
    assert '"recommended_action": "request_help"' in result.final_response


def test_main_agent_loads_memory_and_persists_session_summary(tmp_path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    memory = ProjectMemory(workspace, storage_root=tmp_path / "memory")
    memory.add_decision("Conservar compatibilidad")
    memory.save()
    agent, _, _, _, _ = build_agent(kind="analysis")
    agent.project_memory = memory

    result = agent.run("Analizar arquitectura", approve, task_id="memory-task")

    persisted = ProjectMemory(workspace, storage_root=tmp_path / "memory").load().data
    assert result.status == "completed"
    assert persisted["decisions"][0]["decision"] == "Conservar compatibilidad"
    assert persisted["previous_tasks"][0]["task_id"] == "memory-task"
    assert persisted["session_summaries"][0]["agents"] == ["explorer"]

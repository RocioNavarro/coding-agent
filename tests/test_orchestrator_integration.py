"""Integración del orquestador con subagentes simulados y estado real."""

from dataclasses import dataclass

from agents.orchestrator import MainAgent, TaskAnalysis
from agents.project_memory import ProjectMemory
from core.models import PlanReview
from core.task_state import SourceReference, SubagentResult, TaskState


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
        state.add_repository_finding("Arquitectura detectada; evidencia: src/app.txt.")
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
    def __init__(self) -> None:
        self.calls = 0

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


class FakeTester:
    def __init__(self, statuses: tuple[str, ...] = ("passed",)) -> None:
        self.statuses = statuses
        self.calls = 0

    def run(self, instruction: str, state: TaskState, *args, **kwargs) -> FakeTesterResult:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        summary = f"Tester {status}."
        state.add_subagent_result(
            SubagentResult("tester", instruction, status, summary=summary, confidence=1.0)
        )
        return FakeTesterResult(status, summary)


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
) -> tuple[MainAgent, FakePlanner, FakeImplementer, FakeTester, FakeReviewer]:
    planner = FakePlanner()
    implementer = FakeImplementer()
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
    )
    return agent, planner, implementer, selected_tester, selected_reviewer


def approve(_: str) -> PlanReview:
    return PlanReview("approve")


def test_analysis_without_changes_selects_only_needed_agents() -> None:
    agent, _, implementer, tester, reviewer = build_agent(kind="analysis")

    result = agent.run("Explicar la arquitectura", approve, task_id="analysis")

    assert result.status == "completed"
    assert result.selected_agents == ("explorer",)
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert result.task_state.files_modified == ()
    assert "Análisis completado sin cambios" in result.final_response


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

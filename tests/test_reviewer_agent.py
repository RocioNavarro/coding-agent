"""Tests del Reviewer genérico y sus controles basados en evidencia."""

import json
from collections.abc import Sequence
from typing import Any

from agents.reviewer import DiffProvider, DiffSnapshot, ReviewerAgent
from core.models import LLMResponse, LLMUsage, Message
from core.task_state import (
    SourceReference,
    SubagentResult,
    TaskState,
    ToolExecutionRecord,
)


class FakeDiffProvider(DiffProvider):
    def __init__(self, snapshot: DiffSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, ...]] = []

    def get_diff(self, modified_files: Sequence[str]) -> DiffSnapshot:
        self.calls.append(tuple(modified_files))
        return self.snapshot


class FakeReviewerLLM:
    def __init__(
        self,
        *,
        decision: str = "approved",
        summary: str = "El cambio cumple el pedido y cuenta con validación.",
        issues: list[dict[str, Any]] | None = None,
        severity: str = "none",
        required_changes: list[str] | None = None,
        optional_suggestions: list[str] | None = None,
        confidence: float = 0.9,
    ) -> None:
        self.payload = {
            "decision": decision,
            "summary": summary,
            "issues": issues or [],
            "severity": severity,
            "required_changes": required_changes or [],
            "optional_suggestions": optional_suggestions or [],
            "confidence": confidence,
        }
        self.calls: list[tuple[Sequence[Message], Sequence[dict[str, Any]]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        text = json.dumps(self.payload)
        return LLMResponse(
            assistant_message=Message("assistant", text),
            text=text,
            tool_calls=[],
            model="fake-reviewer",
            usage=LLMUsage(10, 5, 15),
            latency_ms=1.0,
        )


def ready_state(
    *modified_files: str,
    explorer_files: tuple[str, ...] | None = None,
    implementer_files: tuple[str, ...] | None = None,
    tester_status: str = "passed",
    include_research: bool = True,
) -> TaskState:
    state = TaskState.create("Agregar comportamiento solicitado", task_id="review-task")
    state.propose_plan("Cambiar sólo los archivos relevantes y validar el resultado.")
    state.approve_plan()
    state.add_repository_finding("Arquitectura: código en src y tests separados.")
    state.add_repository_finding("Convención detectada en archivos existentes.")

    relevant = explorer_files or tuple(modified_files)
    implemented = implementer_files or tuple(modified_files)
    state.add_subagent_result(
        SubagentResult(
            "explorer",
            "Explorar",
            "completed",
            summary="Estructura y convenciones identificadas.",
            findings=("src contiene el código principal",),
            files_relevant=relevant,
            confidence=0.9,
        )
    )
    source = SourceReference("rag", "knowledge://change", "Evidencia técnica")
    if include_research:
        state.add_source(source)
        state.add_subagent_result(
            SubagentResult(
                "researcher",
                "Investigar",
                "completed",
                summary="Evidencia suficiente.",
                sources=(source,),
                confidence=0.85,
            )
        )
    state.add_subagent_result(
        SubagentResult(
            "implementer",
            "Implementar",
            "completed",
            summary="Cambio localizado aplicado.",
            files_relevant=implemented,
            confidence=0.9,
        )
    )
    state.add_subagent_result(
        SubagentResult(
            "tester",
            "Validar",
            tester_status,
            summary=f"Validación {tester_status}.",
            findings=("check registrado",),
            blockers=("El check falló",) if tester_status == "failed" else (),
            confidence=0.95,
        )
    )
    state.record_tool_call(
        ToolExecutionRecord(
            "validation-1",
            "validation_command",
            {"command": "comando descubierto"},
            tester_status == "passed",
            result={"exit_code": 0 if tester_status == "passed" else 1},
        )
    )
    for path in modified_files:
        state.record_file_modified(path)
    return state


def snapshot(*files: str) -> DiffSnapshot:
    return DiffSnapshot(
        "\n".join(f"diff --git a/{path} b/{path}" for path in files),
        tuple(files),
    )


def test_approves_evidenced_change_and_records_result() -> None:
    state = ready_state("src/service.txt")
    llm = FakeReviewerLLM(optional_suggestions=["Documentar en una etapa futura."])
    provider = FakeDiffProvider(snapshot("src/service.txt"))
    reviewer = ReviewerAgent(llm_client=llm, diff_provider=provider)

    result = reviewer.run("Revisar el resultado", state)

    assert result.decision == "approved"
    assert result.severity == "none"
    assert provider.calls == [("src/service.txt",)]
    assert llm.calls[0][1] == ()
    assert state.subagent_results[-1].subagent_id == "reviewer"
    assert state.subagent_results[-1].status == "approved"
    assert state.sources[-1].origin == "inference"
    assert "decision=approved" in state.observations[-1]


def test_failed_tests_cannot_be_overridden_by_llm_approval() -> None:
    state = ready_state("src/service.txt", tester_status="failed")
    reviewer = ReviewerAgent(
        llm_client=FakeReviewerLLM(decision="approved"),
        diff_provider=FakeDiffProvider(snapshot("src/service.txt")),
    )

    result = reviewer.run("Revisar", state)

    assert result.decision == "changes_requested"
    assert result.severity == "high"
    assert {issue.code for issue in result.issues} == {"tests_failed"}
    assert "Tester informó" in result.required_changes[0]


def test_requests_changes_for_file_outside_explorer_scope() -> None:
    state = ready_state(
        "unexpected.txt",
        explorer_files=("src/service.txt",),
        implementer_files=("unexpected.txt",),
    )
    reviewer = ReviewerAgent(
        llm_client=FakeReviewerLLM(),
        diff_provider=FakeDiffProvider(snapshot("unexpected.txt")),
    )

    result = reviewer.run("Revisar alcance", state)

    assert result.decision == "changes_requested"
    issue = next(issue for issue in result.issues if issue.code == "out_of_scope")
    assert issue.evidence == ("unexpected.txt",)


def test_reports_insufficient_evidence_without_calling_llm() -> None:
    state = ready_state("src/service.txt", include_research=False)
    llm = FakeReviewerLLM()
    reviewer = ReviewerAgent(
        llm_client=llm,
        diff_provider=FakeDiffProvider(snapshot("src/service.txt")),
    )

    result = reviewer.run("Revisar", state)

    assert result.decision == "insufficient_evidence"
    assert result.issues[0].code == "insufficient_evidence"
    assert "Researcher" in result.summary
    assert llm.calls == []


def test_detects_modified_file_not_declared_by_implementer() -> None:
    state = ready_state(
        "src/service.txt",
        "docs/notes.txt",
        explorer_files=("src/service.txt", "docs/notes.txt"),
        implementer_files=("src/service.txt",),
    )
    reviewer = ReviewerAgent(
        llm_client=FakeReviewerLLM(),
        diff_provider=FakeDiffProvider(snapshot("src/service.txt", "docs/notes.txt")),
    )

    result = reviewer.run("Revisar archivos", state)

    assert result.decision == "changes_requested"
    issue = next(issue for issue in result.issues if issue.code == "unrelated_files")
    assert issue.evidence == ("docs/notes.txt",)


def test_preserves_llm_correction_request_when_supported_by_diff() -> None:
    state = ready_state("src/service.txt")
    issue = {
        "code": "requirement_gap",
        "message": "El diff no cubre una condición del pedido.",
        "severity": "medium",
        "evidence": ["diff:src/service.txt"],
    }
    llm = FakeReviewerLLM(
        decision="changes_requested",
        summary="La implementación está incompleta.",
        issues=[issue],
        severity="medium",
        required_changes=["Cubrir la condición faltante."],
        confidence=0.8,
    )
    reviewer = ReviewerAgent(
        llm_client=llm,
        diff_provider=FakeDiffProvider(snapshot("src/service.txt")),
    )

    result = reviewer.run("Comparar con el pedido", state)

    assert result.decision == "changes_requested"
    assert result.severity == "medium"
    assert result.required_changes == ("Cubrir la condición faltante.",)
    assert result.issues[0].code == "requirement_gap"

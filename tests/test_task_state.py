"""Tests unitarios del estado compartido de una tarea."""

import json

import pytest

from core.models import EvidenceAssessment
from core.task_state import (
    ErrorRecord,
    SourceReference,
    SubagentResult,
    TaskState,
    ToolExecutionRecord,
)


def test_records_and_serializes_current_evidence_assessment() -> None:
    state = TaskState.create("Cambiar", task_id="evidence-task")
    state.propose_plan("1. Cambiar")
    state.approve_plan()
    assessment = EvidenceAssessment(
        "sufficient", ("src/app.py",), (), ("Riesgo bajo",), "proceed", 0.9
    )

    state.record_evidence_assessment(assessment)
    restored = TaskState.from_json(state.to_json())

    assert restored.evidence_assessment == assessment
    assert restored.has_current_sufficient_evidence is True


def test_new_plan_invalidates_previous_evidence_assessment() -> None:
    state = TaskState.create("Cambiar")
    state.propose_plan("Plan inicial")
    state.approve_plan()
    state.record_evidence_assessment(
        EvidenceAssessment("sufficient", ("src/app.py",), (), (), "proceed", 1.0)
    )

    state.propose_plan("Plan nuevo")

    assert state.evidence_assessment is None
    assert state.has_current_sufficient_evidence is False


def test_creates_task_with_defaults_and_generated_id() -> None:
    state = TaskState.create("Analizar el repositorio")

    assert state.task_id
    assert state.original_request == "Analizar el repositorio"
    assert state.current_status == "pending"
    assert state.current_phase == "intake"
    assert state.subagent_results == ()
    assert state.sources == ()
    assert state.final_result is None


def test_adds_subagent_result_through_controlled_method() -> None:
    state = TaskState.create("Delegar análisis", task_id="task-1")
    result = SubagentResult(
        subagent_id="researcher-1",
        task="Revisar seguridad",
        status="completed",
        result="La política confina las rutas.",
    )

    state.add_subagent_result(result)

    assert state.subagent_results == (result,)
    with pytest.raises(AttributeError):
        state.subagent_results.append(result)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "origin", ["repository", "project_memory", "rag", "web", "inference"]
)
def test_records_every_supported_source_origin(origin: str) -> None:
    state = TaskState.create("Consultar fuentes")
    source = SourceReference(
        origin=origin,  # type: ignore[arg-type]
        reference="docs/architecture.md",
        summary="Arquitectura vigente",
    )

    state.add_source(source)

    assert state.sources[-1] == source


def test_rejects_unknown_source_origin() -> None:
    with pytest.raises(ValueError, match="Origen de fuente inválido"):
        SourceReference(
            origin="unknown",  # type: ignore[arg-type]
            reference="dato sin procedencia",
        )


def test_records_modified_files_without_duplicates() -> None:
    state = TaskState.create("Modificar un archivo")

    state.record_file_modified("core/task_state.py")
    state.record_file_modified("core/task_state.py")
    state.record_file_modified("tests/test_task_state.py")

    assert state.files_modified == (
        "core/task_state.py",
        "tests/test_task_state.py",
    )


def test_serializes_and_deserializes_without_data_loss() -> None:
    state = TaskState.create("Implementar estado", task_id="task-json")
    state.set_status("running")
    state.set_phase("execution")
    state.propose_plan("1. Diseñar\n2. Probar")
    state.approve_plan()
    state.add_subagent_result(
        SubagentResult("reviewer", "Revisar diseño", "completed", "Aprobado")
    )
    state.add_source(
        SourceReference("repository", "core/models.py", "Modelos existentes")
    )
    state.add_source(SourceReference("inference", "Análisis del diseño"))
    state.add_repository_finding("El historial actual vive en memoria.")
    state.record_file_read("core/models.py")
    state.record_file_modified("core/task_state.py")
    state.record_command("python -m pytest tests")
    state.record_tool_call(
        ToolExecutionRecord(
            tool_call_id="call-1",
            tool_name="read_file",
            arguments={"path": "core/models.py"},
            success=True,
            result={"lines": 70},
        )
    )
    state.record_error(
        ErrorRecord(
            message="Fallo recuperable",
            phase="execution",
            component="reader",
            recoverable=True,
        )
    )
    state.add_warning("El estado todavía no está conectado a la CLI.")
    state.add_observation("Las colecciones se exponen como tuplas.")
    state.set_final_result("Modelo implementado y probado.")

    payload = state.to_json(indent=2)
    restored = TaskState.from_json(payload)

    assert json.loads(payload) == state.to_dict()
    assert restored.to_dict() == state.to_dict()
    assert restored == state


def test_rejects_non_serializable_tool_result() -> None:
    state = TaskState.create("Validar JSON")
    state.record_tool_call(
        ToolExecutionRecord("call-1", "demo", {}, True, result=object())
    )

    with pytest.raises(ValueError, match="no serializables"):
        state.to_json()

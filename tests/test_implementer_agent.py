"""Tests de Implementer sobre proyectos temporales de tecnologías diferentes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agents.base import AgentExecutionError
from agents.implementer import (
    ImplementerAgent,
    ImplementerBlockedError,
    ScopedWritePolicy,
)
from core.models import EvidenceAssessment, LLMResponse, LLMUsage, Message
from core.task_state import SourceReference, SubagentResult, TaskState


class FakeImplementerLLM:
    def __init__(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
    ) -> None:
        self.path = path
        self.old_text = old_text
        self.new_text = new_text
        self.messages: list[Message] = []
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.calls += 1
        self.messages = list(messages)
        assert tools == ()
        payload = {
            "summary": "Cambio localizado preparado.",
            "proposed_change": "Actualizar solamente el comportamiento solicitado.",
            "conventions_check": ["Mantiene nombres y estructura observados."],
            "changes": [
                {
                    "path": self.path,
                    "old_text": self.old_text,
                    "new_text": self.new_text,
                    "explanation": "Ajusta el valor sin alterar código no relacionado.",
                }
            ],
            "findings": [],
            "recommendations": [],
            "sources": [],
            "files_relevant": [self.path],
            "blockers": [],
            "confidence": 0.9,
        }
        text = json.dumps(payload)
        return LLMResponse(
            assistant_message=Message("assistant", text),
            text=text,
            tool_calls=[],
            model="fake-implementer",
            usage=LLMUsage(1, 1, 2),
            latency_ms=1.0,
        )


def ready_state(
    *relevant_files: str,
    with_research: bool = True,
    with_assessment: bool = True,
) -> TaskState:
    state = TaskState.create("Cambiar el saludo", task_id="implementation-task")
    state.propose_plan("1. Leer el archivo relevante.\n2. Cambiar sólo el saludo.")
    state.approve_plan()
    state.add_repository_finding(
        "convention=mantener nombres existentes; evidencia: archivo fuente."
    )
    state.add_repository_finding("impact=el cambio queda limitado al archivo seleccionado.")
    state.add_subagent_result(
        SubagentResult(
            "explorer",
            "Explorar",
            "completed",
            summary="Archivos identificados",
            files_relevant=tuple(relevant_files),
            confidence=0.9,
        )
    )
    if with_research:
        evidence = SourceReference("rag", "docs/greeting", "Contrato del saludo")
        state.add_source(evidence)
        state.add_subagent_result(
            SubagentResult(
                "researcher",
                "Investigar",
                "completed",
                summary="Evidencia suficiente",
                sources=(evidence,),
                confidence=0.9,
            )
        )
    if with_assessment:
        state.record_evidence_assessment(
            EvidenceAssessment("sufficient", ("docs/greeting",), (), (), "proceed", 1.0)
        )
    return state


@pytest.fixture()
def python_project(tmp_path: Path) -> Path:
    root = tmp_path / "python-project"
    root.mkdir()
    (root / "app.py").write_text(
        'GREETING = "hello"\n\ndef greet():\n    return GREETING\n', encoding="utf-8"
    )
    (root / "other.py").write_text("UNCHANGED = True\n", encoding="utf-8")
    return root


@pytest.fixture()
def javascript_project(tmp_path: Path) -> Path:
    root = tmp_path / "javascript-project"
    root.mkdir()
    (root / "index.js").write_text(
        'const greeting = "hello";\nmodule.exports = greeting;\n', encoding="utf-8"
    )
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    return root


def test_propose_only_does_not_write(python_project: Path) -> None:
    original = (python_project / "app.py").read_text(encoding="utf-8")
    llm = FakeImplementerLLM(
        path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
    )
    state = ready_state("app.py")
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    result = agent.run("Cambiar el saludo", state, mode="propose_only")

    assert result.mode == "propose_only"
    assert result.files_modified == ()
    assert state.files_modified == ()
    assert (python_project / "app.py").read_text(encoding="utf-8") == original
    assert result.changes[0].path == "app.py"


def test_applies_allowed_localized_change_in_javascript_project(
    javascript_project: Path,
) -> None:
    llm = FakeImplementerLLM(
        path="index.js",
        old_text='const greeting = "hello";',
        new_text='const greeting = "hola";',
    )
    state = ready_state("index.js")
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(javascript_project)
    )

    result = agent.run("Cambiar el saludo", state, mode="apply_changes")

    assert result.files_modified == ("index.js",)
    assert state.files_modified == ("index.js",)
    assert (javascript_project / "index.js").read_text(encoding="utf-8") == (
        'const greeting = "hola";\nmodule.exports = greeting;\n'
    )
    assert state.commands_executed == ()


def test_rejects_lock_file_without_explicit_policy_authorization(
    javascript_project: Path,
) -> None:
    llm = FakeImplementerLLM(
        path="package-lock.json",
        old_text='"lockfileVersion": 3',
        new_text='"lockfileVersion": 4',
    )
    state = ready_state("package-lock.json")
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(javascript_project)
    )

    with pytest.raises(AgentExecutionError, match="lock file requiere autorización"):
        agent.run("Actualizar lock", state, mode="apply_changes")

    assert llm.calls == 0
    assert state.files_modified == ()
    assert '"lockfileVersion": 3' in (
        javascript_project / "package-lock.json"
    ).read_text(encoding="utf-8")


def test_blocks_without_approved_plan(python_project: Path) -> None:
    state = ready_state("app.py")
    state.approved_plan = None
    llm = FakeImplementerLLM(
        path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
    )
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    with pytest.raises(ImplementerBlockedError, match="plan aprobado"):
        agent.run("Cambiar saludo", state, mode="apply_changes")

    assert llm.calls == 0
    assert state.files_modified == ()


def test_blocks_without_sufficient_research_evidence(python_project: Path) -> None:
    state = ready_state("app.py", with_research=False)
    llm = FakeImplementerLLM(
        path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
    )
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    with pytest.raises(ImplementerBlockedError, match="evidencia técnica suficiente"):
        agent.run("Cambiar saludo", state)

    assert llm.calls == 0


def test_apply_changes_rejects_without_current_sufficient_assessment(
    python_project: Path,
) -> None:
    state = ready_state("app.py", with_assessment=False)
    llm = FakeImplementerLLM(
        path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
    )
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    with pytest.raises(ImplementerBlockedError, match="EvidenceAssessment sufficient vigente"):
        agent.run("Cambiar saludo", state, mode="apply_changes")

    assert llm.calls == 0
    assert state.files_modified == ()


def test_propose_only_remains_available_without_evidence_assessment(
    python_project: Path,
) -> None:
    state = ready_state("app.py", with_assessment=False)
    llm = FakeImplementerLLM(
        path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
    )
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    result = agent.run("Cambiar saludo", state, mode="propose_only")

    assert result.files_modified == ()
    assert llm.calls == 1


def test_evidence_gate_is_sufficient_for_supported_existing_target(
    python_project: Path,
) -> None:
    state = ready_state("app.py", with_assessment=False)
    agent = ImplementerAgent(
        llm_client=FakeImplementerLLM(
            path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
        ),
        write_policy=ScopedWritePolicy(python_project),
    )

    assessment = agent.assess_evidence("Cambiar", state, validation_available=True)

    assert assessment.status == "sufficient"


@pytest.mark.parametrize(
    ("condition", "expected_text"),
    [
        ("ambiguous", "Ambigüedad"),
        ("contradictory", "contradictorias"),
        ("no_validation", "validación"),
        ("excessive_risk", "excesivo"),
    ],
)
def test_evidence_gate_blocks_detected_conditions_without_writing(
    python_project: Path, condition: str, expected_text: str
) -> None:
    state = ready_state("app.py", with_assessment=False)
    if condition == "ambiguous":
        state.add_warning("Ambigüedad: no se identifica el contrato a conservar.")
    elif condition == "contradictory":
        state.add_warning("Fuentes contradictorias describen resultados distintos.")
    elif condition == "excessive_risk":
        state.add_warning("Riesgo excesivo de pérdida de datos.")
    agent = ImplementerAgent(
        llm_client=FakeImplementerLLM(
            path="app.py", old_text='GREETING = "hello"', new_text='GREETING = "hola"'
        ),
        write_policy=ScopedWritePolicy(python_project),
    )

    assessment = agent.assess_evidence(
        "Cambiar", state, validation_available=condition != "no_validation"
    )

    assert assessment.status == "insufficient"
    assert any(expected_text.casefold() in risk.casefold() for risk in assessment.risks)
    assert state.files_modified == ()


def test_evidence_gate_blocks_nonexistent_target_without_writing(
    python_project: Path,
) -> None:
    state = ready_state("missing.py", with_assessment=False)
    agent = ImplementerAgent(
        llm_client=FakeImplementerLLM(path="missing.py", old_text="old", new_text="new"),
        write_policy=ScopedWritePolicy(python_project),
    )

    assessment = agent.assess_evidence("Cambiar", state, validation_available=True)

    assert assessment.status == "insufficient"
    assert any("no existe" in risk for risk in assessment.risks)
    assert state.files_modified == ()


def test_evidence_gate_blocks_denied_write_policy_without_writing(
    python_project: Path,
) -> None:
    protected = python_project / "secret.txt"
    protected.write_text("value", encoding="utf-8")
    state = ready_state("secret.txt", with_assessment=False)
    agent = ImplementerAgent(
        llm_client=FakeImplementerLLM(
            path="secret.txt", old_text="value", new_text="changed"
        ),
        write_policy=ScopedWritePolicy(python_project),
    )

    assessment = agent.assess_evidence("Cambiar", state, validation_available=True)

    assert assessment.status == "insufficient"
    assert "permisos de modificación" in assessment.missing_information
    assert protected.read_text(encoding="utf-8") == "value"


def test_rejects_change_outside_explorer_relevant_files(
    python_project: Path,
) -> None:
    llm = FakeImplementerLLM(
        path="other.py", old_text="UNCHANGED = True", new_text="UNCHANGED = False"
    )
    state = ready_state("app.py")
    agent = ImplementerAgent(
        llm_client=llm, write_policy=ScopedWritePolicy(python_project)
    )

    with pytest.raises(AgentExecutionError, match="fuera del alcance"):
        agent.run("Cambiar saludo", state, mode="apply_changes")

    assert state.files_modified == ()
    assert (python_project / "other.py").read_text(encoding="utf-8") == "UNCHANGED = True\n"

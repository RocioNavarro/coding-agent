"""Selección acotada de contexto para tareas y proyectos extensos."""

from agents.base import AgentContext
from agents.context_manager import StateContextManager
from core.task_state import ErrorRecord, SourceReference, SubagentResult, TaskState


def populated_long_task() -> TaskState:
    state = TaskState.create(
        "Corregir el timeout del servicio de pagos sin modificar autenticación",
        task_id="long-context",
    )
    state.set_phase("implementation")
    state.propose_plan("1. Revisar pagos. 2. Ajustar timeout. 3. Ejecutar tests.")
    state.approve_plan()
    state.add_repository_finding("architecture=servicios y adaptadores; evidencia: docs")
    state.add_repository_finding("language=Python; evidencia: pyproject.toml")
    state.add_repository_finding("framework=FastAPI; evidencia: pyproject.toml")
    state.add_repository_finding("decision=mantener el cliente HTTP actual")
    state.add_repository_finding("architecture=frontend no relacionado")
    for index in range(12):
        relevant = index in {9, 10, 11}
        state.add_subagent_result(
            SubagentResult(
                "researcher" if relevant else "explorer",
                "Investigar timeout de pagos" if relevant else "Inventariar repositorio",
                "completed",
                summary=(
                    f"Resultado {index} sobre timeout y pagos"
                    if relevant
                    else f"Output extenso no relacionado {index} " + "x" * 300
                ),
                findings=("El timeout actual es insuficiente.",) if relevant else (),
                files_relevant=("src/payments.py",) if relevant else (f"vendor/file{index}.txt",),
            )
        )
    state.add_source(SourceReference("rag", "docs/payments.md", "Timeout de pagos"))
    state.add_source(SourceReference("web", "https://irrelevant.test", "Tema distinto"))
    state.add_source(SourceReference("inference", "analysis:timeout", "Podría haber latencia"))
    state.record_error(ErrorRecord("TimeoutError en pagos", "implementation", "tester", True))
    state.record_error(ErrorRecord("Error histórico de lint", "exploration", "explorer", True))
    return state


def test_selects_only_relevant_bounded_context_for_long_task() -> None:
    state = populated_long_task()
    manager = StateContextManager(max_context_chars=1800, max_item_chars=180)

    context = manager.select("Ajustar timeout de pagos", state)
    serialized = str(context.to_dict())

    assert len(serialized) <= 1800
    assert any(item.startswith("Pedido: ") for item in context.facts)
    assert any(item.startswith("Plan: ") for item in context.facts)
    assert "Fase: implementation" in context.facts
    assert any("Arquitectura:" in item for item in context.facts)
    assert any("Tecnología:" in item for item in context.facts)
    assert any("Decisión persistente:" in item for item in context.constraints)
    assert any("TimeoutError" in item for item in context.constraints)
    assert context.files == ("src/payments.py",)
    assert all("vendor/" not in path for path in context.files)
    assert all("irrelevant.test" not in source.reference for source in context.sources)
    assert any(item.startswith("HECHO —") for item in context.facts)
    assert any(item.startswith("INFERENCIA —") for item in context.facts)
    assert "Output extenso no relacionado" not in serialized


def test_progressively_summarizes_older_results_without_repetition() -> None:
    state = populated_long_task()
    manager = StateContextManager(max_context_chars=2400, max_results=2)

    first = manager.select("timeout pagos", state)
    state.add_subagent_result(
        SubagentResult(
            "tester", "Validar timeout", "passed",
            summary="Tests de timeout de pagos aprobados.",
            files_relevant=("tests/test_payments.py",),
        )
    )
    second = manager.select("timeout pagos", state)

    assert any("Resumen progresivo" in item for item in first.facts)
    assert any("Resumen progresivo" in item for item in second.facts)
    assert sum("Tests de timeout" in item for item in second.facts) == 1


def test_keeps_decisions_when_budget_requires_safe_truncation() -> None:
    state = populated_long_task()
    state.add_repository_finding("decision=" + "conservar compatibilidad " * 40)
    manager = StateContextManager(max_context_chars=700, max_item_chars=100)

    context = manager.select("timeout pagos", state)
    serialized = str(context.to_dict())

    assert len(serialized) <= 700
    assert any("Decisión persistente:" in item for item in context.constraints)
    assert "…[truncado]" in serialized


def test_adapts_selection_to_different_project_characteristics() -> None:
    state = TaskState.create("Actualizar renderizado del panel", task_id="frontend")
    state.set_phase("planning")
    state.add_repository_finding("architecture=aplicación web por componentes")
    state.add_repository_finding("language=TypeScript")
    state.add_repository_finding("framework=framework-configurado")
    state.add_subagent_result(
        SubagentResult(
            "explorer", "Explorar panel", "completed",
            summary="Panel localizado.", files_relevant=("ui/panel.ts", "api/server.py"),
        )
    )

    context = StateContextManager().select("renderizado panel ui", state)

    assert any("TypeScript" in item for item in context.facts)
    assert any("framework-configurado" in item for item in context.facts)
    assert context.files == ("ui/panel.ts",)


def test_requested_context_is_filtered_and_deduplicated() -> None:
    state = populated_long_task()
    requested = AgentContext(
        facts=("timeout de pagos confirmado", "timeout de pagos confirmado"),
        sources=(state.sources[0], state.sources[0], state.sources[1]),
        files=("src/payments.py", "secrets.env"),
        constraints=("No cambiar autenticación", "No cambiar autenticación"),
    )

    context = StateContextManager().select("timeout pagos", state, requested)

    assert context.files == ("src/payments.py",)
    assert sum("timeout de pagos confirmado" in item for item in context.facts) == 1
    assert context.constraints.count("Restricción: No cambiar autenticación") == 1

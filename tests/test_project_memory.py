"""Persistencia, aislamiento, recuperación y seguridad de ProjectMemory."""

import json
from pathlib import Path

import pytest

from agents.project_memory import MemoryCorruptionError, ProjectMemory
from core.task_state import ErrorRecord, TaskState


def test_persists_and_loads_all_core_information(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    storage = tmp_path / "memory"
    memory = ProjectMemory(workspace, storage_root=storage)
    memory.update_project_summary("Servicio de procesamiento")
    memory.update_architecture("Módulos separados por responsabilidad")
    memory.add_technology("Runtime detectado")
    memory.add_module("src")
    memory.add_important_file("src/main.ext")
    memory.add_dependency("library-x")
    memory.add_known_command("tool test", "project.config")
    memory.add_convention("tests separados")
    memory.add_decision("Mantener la API existente")
    memory.add_bug("Falla de borde", "Validar entrada")
    memory.save()

    loaded = ProjectMemory(workspace, storage_root=storage).load()

    assert loaded.data["project_summary"] == "Servicio de procesamiento"
    assert loaded.data["architecture"] == "Módulos separados por responsabilidad"
    assert loaded.data["technologies"] == ["Runtime detectado"]
    assert loaded.data["known_commands"][0]["evidence"] == "project.config"
    assert loaded.data["bugs"][0]["resolution"] == "Validar entrada"
    assert loaded.path.stat().st_mode & 0o777 == 0o600


def test_isolates_workspaces_in_shared_storage(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    storage = tmp_path / "shared-memory"
    first_memory = ProjectMemory(first, storage_root=storage)
    second_memory = ProjectMemory(second, storage_root=storage)

    first_memory.update_project_summary("Primer proyecto")
    first_memory.save()
    second_memory.update_project_summary("Segundo proyecto")
    second_memory.save()

    assert first_memory.workspace_id != second_memory.workspace_id
    assert first_memory.path != second_memory.path
    assert ProjectMemory(first, storage_root=storage).load().data["project_summary"] == "Primer proyecto"
    assert ProjectMemory(second, storage_root=storage).load().data["project_summary"] == "Segundo proyecto"


def test_updates_task_history_without_duplicating_task_id(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    memory = ProjectMemory(workspace, identifier="configured-project", storage_root=tmp_path / "memory")
    state = TaskState.create("Corregir comportamiento", task_id="task-1")
    state.set_status("completed")
    state.record_file_modified("src/file.ext")
    state.record_error(ErrorRecord("Error conocido", "testing"))
    state.set_final_result("Cambio validado")

    memory.save_task_summary(state)
    state.set_final_result("Cambio validado nuevamente")
    memory.save_task_summary(state)
    memory.save()

    data = ProjectMemory(
        workspace, identifier="configured-project", storage_root=tmp_path / "memory"
    ).load().data
    assert len(data["previous_tasks"]) == 1
    assert data["previous_tasks"][0]["result"] == "Cambio validado nuevamente"
    assert data["modified_files"] == ["src/file.ext"]
    assert data["frequent_errors"] == ["Error conocido"]
    assert len(data["session_summaries"]) == 1


def test_recovers_relevant_memory_as_traceable_research_fragments(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    memory = ProjectMemory(workspace, storage_root=tmp_path / "memory")
    memory.update_architecture("El módulo billing procesa facturas")
    memory.add_decision("Billing conserva identificadores estables")
    memory.add_known_command("check billing", "project.config")
    memory.save()

    results = ProjectMemory(workspace, storage_root=tmp_path / "memory").search(
        "validar billing", limit=2
    )

    assert len(results) == 2
    assert all(result.origin == "project_memory" for result in results)
    assert all(result.reference.startswith(f"memory://{memory.workspace_id}/") for result in results)
    assert any("billing" in result.content.casefold() for result in results)


def test_missing_file_is_empty_but_corrupt_file_is_reported(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    memory = ProjectMemory(workspace, storage_root=tmp_path / "memory").load()
    assert memory.data["technologies"] == []

    memory.storage_root.mkdir(parents=True)
    memory.path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(MemoryCorruptionError, match="corrupta"):
        memory.load()


def test_redacts_secrets_and_ignores_sensitive_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    memory = ProjectMemory(workspace, storage_root=tmp_path / "memory")
    memory.add_decision("Usar api_key=super-secret-value para el proveedor")
    memory.add_bug("Falló con token: ghp_abcdefghijklmnopqrstuvwxyz")
    memory.add_important_file(".env")
    state = TaskState.create("password=hunter2", task_id="secret-task")
    state.set_final_result("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    memory.save_task_summary(state)
    memory.save()

    serialized = memory.path.read_text(encoding="utf-8")
    parsed = json.loads(serialized)

    assert "super-secret-value" not in serialized
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "hunter2" not in serialized
    assert "abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized
    assert parsed["important_files"] == []

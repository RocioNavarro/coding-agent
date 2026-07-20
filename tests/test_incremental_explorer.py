"""Exploración incremental respaldada por memoria persistida."""

from pathlib import Path

from agents.explorer import ExplorerAgent
from agents.project_memory import ProjectMemory
from core.task_state import TaskState
from tests.integration.fakes import FakeObservability, ScriptedLLM


def explorer_payload(files: list[str]) -> dict[str, object]:
    return {
        "summary": "Repository structure inspected with concrete evidence.",
        "findings": ["Architecture and impact are bounded by inspected files."],
        "recommendations": ["Keep validation localized."],
        "sources": [
            {"origin": "repository", "reference": path, "summary": "inspected"}
            for path in files
        ],
        "files_relevant": files,
        "blockers": [],
        "confidence": 0.9,
    }


def repository(tmp_path: Path, name: str = "repository") -> Path:
    root = tmp_path / name
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# Component\n", encoding="utf-8")
    (root / "docs/contract.md").write_text("# Contract\nStable.\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src/value.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def run_explorer(root: Path, storage: Path, task_id: str):
    memory = ProjectMemory(root, identifier="incremental", storage_root=storage)
    observed = FakeObservability()
    llm = ScriptedLLM((explorer_payload(["src/value.py"]),))
    agent = ExplorerAgent(
        repository_root=root, llm_client=llm, project_memory=memory,
        observability=observed,
    )
    state = TaskState.create("Inspect value component", task_id=task_id)
    result = agent.run("Inspect value component", state)
    return result, state, memory, observed


def strategy(state: TaskState) -> str:
    return next(
        item for item in state.observations
        if item.startswith("Estrategia de exploración:")
    )


def test_first_full_then_new_instance_incremental_reduces_reads(tmp_path: Path) -> None:
    root = repository(tmp_path)
    storage = tmp_path / "memory"

    _, first_state, first_memory, _ = run_explorer(root, storage, "first")
    _, second_state, second_memory, observed = run_explorer(root, storage, "second")

    assert "full" in strategy(first_state)
    assert "incremental" in strategy(second_state)
    assert len(second_state.files_read) < len(first_state.files_read)
    assert first_memory.path == second_memory.path
    assert second_memory.load().data["last_explored_at"] is not None
    assert any("Archivos evitados:" in item and "docs/contract.md" in item
               for item in second_state.observations)
    event = next(event for event in observed.events if event.name == "explorer-strategy")
    assert event.payload["strategy"] == "incremental"
    assert event.payload["files_avoided"] >= 1


def test_modified_file_is_revalidated(tmp_path: Path) -> None:
    root = repository(tmp_path)
    storage = tmp_path / "memory"
    run_explorer(root, storage, "first")
    (root / "docs/contract.md").write_text(
        "# Contract\nChanged and expanded.\n", encoding="utf-8"
    )

    _, state, _, _ = run_explorer(root, storage, "second")

    assert "docs/contract.md" in state.files_read
    assert any("Archivos modificados desde memoria: docs/contract.md" in item
               for item in state.observations)


def test_new_file_is_detected_and_inspected(tmp_path: Path) -> None:
    root = repository(tmp_path)
    storage = tmp_path / "memory"
    run_explorer(root, storage, "first")
    (root / "docs/new.md").write_text("# New evidence\n", encoding="utf-8")

    _, state, _, _ = run_explorer(root, storage, "second")

    assert "docs/new.md" in state.files_read
    assert any("Archivos nuevos: docs/new.md" in item for item in state.observations)


def test_corrupt_memory_falls_back_to_full_without_breaking_explorer(tmp_path: Path) -> None:
    root = repository(tmp_path)
    storage = tmp_path / "memory"
    _, _, memory, _ = run_explorer(root, storage, "first")
    memory.path.write_text("{invalid", encoding="utf-8")

    result, state, _, observed = run_explorer(root, storage, "second")

    assert result.status == "completed"
    assert "full" in strategy(state)
    event = next(event for event in observed.events if event.name == "explorer-strategy")
    assert event.payload["memory_invalidated"] is True


def test_memory_is_isolated_between_workspaces(tmp_path: Path) -> None:
    storage = tmp_path / "shared-memory"
    first_root = repository(tmp_path, "first-repository")
    second_root = repository(tmp_path, "second-repository")
    _, _, first_memory, _ = run_explorer(first_root, storage, "first")

    _, second_state, second_memory, _ = run_explorer(second_root, storage, "second")

    assert first_memory.workspace_id != second_memory.workspace_id
    assert first_memory.path != second_memory.path
    assert "full" in strategy(second_state)

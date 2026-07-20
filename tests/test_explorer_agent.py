"""Tests de Explorer sobre repositorios mínimos de ecosistemas diferentes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agents.base import AgentExecutionError
from agents.explorer import EXPLORER_ALLOWED_TOOLS, ExplorerAgent
from agents.project_memory import ProjectMemory
from agents.repository_detection import (
    DetectionEvidence,
    RepositoryDetection,
    RepositoryDetector,
    RepositorySnapshot,
)
from core.models import LLMResponse, LLMUsage, Message, ToolCall
from core.task_state import TaskState
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry


class FakeExplorerLLM:
    def __init__(self, tool_calls: list[ToolCall] | None = None) -> None:
        self.tool_calls = tool_calls or []
        self.messages: list[Message] = []
        self.schemas: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.messages = list(messages)
        self.schemas = list(tools)
        payload = {
            "summary": "Repositorio analizado con evidencia local.",
            "findings": ["La arquitectura fue obtenida del inventario."],
            "recommendations": ["Ejecutar los comandos detectados antes de cambiar."],
            "sources": [
                {"origin": "repository", "reference": "README.md", "summary": "Documentación"}
            ],
            "files_relevant": ["README.md"],
            "blockers": [],
            "confidence": 0.9,
        }
        text = json.dumps(payload)
        return LLMResponse(
            assistant_message=Message("assistant", text, self.tool_calls),
            text=text,
            tool_calls=self.tool_calls,
            model="fake",
            usage=LLMUsage(1, 1, 2),
            latency_ms=1.0,
        )


def write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def python_repository(tmp_path: Path) -> Path:
    root = tmp_path / "python-project"
    write(
        root / "pyproject.toml",
        "[build-system]\nrequires = ['setuptools']\n"
        "[project]\ndependencies = ['fastapi', 'pytest', 'ruff']\n",
    )
    write(root / "src/service/app.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    write(root / "tests/test_app.py", "def test_app(): pass\n")
    write(root / "README.md", "# Service\nRun tests before changes.\n")
    write(root / ".github/workflows/ci.yml", "steps:\n  - run: python -m pytest\n")
    write(root / "scripts/check.sh", "python -m pytest\n")
    write(root / "build/generated.py", "ignored = True\n")
    return root


@pytest.fixture()
def javascript_repository(tmp_path: Path) -> Path:
    root = tmp_path / "javascript-project"
    write(
        root / "package.json",
        json.dumps(
            {
                "scripts": {"test": "jest", "lint": "eslint .", "start": "node index.js"},
                "dependencies": {"express": "^4.0.0"},
                "devDependencies": {"jest": "^29.0.0", "eslint": "^9.0.0"},
            }
        ),
    )
    write(root / "src/index.js", "const express = require('express')\n")
    write(root / "src/server.test.js", "test('server', () => {})\n")
    write(root / "README.md", "# API\n")
    return root


@pytest.fixture()
def maven_repository(tmp_path: Path) -> Path:
    root = tmp_path / "maven-project"
    write(
        root / "pom.xml",
        "<project><dependencies><dependency>"
        "<groupId>org.springframework.boot</groupId>"
        "<artifactId>spring-boot-starter-web</artifactId>"
        "</dependency><dependency><groupId>org.junit.jupiter</groupId>"
        "<artifactId>junit-jupiter</artifactId></dependency>"
        "</dependencies></project>",
    )
    write(root / "src/main/java/com/example/Main.java", "class Main {}\n")
    write(root / "src/test/java/com/example/MainTest.java", "class MainTest {}\n")
    write(root / "README.md", "# Java service\n")
    return root


def detection_names(report: object, category: str) -> set[str]:
    return {
        item.name
        for item in report.detections  # type: ignore[attr-defined]
        if item.category == category
    }


def test_detects_python_structure_technologies_and_commands(
    python_repository: Path,
) -> None:
    explorer = ExplorerAgent(repository_root=python_repository, llm_client=FakeExplorerLLM())

    report = explorer.explore("Modificar el endpoint de app")

    assert "Python" in detection_names(report, "language")
    assert "Python packaging" in detection_names(report, "build_system")
    assert {"FastAPI"} <= detection_names(report, "framework")
    assert {"pytest"} <= detection_names(report, "test_framework")
    assert {"python -m pytest", "ruff check .", "python -m build"} <= set(report.commands)
    assert "src/service/app.py" in report.inventory.source_files
    assert "tests/test_app.py" in report.inventory.test_files
    assert ".github/workflows/ci.yml" in report.inventory.ci_files
    assert "scripts/check.sh" in report.inventory.script_files
    assert "src/service/app.py" in report.inventory.entry_points
    assert "build/generated.py" not in report.inventory.files
    assert "src/service/app.py" in report.relevant_files


def test_records_detected_repository_information_in_project_memory(
    python_repository: Path, tmp_path: Path
) -> None:
    memory = ProjectMemory(
        python_repository, storage_root=tmp_path / "persistent-memory"
    )
    explorer = ExplorerAgent(
        repository_root=python_repository,
        llm_client=FakeExplorerLLM(),
        project_memory=memory,
    )

    explorer.run(
        "Modificar el endpoint de app",
        TaskState.create("Modificar el endpoint de app"),
    )
    persisted = ProjectMemory(
        python_repository, storage_root=tmp_path / "persistent-memory"
    ).load().data

    assert {"Python", "Python packaging", "FastAPI", "pytest", "Ruff"} <= set(
        persisted["technologies"]
    )
    assert "fastapi" in {dependency.casefold() for dependency in persisted["dependencies"]}
    assert any(item["command"] == "python -m pytest" for item in persisted["known_commands"])
    assert "src/service/app.py" in persisted["important_files"]


def test_detects_javascript_framework_tools_and_package_scripts(
    javascript_repository: Path,
) -> None:
    report = ExplorerAgent(
        repository_root=javascript_repository, llm_client=FakeExplorerLLM()
    ).explore("Revisar server")

    assert "JavaScript" in detection_names(report, "language")
    assert "npm" in detection_names(report, "build_system")
    assert "Express" in detection_names(report, "framework")
    assert "Jest" in detection_names(report, "test_framework")
    assert "ESLint" in detection_names(report, "tool")
    assert {"npm run test", "npm run lint", "npm run start"} <= set(report.commands)
    assert "src/server.test.js" in report.inventory.test_files


def test_detects_maven_as_a_different_build_system(maven_repository: Path) -> None:
    report = ExplorerAgent(
        repository_root=maven_repository, llm_client=FakeExplorerLLM()
    ).explore("Entender Main")

    assert "Java" in detection_names(report, "language")
    assert "Maven" in detection_names(report, "build_system")
    assert "Spring" in detection_names(report, "framework")
    assert "JUnit" in detection_names(report, "test_framework")
    assert {"mvn test", "mvn package"} <= set(report.commands)
    spring = next(item for item in report.detections if item.name == "Spring")
    assert {evidence.path for evidence in spring.evidence} == {"pom.xml"}
    assert "src/main/java/com/example/Main.java" in report.inventory.entry_points


def test_gradle_settings_are_authoritative_for_modules(tmp_path: Path) -> None:
    root = tmp_path / "gradle-project"
    write(
        root / "settings.gradle.kts",
        'rootProject.name = "Demo"\ninclude("api", "core", "runner")\ninclude("runner")\n',
    )
    write(root / "api/build.gradle", "plugins {}")
    write(root / "core/build.gradle", "plugins {}")
    write(root / "runner/build.gradle", "plugins {}")
    write(root / "buildSrc/build.gradle.kts", "plugins {}")
    write(root / "docs/README.md", "# Docs")

    report = ExplorerAgent(
        repository_root=root, llm_client=FakeExplorerLLM()
    ).explore("Arquitectura")

    assert report.declared_modules == ("api", "core", "runner")
    assert report.build_infrastructure == ("buildSrc",)
    assert "buildSrc" not in report.declared_modules
    assert report.module_warnings == (
        "El módulo 'runner' aparece repetido en settings.gradle.kts.",
    )


def registry_with_write() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "write_file",
            "Escribe",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            lambda path: path,
            True,
        )
    )
    return registry


def test_rejects_write_tool(python_repository: Path) -> None:
    call = ToolCall("write", "write_file", {"path": "src/service/app.py"})
    explorer = ExplorerAgent(
        repository_root=python_repository, llm_client=FakeExplorerLLM([call])
    )
    state = TaskState.create("Sólo explorar", task_id="denied")

    with pytest.raises(AgentExecutionError, match="no está permitida"):
        explorer.run("Revisar app", state, available_tools=registry_with_write())

    assert "write_file" not in EXPLORER_ALLOWED_TOOLS
    assert state.subagent_results == ()
    assert state.files_modified == ()


def test_registers_evidence_and_result_in_task_state(python_repository: Path) -> None:
    llm = FakeExplorerLLM()
    explorer = ExplorerAgent(repository_root=python_repository, llm_client=llm)
    state = TaskState.create("Entender app", task_id="explorer-state")

    result = explorer.run("Encontrar app y tests", state)

    assert state.subagent_results == (result,)
    assert any("language=Python" in finding for finding in state.repository_findings)
    assert any(source.reference == "pyproject.toml" for source in state.sources)
    assert "pyproject.toml" in state.files_read
    assert any("python -m pytest" in item for item in state.observations)
    assert state.files_modified == ()
    assert {schema["function"]["name"] for schema in llm.schemas} == {
        "find_files", "list_files", "read_file", "search_text"
    }
    sent = json.loads(llm.messages[1].content)
    assert "original_request" not in sent
    assert any("Evidencia:" in fact for fact in sent["context"]["facts"])


def test_accepts_an_extensible_custom_detector(python_repository: Path) -> None:
    class LicenseDetector(RepositoryDetector):
        def detect(
            self, snapshot: RepositorySnapshot
        ) -> tuple[RepositoryDetection, ...]:
            paths = snapshot.has_file("README.md")
            return (
                RepositoryDetection(
                    "custom",
                    "documented-project",
                    tuple(DetectionEvidence(path, "documentación raíz") for path in paths),
                ),
            )

    report = ExplorerAgent(
        repository_root=python_repository,
        llm_client=FakeExplorerLLM(),
        detectors=(LicenseDetector(),),
    ).explore("Inspeccionar documentación")

    assert detection_names(report, "custom") == {"documented-project"}
    assert report.detections[0].evidence[0].path == "README.md"

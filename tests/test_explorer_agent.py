"""Tests de Explorer sobre repositorios mínimos de ecosistemas diferentes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agents.base import AgentExecutionError
from agents.analysis_summary import DeterministicAnalysisSummary
from agents.explorer import EXPLORER_ALLOWED_TOOLS, ExplorerAgent
from agents.project_memory import ProjectMemory
from agents.repository_detection import (
    DetectionEvidence,
    RepositoryDetection,
    RepositoryDetector,
    RepositorySnapshot,
)
from core.models import LLMResponse, LLMUsage, Message, ToolCall
from core.task_state import SourceReference, SubagentResult, TaskState
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


def test_incremental_scan_reloads_real_gradle_and_functional_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "printscript-like"
    storage = tmp_path / "memory"
    write(root / "settings.gradle.kts", '''
        pluginManagement { plugins { kotlin("jvm") version "2.1.10" } }
        plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0" }
        include(
            "linter", "cli", "formatter", "interpreter", "lexer", "parser",
            "token", "common", "runner"
        )
        include("runner")
    ''')
    write(root / "build.gradle.kts", '''
        tasks.register("installGitHooks") {
            file("gradle/scripts/pre-commit").writeText("./gradlew spotlessApply\\n./gradlew test")
            Files.copy(source, file(".git/hooks/pre-commit"))
        }
    ''')
    write(root / "buildSrc/build.gradle.kts", '''
        dependencies {
            implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:2.1.10")
            implementation("com.diffplug.spotless:spotless-plugin-gradle:6.25.0")
        }
        java { toolchain { languageVersion.set(JavaLanguageVersion.of(21)) } }
    ''')
    write(root / "buildSrc/src/main/kotlin/org.printscript.conventions.gradle.kts", '''
        tasks.named("check") { dependsOn("spotlessCheck", "jacocoTestReport") }
    ''')
    dependencies = {
        "cli": ("lexer", "parser", "interpreter", "formatter", "linter", "common", "token"),
        "runner": ("common", "token", "lexer", "parser", "interpreter"),
        "interpreter": ("parser", "common", "token", "lexer"),
        "parser": ("token", "common"), "linter": ("parser", "common", "lexer", "token"),
        "formatter": ("parser", "common", "token"), "lexer": ("token",),
    }
    versions = {"cli": "1.5-SNAPSHOT", "formatter": "3.2-SNAPSHOT"}
    for module in ("linter", "cli", "formatter", "interpreter", "lexer", "parser", "token", "common", "runner"):
        deps = "\n".join(f"implementation project(':{item}')" for item in dependencies.get(module, ()))
        external = {
            "cli": "implementation 'info.picocli:picocli:4.7.6'\ntestImplementation 'org.junit.jupiter:junit-jupiter:5.10.3'",
            "formatter": "implementation 'com.google.code.gson:gson:2.11.0'",
            "linter": "implementation 'com.google.code.gson:gson:2.11.0'",
        }.get(module, "")
        test_block = 'test { minHeapSize = "64m"\nmaxHeapSize = "128m"\n}' if module in {"runner", "interpreter"} else ""
        write(root / module / "build.gradle", f"version = '{versions.get(module, '1.0-SNAPSHOT')}'\n{deps}\n{external}\ntestImplementation 'org.jetbrains.kotlin:kotlin-test'\n{test_block}")
    functional = {
        "common/src/main/kotlin/org/printscript/common/Position.kt": "class Position",
        "token/src/main/kotlin/token/Token.kt": "class Token",
        "lexer/src/main/kotlin/org/printscript/lexer/Lexer.kt": "class Lexer",
        "parser/src/main/kotlin/org/printscript/parser/DefaultParser.kt": "class DefaultParser",
        "interpreter/src/main/kotlin/org/printscript/interpreter/Interpreter.kt": "class Interpreter",
        "formatter/src/main/kotlin/org/printscript/formatter/CodeFormatter.kt": "class CodeFormatter",
        "linter/src/main/kotlin/org/printscript/linter/Linter.kt": "class Linter",
        "runner/src/main/kotlin/org/printscript/runner/Runner.kt": "InputStream Lexer Parser Interpreter",
        "cli/src/main/kotlin/org/printscript/cli/Main.kt": "fun main()",
        "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt": "FrontendAdapter InterpreterAdapter",
        "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt": "class AnalyzeCmd",
        "cli/src/main/kotlin/org/printscript/cli/adapters/FrontendAdapter.kt": "Lexer DefaultParser",
        "cli/src/main/kotlin/org/printscript/cli/adapters/InterpreterAdapter.kt": "Interpreter",
    }
    for path, content in functional.items():
        write(root / path, content)
    write(root / "README.md", "./gradlew build\n./gradlew test\n./gradlew check")
    write(root / ".github/workflows/ci.yml", "./gradlew spotlessCheck\n./gradlew jacocoTestReport\n./gradlew publish")

    first_memory = ProjectMemory(root, storage_root=storage)
    ExplorerAgent(repository_root=root, llm_client=FakeExplorerLLM(), project_memory=first_memory).run(
        "Analizar arquitectura", TaskState.create("primera")
    )
    state = TaskState.create("segunda incremental")
    report = ExplorerAgent(
        repository_root=root, llm_client=FakeExplorerLLM(),
        project_memory=ProjectMemory(root, storage_root=storage),
    ).run("Analizar arquitectura", state)

    assert dependencies.items() <= report_result_dependencies(state).items()
    assert any("Kotlin Gradle Plugin 2.1.10" in item for item in state.repository_findings)
    assert any("Java toolchain 21" in item for item in state.repository_findings)
    assert any("Spotless 6.25.0" in item for item in state.repository_findings)
    assert any("Picocli 4.7.6" in item for item in state.repository_findings)
    assert any("Gson 2.11.0" in item for item in state.repository_findings)
    assert any("JUnit Jupiter 5.10.3" in item for item in state.repository_findings)
    assert any("Foojay Resolver Convention 0.8.0" in item for item in state.repository_findings)
    commands = "\n".join(state.observations)
    for command in ("build", "test", "check", "spotlessCheck", "spotlessApply", "jacocoTestReport", "publish"):
        assert f"./gradlew {command}" in commands
    assert sum(item.startswith("risk=") for item in state.repository_findings) == 6
    assert any(
        source.reference == "runner/src/main/kotlin/org/printscript/runner/Runner.kt"
        for source in state.sources
    )
    assert state.commands_executed == ()
    rag_source = SourceReference("rag", "docs/printscript-language-spec.md")
    state.add_source(rag_source)
    state.add_subagent_result(SubagentResult(
        "researcher", "analizar", "completed",
        summary="HECHO CONFIRMADO: lexer, parser e interpreter forman el pipeline.",
        findings=("HECHO CONFIRMADO: flujo corroborado.",), sources=(rag_source,),
    ))
    summary = DeterministicAnalysisSummary().build(state)
    for module, module_dependencies in dependencies.items():
        assert f"{module} → {', '.join(module_dependencies)}" in summary
    for technology in (
        "Kotlin Gradle Plugin 2.1.10", "Java toolchain 21", "Spotless 6.25.0",
        "Picocli 4.7.6", "Gson 2.11.0", "JUnit Jupiter 5.10.3",
        "kotlin-test", "Foojay Resolver Convention 0.8.0",
    ):
        assert technology in summary
    for command in ("build", "test", "check", "spotlessCheck", "spotlessApply", "jacocoTestReport", "publish"):
        assert f"./gradlew {command}" in summary
    assert summary.count("Inferencia:") >= 6
    assert "Módulos declarados más de una vez: runner" in summary
    assert "Camino alternativo: `runner` conecta Lexer → Parser → Interpreter" in summary
    assert "Evidencia: `runner/src/main/kotlin/org/printscript/runner/Runner.kt`" in summary


def report_result_dependencies(state: TaskState) -> dict[str, tuple[str, ...]]:
    result = {}
    for finding in state.repository_findings:
        if not finding.startswith("internal_dependency="):
            continue
        relation = finding.removeprefix("internal_dependency=").split(";", 1)[0]
        module, dependencies = relation.split("->", 1)
        result[module.strip()] = tuple(item.strip() for item in dependencies.split(","))
    return result


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

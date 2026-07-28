"""Síntesis determinística del informe final de tareas analysis."""

from agents.analysis_summary import DeterministicAnalysisSummary
from agents.explorer import ExplorerAgent
from core.task_state import SourceReference, SubagentResult, TaskState


def printscript_state(*, with_researcher: bool = True) -> TaskState:
    state = TaskState.create("Analizar PrintScript", task_id="fake-printscript")
    state.add_repository_finding(
        "modules=common, token, lexer, parser, interpreter, formatter, linter, runner, cli; "
        "evidencia: settings.gradle.kts."
    )
    state.add_repository_finding("build_infrastructure=buildSrc; no son subproyectos declarados.")
    state.add_repository_finding("module_warning=El módulo 'runner' aparece repetido en settings.gradle.kts.")
    relations = {
        "cli": "lexer, parser, interpreter, formatter, linter, common, token",
        "runner": "common, token, lexer, parser, interpreter",
        "interpreter": "parser, common, token, lexer",
        "parser": "token, common", "linter": "parser, common, lexer, token",
        "formatter": "parser, common, token", "lexer": "token",
    }
    for module, dependencies in relations.items():
        state.add_repository_finding(
            f"internal_dependency={module} -> {dependencies}; evidencia: {module}/build.gradle."
        )
    state.add_repository_finding("module_versions=cli:1.5-SNAPSHOT, formatter:3.2-SNAPSHOT")
    state.add_repository_finding("duplicated_test_configuration=runner/interpreter")
    state.add_repository_finding("root_writes_hooks=installGitHooks; evidencia: build.gradle.kts.")
    state.add_repository_finding(
        "gradle_technology=Kotlin, Java toolchain, Gradle, Picocli, Gson, JUnit, "
        "Kotlin Test, Spotless, Foojay"
    )
    state.add_repository_finding(
        "entry points: cli/src/main/kotlin/org/printscript/cli/Main.kt, "
        "lexer/src/main/kotlin/org/printscript/lexer/Main.kt, "
        "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt, "
        "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt; evidencia: inventario."
    )
    paths = (
        "common/src/main/kotlin/org/printscript/common/Position.kt",
        "token/src/main/kotlin/token/Token.kt",
        "lexer/src/main/kotlin/org/printscript/lexer/Lexer.kt",
        "parser/src/main/kotlin/org/printscript/parser/DefaultParser.kt",
        "interpreter/src/main/kotlin/org/printscript/interpreter/Interpreter.kt",
        "formatter/src/main/kotlin/org/printscript/formatter/CodeFormatter.kt",
        "linter/src/main/kotlin/org/printscript/linter/Linter.kt",
        "runner/src/main/kotlin/org/printscript/runner/Runner.kt",
        "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt",
        "settings.gradle.kts", "build.gradle.kts", "docs/printscript-language-spec.md",
    )
    state.add_subagent_result(SubagentResult(
        "explorer", "Analizar", "completed", summary="Inventario confirmado.",
        files_relevant=paths,
    ))
    for path in paths:
        state.add_source(SourceReference("repository", path, "evidencia"))
    state.add_source(SourceReference("repository", "settings.gradle.kts", "duplicada"))
    for command in (
        "./gradlew build", "./gradlew test", "./gradlew check",
        "./gradlew spotlessCheck", "./gradlew spotlessApply",
        "./gradlew jacocoTestReport", "./gradlew publish",
    ):
        state.add_observation(f"Comando detectado: {command}; evidencia: archivo real.")
    if with_researcher:
        source = SourceReference("rag", "docs/printscript-language-spec.md", "arquitectura")
        state.add_source(source)
        state.add_subagent_result(SubagentResult(
            "researcher", "Analizar", "completed",
            summary=("## Evidencia confirmada\nPrintScript procesa archivos mediante lexer, "
                     "parser e interpreter.\nHECHO CONFIRMADO: repositorio y especificación coinciden."),
            findings=("HECHO CONFIRMADO: flujo corroborado por repositorio y RAG.",),
            recommendations=("No se confirmó una política única de releases.",),
            sources=(source,), confidence=0.9,
        ))
    return state


def test_builds_complete_bounded_printscript_report_from_research_and_explorer() -> None:
    report = DeterministicAnalysisSummary().build(printscript_state())

    for section in range(1, 15):
        assert f"## {section}." in report
    assert "PrintScript procesa archivos mediante lexer, parser e interpreter" in report
    assert "**common**: tipos y utilidades compartidas" in report
    assert "cli → lexer, parser, interpreter, formatter, linter, common, token" in report
    assert all(name in report for name in (
        "Kotlin", "Java toolchain", "Gradle", "Picocli", "Gson", "JUnit",
        "Kotlin Test", "Spotless", "Foojay",
    ))
    assert "cli/src/main/kotlin/org/printscript/cli/Main.kt" in report
    assert "archivo → FrontendAdapter → Lexer → Parser → AST → Interpreter" in report
    assert "./gradlew publish" in report
    assert "Declaración duplicada de runner" in report
    assert "Evidencia:" in report and "Inferencia:" in report
    assert "buildSrc`: infraestructura de build; no es un módulo declarado" in report
    assert "**buildSrc**" not in report
    assert "ExecutionCmd.kt" not in report
    assert report.count("[repository] `settings.gradle.kts`") == 1
    assert report.count("`docs/printscript-language-spec.md`") == 1
    assert "tests:" not in report
    assert len(report) <= DeterministicAnalysisSummary.MAX_OUTPUT_CHARS


def test_report_remains_structured_without_researcher_summary() -> None:
    report = DeterministicAnalysisSummary().build(printscript_state(with_researcher=False))

    assert "No se recibió una síntesis técnica de Researcher" in report
    assert "## 3. Módulos y responsabilidades" in report
    assert "## 14. Fuentes principales" in report


def test_raw_inventory_is_bounded_and_not_dumped() -> None:
    state = printscript_state()
    for index in range(200):
        state.add_repository_finding(f"tests: test_{index}.kt, other_{index}.kt")

    report = DeterministicAnalysisSummary().build(state)

    assert "test_199.kt" not in report
    assert len(report) <= DeterministicAnalysisSummary.MAX_OUTPUT_CHARS


def test_explorer_persists_gradle_evidence_without_running_commands() -> None:
    dependencies, versions, commands, risks, technologies = ExplorerAgent._gradle_evidence({
        "cli/build.gradle": (
            "version = '1.0'\nimplementation project(':lexer')\n"
            "implementation 'info.picocli:picocli:4.7.6'"
        ),
        "lexer/build.gradle": "version = '2.0'",
        "build.gradle.kts": (
            "tasks.register(\"installGitHooks\") { Files.copy(source, target); "
            "file.writeText(\"./gradlew spotlessApply\") }"
        ),
        "settings.gradle.kts": "id(\"org.gradle.toolchains.foojay-resolver-convention\")",
        ".github/workflows/ci.yml": "run: ./gradlew build\nrun: ./gradlew check",
        "buildSrc/build.gradle.kts": (
            "kotlin(\"jvm\")\nJavaLanguageVersion.of(21)\n"
            "com.diffplug.spotless\norg.jetbrains.kotlin:kotlin-test\n"
            "org.junit.jupiter\ncom.google.code.gson:gson"
        ),
    })

    assert dependencies == {"cli": ("lexer",)}
    assert versions == {"cli": "1.0", "lexer": "2.0"}
    assert {"./gradlew spotlessApply", "./gradlew build", "./gradlew check"} <= set(commands)
    assert any("instala hooks" in item for item in risks)
    assert "Gradle" in technologies
    assert any(item.startswith("Java toolchain 21") for item in technologies)

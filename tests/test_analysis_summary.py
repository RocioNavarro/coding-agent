"""Síntesis determinística del informe final de tareas analysis."""

import pytest

from agents.analysis_summary import DeterministicAnalysisSummary
from agents.explorer import ExplorerAgent
from core.task_state import SourceReference, SubagentResult, TaskState


def printscript_state(
    *, with_researcher: bool = True, complete_entry_finding: bool = True
) -> TaskState:
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
        "runner_flow=InputStream -> Lexer -> Parser -> Interpreter; "
        "evidencia: runner/src/main/kotlin/org/printscript/runner/Runner.kt."
    )
    state.add_repository_finding(
        "gradle_technology=Kotlin, Java toolchain, Gradle, Picocli, Gson, JUnit, "
        "Kotlin Test, Spotless, Foojay"
    )
    entry_points = [
        "cli/src/main/kotlin/org/printscript/cli/Main.kt",
        "lexer/src/main/kotlin/org/printscript/lexer/Main.kt",
    ]
    if complete_entry_finding:
        entry_points.extend((
            "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt",
            "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt",
        ))
    state.add_repository_finding(
        "entry points: " + ", ".join(entry_points) + "; evidencia: inventario."
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
        "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt",
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


def test_objective_is_functional_and_does_not_start_with_research_inventory() -> None:
    report = DeterministicAnalysisSummary().build(printscript_state())
    objective = report.split("## 1. Objetivo general\n", 1)[1].split("\n\n## 2.", 1)[0]

    assert objective.startswith(
        "El repositorio contiene una implementación modular de un lenguaje"
    )
    assert "análisis léxico" in objective
    assert "interpretación" in objective
    assert not objective.startswith("Evidencia confirmada")
    assert objective.count("`") <= 6


def test_confirmed_entry_points_are_listed_before_flows() -> None:
    report = DeterministicAnalysisSummary().build(printscript_state())
    section = report.split("## 6. Puntos de entrada y flujo de ejecución\n", 1)[1].split(
        "\n\n## 7.", 1
    )[0]

    for path in (
        "cli/src/main/kotlin/org/printscript/cli/Main.kt",
        "lexer/src/main/kotlin/org/printscript/lexer/Main.kt",
        "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt",
        "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt",
    ):
        assert f"`{path}`" in section
    assert "No confirmado" not in section
    assert section.index("Main.kt`") < section.index("Flujo principal:")
    assert "InputStream → Lexer → Parser → Interpreter" in section


def test_entry_points_merge_findings_and_confirmed_paths_in_stable_order() -> None:
    state = printscript_state(complete_entry_finding=False)
    # Repetir una ruta entre findings y fuentes prueba la deduplicación por ruta.
    state.add_source(SourceReference(
        "repository", "cli/src/main/kotlin/org/printscript/cli/Main.kt"
    ))

    report = DeterministicAnalysisSummary().build(state)
    section = report.split("## 6. Puntos de entrada y flujo de ejecución\n", 1)[1].split(
        "\n\n## 7.", 1
    )[0]
    expected = (
        "cli/src/main/kotlin/org/printscript/cli/Main.kt",
        "lexer/src/main/kotlin/org/printscript/lexer/Main.kt",
        "cli/src/main/kotlin/org/printscript/cli/commands/ExecuteCmd.kt",
        "cli/src/main/kotlin/org/printscript/cli/commands/AnalyzeCmd.kt",
    )
    positions = [section.index(f"`{path}`") for path in expected]
    assert positions == sorted(positions)
    assert all(section.count(f"`{path}`") == 1 for path in expected)
    assert "No confirmado" not in section


def test_missing_entry_points_are_reported_as_unconfirmed() -> None:
    state = TaskState.create("Analizar repositorio", task_id="without-entry-points")
    state.add_repository_finding("modules=lexer, parser, interpreter")
    for path in (
        "lexer/src/main/kotlin/example/Lexer.kt",
        "parser/src/main/kotlin/example/Parser.kt",
        "interpreter/src/main/kotlin/example/Interpreter.kt",
    ):
        state.add_source(SourceReference("repository", path))

    report = DeterministicAnalysisSummary().build(state)
    section = report.split("## 6. Puntos de entrada y flujo de ejecución\n", 1)[1].split(
        "\n\n## 7.", 1
    )[0]
    assert section.startswith("- No confirmado.")


def test_report_remains_structured_without_researcher_summary() -> None:
    report = DeterministicAnalysisSummary().build(printscript_state(with_researcher=False))

    assert "No se recibió una síntesis técnica de Researcher" in report
    assert "## 3. Módulos y responsabilidades" in report
    assert "## 14. Fuentes principales" in report


@pytest.mark.parametrize(
    "researcher",
    (
        SubagentResult("researcher", "analizar", "completed", summary="Resumen útil."),
        SubagentResult("researcher", "analizar", "completed", result="Resultado útil."),
        SubagentResult(
            "researcher", "analizar", "completed",
            findings=("HECHO CONFIRMADO: evidencia útil.",),
        ),
        SubagentResult(
            "researcher", "analizar", "completed",
            sources=(SourceReference("rag", "rag://evidencia"),),
        ),
    ),
)
def test_researcher_content_prevents_false_missing_summary_warning(
    researcher: SubagentResult,
) -> None:
    state = printscript_state(with_researcher=False)
    state.add_subagent_result(researcher)

    report = DeterministicAnalysisSummary().build(state)

    assert "No se recibió una síntesis técnica de Researcher" not in report
    assert researcher.summary is None or researcher.summary in report
    assert researcher.result is None or researcher.result in report


def test_empty_researcher_result_keeps_missing_summary_warning() -> None:
    state = printscript_state(with_researcher=False)
    state.add_subagent_result(SubagentResult("researcher", "analizar", "completed"))

    report = DeterministicAnalysisSummary().build(state)

    assert "No se recibió una síntesis técnica de Researcher" in report


def test_raw_inventory_is_bounded_and_not_dumped() -> None:
    state = printscript_state()
    for index in range(200):
        state.add_repository_finding(f"tests: test_{index}.kt, other_{index}.kt")

    report = DeterministicAnalysisSummary().build(state)

    assert "test_199.kt" not in report
    assert len(report) <= DeterministicAnalysisSummary.MAX_OUTPUT_CHARS


def test_explorer_persists_gradle_evidence_without_running_commands() -> None:
    dependencies, test_dependencies, versions, commands, risks, technologies, runner_flow = ExplorerAgent._gradle_evidence({
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
    assert test_dependencies == {}
    assert versions == {"cli": "1.0", "lexer": "2.0"}
    assert {"./gradlew spotlessApply", "./gradlew build", "./gradlew check"} <= set(commands)
    assert any("instala hooks" in item for item in risks)
    assert "Gradle" in technologies
    assert any(item.startswith("Java toolchain 21") for item in technologies)
    assert runner_flow == ()


def test_gradle_internal_dependencies_keep_production_and_test_scopes_separate() -> None:
    production, tests, *_ = ExplorerAgent._gradle_evidence({
        "parser/build.gradle": """
            implementation project(':token')
            api project(':common')
            compileOnly project(':annotations')
            runtimeOnly project(':runtime')
            testImplementation project(':lexer')
            testRuntimeOnly project(':test-runtime')
            testCompileOnly project(':test-annotations')
            testFixturesImplementation project(':fixtures')
        """,
        "api/build.gradle.kts": "implementation(project(\":common\"))",
    })

    assert production == {
        "parser": ("token", "common", "annotations", "runtime"),
        "api": ("common",),
    }
    assert tests == {
        "parser": ("lexer", "test-runtime", "test-annotations", "fixtures"),
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("./gradlew check", "./gradlew check"),
        ("./gradlew -q check", "./gradlew check"),
        ("./gradlew --quiet check", "./gradlew check"),
        ("./gradlew -q spotlessApply", "./gradlew spotlessApply"),
        ("./gradlew --no-daemon test", "./gradlew test"),
        ("./gradlew -p subproject build", "./gradlew build"),
        ("./gradlew --project-dir subproject build", "./gradlew build"),
        ("./gradlew --stacktrace jacocoTestReport", "./gradlew jacocoTestReport"),
        ("./gradlew -Dkey=value -Pprofile=ci publish", "./gradlew publish"),
    ),
)
def test_normalizes_gradle_commands_with_options(raw: str, expected: str) -> None:
    assert ExplorerAgent._gradle_commands(raw) == (expected,)


def test_gradle_commands_discard_incomplete_flags_and_deduplicate() -> None:
    content = "./gradlew -q\n./gradlew -q check\n./gradlew --quiet check"

    assert ExplorerAgent._gradle_commands(content) == ("./gradlew check",)


def test_runner_flow_requires_input_stream_and_pipeline_operations() -> None:
    base = {
        "runner/src/main/kotlin/org/example/runner/Runner.kt": (
            "fun run(input: InputStream) { val reader = InputStreamReader(input); "
            "val lexer = Lexer(provider); val tokens = lexer.lex(reader); "
            "val parser: Parser = parser; val ast = parser.parse(tokens); "
            "val interpreter: Interpreter = interpreter; interpreter.executeNode(ast.first()) }"
        )
    }
    *_, flow = ExplorerAgent._gradle_evidence(base)
    *_, short_flow = ExplorerAgent._gradle_evidence({
        "runner/src/main/kotlin/org/example/runner/Runner.kt": (
            "val lexer = Lexer(provider); lexer.lex(reader); parser.parse(tokens); "
            "interpreter.executeNode(node); Parser; Interpreter"
        )
    })

    assert flow == ("InputStream", "Lexer", "Parser", "Interpreter")
    assert short_flow == ("Lexer", "Parser", "Interpreter")

"""Explorer genérico basado en inventario acotado y detectores extensibles."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping, Sequence

from agents.base import AgentContext, BaseAgent
from agents.project_memory import MemoryCorruptionError, ProjectMemory, ProjectMemoryError
from agents.repository_detection import (
    BuildSystemDetector,
    DetectionEvidence,
    LanguageDetector,
    RepositoryDetection,
    RepositoryDetector,
    RepositorySnapshot,
    TechnologyDetector,
)
from core.llm_client import LLMClient
from core.task_state import SourceReference, SubagentResult, TaskState
from core.profiles import ProjectProfile
from core.observability import (
    NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent,
    emit_observation,
)
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry


EXPLORER_ALLOWED_TOOLS = frozenset(
    {"list_files", "find_files", "read_file", "search_text"}
)
IGNORED_DIRECTORIES = frozenset(
    {".coding-agent", ".git", ".gradle", ".idea", ".mypy_cache", ".pytest_cache", ".venv",
     "__pycache__", "build", "dist", "node_modules", "out", "target", "vendor"}
)
MAX_DISCOVERED_FILES = 5_000
MAX_SELECTED_FILE_BYTES = 256_000
MAX_TOOL_RESULTS = 200
DOCUMENT_NAMES = frozenset(
    {"readme", "contributing", "changelog", "architecture", "license"}
)
DOCUMENT_EXTENSIONS = frozenset({".md", ".rst", ".adoc"})
SCRIPT_EXTENSIONS = frozenset({".sh", ".bash", ".ps1", ".bat", ".cmd"})
CONFIG_NAMES = frozenset(
    {
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "tox.ini",
        "package.json", "tsconfig.json", "vite.config.js", "vite.config.ts",
        "webpack.config.js", "pom.xml", "build.gradle", "build.gradle.kts",
        "settings.gradle", "settings.gradle.kts", "gradle.properties", "go.mod",
        "Cargo.toml", "Gemfile", "composer.json", "CMakeLists.txt", "Makefile",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".editorconfig",
        ".pre-commit-config.yaml", ".eslintrc", ".eslintrc.json",
    }
)
ENTRY_POINT_NAMES = frozenset(
    {"main.py", "__main__.py", "app.py", "manage.py", "index.js", "index.ts",
     "server.js", "server.ts", "main.js", "main.ts", "main.go", "main.rs",
     "Main.java", "Main.kt", "Program.cs"}
)


EXPLORER_SYSTEM_PROMPT = """Sos Explorer, un analista genérico de repositorios.
Tu responsabilidad es comprender el workspace antes de que se realicen cambios.
Usá exclusivamente el inventario y la evidencia concreta recibida, o tools de lectura.
Describí estructura, lenguajes, build, dependencias, frameworks, configuración,
documentación, fuentes, tests, scripts, CI, entry points, convenciones y comandos.
Toda conclusión debe citar rutas que la respalden. Priorizá archivos relevantes para
la tarea concreta y explicitá límites o incertidumbres. No escribas ni modifiques,
no instales dependencias, no ejecutes acciones destructivas y no consultes la web."""


@dataclass(frozen=True)
class RepositoryInventory:
    """Clasificación neutral del árbol, independiente de lenguajes concretos."""

    files: tuple[str, ...]
    directories: tuple[str, ...]
    configuration_files: tuple[str, ...]
    documentation_files: tuple[str, ...]
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    script_files: tuple[str, ...]
    ci_files: tuple[str, ...]
    entry_points: tuple[str, ...]
    files_inspected: tuple[str, ...]


@dataclass(frozen=True)
class ExplorerReport:
    """Resultado determinista del reconocimiento previo a la síntesis del LLM."""

    inventory: RepositoryInventory
    detections: tuple[RepositoryDetection, ...]
    commands: tuple[str, ...]
    conventions: tuple[RepositoryDetection, ...]
    relevant_files: tuple[str, ...]
    architecture_summary: str
    exploration_metrics: Mapping[str, object] = field(default_factory=dict)

    def facts(self) -> tuple[str, ...]:
        facts = [f"Resumen de arquitectura: {self.architecture_summary}"]
        artifact_groups = (
            ("configuración", self.inventory.configuration_files),
            ("documentación", self.inventory.documentation_files),
            ("código fuente", self.inventory.source_files),
            ("tests", self.inventory.test_files),
            ("scripts", self.inventory.script_files),
            ("CI", self.inventory.ci_files),
            ("entry points", self.inventory.entry_points),
        )
        facts.extend(
            f"{category}: {', '.join(paths)}. Evidencia: esas rutas del inventario."
            for category, paths in artifact_groups
            if paths
        )
        for detection in (*self.detections, *self.conventions):
            paths = ", ".join(item.path for item in detection.evidence)
            facts.append(
                f"{detection.category}: {detection.name}. Evidencia: {paths}."
            )
        for detection in self.detections:
            evidence = ", ".join(item.path for item in detection.evidence)
            facts.extend(
                f"comando: {command}. Evidencia: {evidence}."
                for command in detection.commands
            )
        return tuple(facts)


class ExplorerAgent(BaseAgent):
    """Inspecciona el workspace con detectores inyectables y acceso de sólo lectura."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        llm_client: LLMClient,
        detectors: Sequence[RepositoryDetector] | None = None,
        project_memory: ProjectMemory | None = None,
        profile: ProjectProfile | None = None,
        observability: ObservabilityClient | None = None,
        name: str = "explorer",
    ) -> None:
        super().__init__(
            name=name,
            role="Generic Repository Explorer",
            system_prompt=EXPLORER_SYSTEM_PROMPT,
            allowed_tools=sorted(EXPLORER_ALLOWED_TOOLS),
            llm_client=llm_client,
        )
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root debe ser un directorio existente.")
        self.repository_root = root
        self.project_memory = project_memory
        self.observability = observability or NoOpObservabilityClient()
        self.profile = profile or ProjectProfile()
        self.detectors = tuple(
            detectors
            if detectors is not None
            else (LanguageDetector(), BuildSystemDetector(), TechnologyDetector())
        )
        if not all(isinstance(detector, RepositoryDetector) for detector in self.detectors):
            raise TypeError("Todos los detectores deben implementar RepositoryDetector.")
        self.tool_registry = build_explorer_registry(root)

    def specialization_prompt(self) -> str:
        return (
            "El contexto contiene una exploración acotada generada por detectores. "
            "No asumas tecnologías ausentes ni haber leído fuentes no inspeccionadas."
        )

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
    ) -> SubagentResult:
        report = self.explore(instruction)
        self._record_exploration_strategy(report, task_state)
        self._record_report(report, task_state)
        self._record_profile(report, task_state)
        self._record_memory(report)
        existing = context or AgentContext()
        bounded_context = AgentContext(
            facts=(*existing.facts, *report.facts()),
            sources=existing.sources,
            files=tuple(dict.fromkeys((*existing.files, *report.relevant_files))),
            constraints=(
                *existing.constraints,
                "Citar evidencia concreta para cada conclusión.",
                "No solicitar escritura, instalaciones, comandos mutadores ni web.",
            ),
        )
        return super().run(
            instruction,
            task_state,
            bounded_context,
            available_tools or self.tool_registry,
        )

    def explore(self, instruction: str) -> ExplorerReport:
        inventory, contents, metrics = self._scan()
        snapshot = RepositorySnapshot(
            files=inventory.files,
            directories=inventory.directories,
            contents=contents,
        )
        detections = tuple(
            detection
            for detector in self.detectors
            for detection in detector.detect(snapshot)
        )
        commands = tuple(sorted({command for item in detections for command in item.commands}))
        conventions = self._detect_conventions(inventory)
        relevant = self._select_relevant_files(instruction, inventory, detections)
        summary = self._summarize(inventory, detections)
        return ExplorerReport(
            inventory=inventory,
            detections=detections,
            commands=commands,
            conventions=conventions,
            relevant_files=relevant,
            architecture_summary=summary,
            exploration_metrics=metrics,
        )

    def _scan(self) -> tuple[RepositoryInventory, dict[str, str], dict[str, object]]:
        files, directories = self._walk()
        language_detector = next(
            (item for item in self.detectors if isinstance(item, LanguageDetector)),
            LanguageDetector(),
        )
        source_extensions = language_detector.source_extensions
        configuration = tuple(path for path in files if self._is_configuration(path))
        documentation = tuple(path for path in files if self._is_documentation(path))
        sources = tuple(path for path in files if Path(path).suffix in source_extensions)
        tests = tuple(path for path in sources if self._is_test(path))
        scripts = tuple(path for path in files if self._is_script(path))
        ci_files = tuple(path for path in files if self._is_ci(path))
        entry_points = tuple(path for path in files if Path(path).name in ENTRY_POINT_NAMES)
        selected = tuple(dict.fromkeys((*configuration, *documentation, *scripts, *ci_files)))
        current_fingerprints = {
            path: self._metadata_fingerprint(path) for path in files
        }
        previous: dict[str, str] = {}
        memory_context: Mapping[str, object] = {}
        memory_valid = False
        fallback_reason = "No existe memoria de exploración previa."
        if self.project_memory is not None:
            try:
                memory_context = self.project_memory.exploration_context()
                raw = memory_context.get("known_file_fingerprints", {})
                if isinstance(raw, Mapping) and raw:
                    previous = {
                        str(path): str(value) for path, value in raw.items()
                    }
                    memory_valid = True
                    fallback_reason = "Fingerprints previos disponibles."
            except (MemoryCorruptionError, ProjectMemoryError, ValueError) as error:
                fallback_reason = f"Memoria inválida; exploración completa: {error}"
        if memory_valid:
            hinted = tuple(
                path
                for key in ("known_important_files", "known_manifests")
                for path in memory_context.get(key, ())
                if isinstance(path, str) and path in files
            )
            selected = tuple(dict.fromkeys((*hinted, *selected)))
        new_files = tuple(path for path in files if path not in previous)
        modified_files = tuple(
            path for path in files
            if path in previous and previous[path] != current_fingerprints[path]
        )
        stable_selected = tuple(
            path for path in selected
            if path in previous and path not in modified_files
        )
        revalidated = stable_selected[:1] if memory_valid else ()
        selected_to_read = (
            selected if not memory_valid else
            tuple(dict.fromkeys((
                *(path for path in selected if path in new_files or path in modified_files),
                *revalidated,
            )))
        )
        contents = self._read_selected(selected_to_read)
        avoided = tuple(path for path in stable_selected if path not in revalidated)
        strategy = "incremental" if memory_valid else "full"
        metrics: dict[str, object] = {
            "strategy": strategy,
            "reason": fallback_reason,
            "files_reused": stable_selected,
            "files_avoided": avoided,
            "files_revalidated": revalidated,
            "files_new": new_files,
            "files_modified": modified_files,
            "files_inspected": tuple(contents),
            "file_fingerprints": current_fingerprints,
            "memory_hints": {
                "known_architecture": memory_context.get("known_architecture", ""),
                "known_important_files": tuple(memory_context.get("known_important_files", ())),
                "known_technologies": tuple(memory_context.get("known_technologies", ())),
                "known_commands": tuple(memory_context.get("known_commands", ())),
                "known_modules": tuple(memory_context.get("known_modules", ())),
                "known_manifests": tuple(memory_context.get("known_manifests", ())),
                "last_explored_at": memory_context.get("last_explored_at"),
            } if memory_valid else {},
        }
        inventory = RepositoryInventory(
            files=files,
            directories=directories,
            configuration_files=configuration,
            documentation_files=documentation,
            source_files=sources,
            test_files=tests,
            script_files=scripts,
            ci_files=ci_files,
            entry_points=entry_points,
            files_inspected=tuple(contents),
        )
        return inventory, contents, metrics

    def _metadata_fingerprint(self, relative: str) -> str:
        stat = (self.repository_root / relative).stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _walk(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        files: list[str] = []
        directories: list[str] = []
        pending = [self.repository_root]
        while pending:
            directory = pending.pop()
            for item in sorted(directory.iterdir(), key=lambda path: path.name):
                if item.is_symlink() or item.name == ".env":
                    continue
                relative = item.relative_to(self.repository_root).as_posix()
                if item.is_dir():
                    if item.name not in IGNORED_DIRECTORIES:
                        directories.append(relative)
                        pending.append(item)
                else:
                    files.append(relative)
                    if len(files) > MAX_DISCOVERED_FILES:
                        raise ValueError("El repositorio supera el límite de exploración.")
        return tuple(sorted(files)), tuple(sorted(directories))

    def _read_selected(self, paths: Sequence[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        for relative in paths:
            path = self.repository_root / relative
            try:
                if path.stat().st_size <= MAX_SELECTED_FILE_BYTES:
                    contents[relative] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
        return contents

    @staticmethod
    def _detect_conventions(
        inventory: RepositoryInventory,
    ) -> tuple[RepositoryDetection, ...]:
        detections: list[RepositoryDetection] = []
        test_suffixes = sorted(
            {Path(path).stem[-4:] for path in inventory.test_files if Path(path).stem.endswith("Test")}
        )
        if test_suffixes:
            evidence = tuple(
                DetectionEvidence(path, "nombre de archivo de test")
                for path in inventory.test_files[:10]
            )
            detections.append(RepositoryDetection("convention", "tests con sufijo Test", evidence))
        conventional_dirs = tuple(
            path
            for path in inventory.directories
            if Path(path).name.casefold() in {"src", "lib", "app", "test", "tests", "spec"}
        )
        if conventional_dirs:
            detections.append(
                RepositoryDetection(
                    "convention",
                    "separación de fuentes/tests por directorios",
                    tuple(DetectionEvidence(path, "directorio estructural") for path in conventional_dirs),
                )
            )
        return tuple(detections)

    def _select_relevant_files(
        self,
        instruction: str,
        inventory: RepositoryInventory,
        detections: Sequence[RepositoryDetection],
    ) -> tuple[str, ...]:
        tokens = {
            token.casefold()
            for token in " ".join((instruction, *self.profile.search_tags)).replace("_", " ").split()
            if len(token) >= 3
        }
        matches = [
            path for path in inventory.files if any(token in path.casefold() for token in tokens)
        ]
        evidence = [item.path for detection in detections for item in detection.evidence]
        supporting = [
            *inventory.configuration_files,
            *inventory.documentation_files[:10],
            *inventory.entry_points,
        ]
        important = [path for path in self.profile.important_files if path in inventory.files]
        return tuple(dict.fromkeys((*important, *matches[:50], *evidence, *supporting)))

    def _record_profile(self, report: ExplorerReport, state: TaskState) -> None:
        if not any((self.profile.name, self.profile.expected_technologies,
                    self.profile.important_files, self.profile.search_tags)):
            return
        detected = {
            item.name.casefold(): item.name
            for item in report.detections
            if item.category in {"language", "framework", "test_framework", "tool", "build_system", "technology"}
        }
        state.add_observation(f"Perfil efectivo: {self.profile.name or 'sin nombre'}.")
        for expected in self.profile.expected_technologies:
            confirmed = detected.get(expected.casefold())
            if confirmed:
                state.add_observation(
                    f"Tecnología esperada por perfil confirmada: {expected}; detectada: {confirmed}."
                )
            else:
                state.add_warning(
                    f"Discrepancia de perfil: tecnología esperada no confirmada: {expected}."
                )
        used = tuple(path for path in self.profile.important_files if path in report.inventory.files)
        missing = tuple(path for path in self.profile.important_files if path not in report.inventory.files)
        if used:
            state.add_observation("Archivos importantes utilizados: " + ", ".join(used))
        if missing:
            state.add_warning("Archivos importantes no encontrados: " + ", ".join(missing))
        if self.profile.search_tags:
            state.add_observation("Tags de búsqueda usados: " + ", ".join(self.profile.search_tags))

    def _record_exploration_strategy(
        self, report: ExplorerReport, state: TaskState
    ) -> None:
        metrics = report.exploration_metrics
        strategy = str(metrics.get("strategy", "full"))
        state.add_observation(f"Memoria de exploración consultada: {self.project_memory is not None}.")
        state.add_observation(
            f"Estrategia de exploración: {strategy}; motivo: {metrics.get('reason', '')}"
        )
        labels = (
            ("Elementos reutilizados", "files_reused"),
            ("Archivos evitados", "files_avoided"),
            ("Archivos revalidados", "files_revalidated"),
            ("Archivos nuevos", "files_new"),
            ("Archivos modificados desde memoria", "files_modified"),
        )
        for label, key in labels:
            values = tuple(metrics.get(key, ()))
            state.add_observation(f"{label}: {', '.join(values) if values else 'ninguno'}.")
        hints = metrics.get("memory_hints", {})
        if isinstance(hints, Mapping) and hints:
            state.add_observation(
                "Memoria reutilizada como pista validada parcialmente: "
                f"arquitectura={bool(hints.get('known_architecture'))}; "
                f"archivos_importantes={len(tuple(hints.get('known_important_files', ())))}; "
                f"tecnologías={len(tuple(hints.get('known_technologies', ())))}; "
                f"comandos={len(tuple(hints.get('known_commands', ())))}."
            )
        emit_observation(
            self.observability,
            ObservabilityEvent(
                "agent", "explorer-strategy", agent=self.name,
                payload={
                    "strategy": strategy,
                    "reason": metrics.get("reason"),
                    "files_avoided": len(tuple(metrics.get("files_avoided", ()))),
                    "files_inspected": len(tuple(metrics.get("files_inspected", ()))),
                    "files_revalidated": tuple(metrics.get("files_revalidated", ())),
                    "files_new": tuple(metrics.get("files_new", ())),
                    "files_modified": tuple(metrics.get("files_modified", ())),
                    "memory_invalidated": str(metrics.get("reason", "")).startswith(
                        "Memoria inválida"
                    ),
                },
            ),
        )

    @staticmethod
    def _summarize(
        inventory: RepositoryInventory, detections: Sequence[RepositoryDetection]
    ) -> str:
        languages = sorted(item.name for item in detections if item.category == "language")
        builds = sorted(item.name for item in detections if item.category == "build_system")
        frameworks = sorted(item.name for item in detections if item.category == "framework")
        return (
            f"Repositorio con {len(inventory.files)} archivos y {len(inventory.directories)} "
            f"directorios; lenguajes: {', '.join(languages) or 'no determinados'}; "
            f"build/dependencias: {', '.join(builds) or 'no determinado'}; "
            f"frameworks: {', '.join(frameworks) or 'no detectados'}; "
            f"{len(inventory.entry_points)} entry point(s) y {len(inventory.test_files)} test(s)."
        )

    def _record_report(self, report: ExplorerReport, state: TaskState) -> None:
        state.add_repository_finding(f"Arquitectura: {report.architecture_summary}")
        artifact_groups = (
            ("configuración", report.inventory.configuration_files),
            ("documentación", report.inventory.documentation_files),
            ("código fuente", report.inventory.source_files),
            ("tests", report.inventory.test_files),
            ("scripts", report.inventory.script_files),
            ("CI", report.inventory.ci_files),
            ("entry points", report.inventory.entry_points),
        )
        for category, paths in artifact_groups:
            if paths:
                state.add_repository_finding(
                    f"{category}: {', '.join(paths)}; evidencia: rutas del inventario."
                )
        for detection in (*report.detections, *report.conventions):
            paths = ", ".join(item.path for item in detection.evidence)
            state.add_repository_finding(
                f"{detection.category}={detection.name}; evidencia: {paths}."
            )
            for evidence in detection.evidence:
                state.add_source(
                    SourceReference("repository", evidence.path, evidence.detail)
                )
            for command in detection.commands:
                state.add_observation(
                    f"Comando detectado: {command}; evidencia: {paths}."
                )
        for path in report.inventory.files_inspected:
            state.record_file_read(path)

    def _record_memory(self, report: ExplorerReport) -> None:
        if self.project_memory is None:
            return
        try:
            self.project_memory.load()
        except MemoryCorruptionError:
            return
        self.project_memory.update_project_summary(report.architecture_summary)
        self.project_memory.update_architecture(report.architecture_summary)
        for directory in report.inventory.directories:
            if "/" not in directory:
                self.project_memory.add_module(directory)
        for path in report.relevant_files:
            self.project_memory.add_important_file(path)
        for detection in report.detections:
            if detection.category in {
                "language", "framework", "test_framework", "tool", "build_system"
            }:
                self.project_memory.add_technology(detection.name)
            if detection.category in {"dependency", "build_system", "framework"}:
                self.project_memory.add_dependency(detection.name)
            evidence = ", ".join(item.path for item in detection.evidence)
            for command in detection.commands:
                self.project_memory.add_known_command(command, evidence)
        for convention in report.conventions:
            self.project_memory.add_convention(convention.name)
        self.project_memory.record_exploration(
            file_fingerprints=dict(
                report.exploration_metrics.get("file_fingerprints", {})
            ),
            manifests=report.inventory.configuration_files,
        )
        self.project_memory.save()

    @staticmethod
    def _is_configuration(path: str) -> bool:
        candidate = Path(path)
        return candidate.name in CONFIG_NAMES or candidate.name.startswith(
            (".github", ".gitlab", ".circleci")
        )

    @staticmethod
    def _is_documentation(path: str) -> bool:
        candidate = Path(path)
        return (
            candidate.suffix.casefold() in DOCUMENT_EXTENSIONS
            or candidate.stem.casefold() in DOCUMENT_NAMES
        )

    @staticmethod
    def _is_test(path: str) -> bool:
        candidate = Path(path)
        parts = {part.casefold() for part in candidate.parts}
        stem = candidate.stem.casefold()
        return bool(parts & {"test", "tests", "spec", "specs", "__tests__"}) or (
            stem.startswith("test_") or stem.endswith(("_test", ".test", ".spec", "test"))
        )

    @staticmethod
    def _is_script(path: str) -> bool:
        candidate = Path(path)
        return candidate.suffix.casefold() in SCRIPT_EXTENSIONS or bool(
            {part.casefold() for part in candidate.parts} & {"bin", "scripts"}
        )

    @staticmethod
    def _is_ci(path: str) -> bool:
        lowered = path.casefold()
        return (
            lowered.startswith(".github/workflows/")
            or lowered.startswith(".circleci/")
            or Path(path).name.casefold() in {".gitlab-ci.yml", "jenkinsfile", "azure-pipelines.yml"}
        )


def build_explorer_registry(repository_root: str | Path) -> ToolRegistry:
    """Crea cuatro tools de lectura confinadas a la raíz de Explorer."""
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError("La raíz del repositorio debe existir.")

    def resolve(relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or any(part in {"..", ".env"} for part in candidate.parts):
            raise ValueError("Ruta no permitida.")
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("La ruta escapa del repositorio.") from error
        return resolved

    def visible_files() -> list[Path]:
        result: list[Path] = []
        pending = [root]
        while pending and len(result) < MAX_DISCOVERED_FILES:
            directory = pending.pop()
            for item in directory.iterdir():
                if item.is_symlink() or item.name == ".env":
                    continue
                if item.is_dir() and item.name not in IGNORED_DIRECTORIES:
                    pending.append(item)
                elif item.is_file():
                    result.append(item)
        return sorted(result, key=lambda item: item.relative_to(root).as_posix())

    def list_files(path: str = ".") -> list[str]:
        directory = resolve(path)
        if not directory.is_dir():
            raise ValueError("La ruta no es un directorio.")
        return [
            item.relative_to(root).as_posix()
            for item in sorted(directory.iterdir(), key=lambda item: item.name)
            if not item.is_symlink() and item.name not in IGNORED_DIRECTORIES and item.name != ".env"
        ][:MAX_TOOL_RESULTS]

    def find_files(pattern: str) -> list[str]:
        return [
            item.relative_to(root).as_posix()
            for item in visible_files()
            if fnmatch(item.relative_to(root).as_posix(), pattern) or fnmatch(item.name, pattern)
        ][:MAX_TOOL_RESULTS]

    def read_file(path: str) -> str:
        candidate = resolve(path)
        if not candidate.is_file() or candidate.stat().st_size > MAX_SELECTED_FILE_BYTES:
            raise ValueError("Archivo inválido o demasiado grande.")
        return candidate.read_text(encoding="utf-8")

    def search_text(query: str, path: str = ".") -> list[str]:
        if not query.strip():
            raise ValueError("query no puede estar vacío.")
        base = resolve(path)
        candidates = [base] if base.is_file() else [item for item in visible_files() if item.is_relative_to(base)]
        matches: list[str] = []
        for candidate in candidates:
            try:
                if candidate.stat().st_size > MAX_SELECTED_FILE_BYTES:
                    continue
                for number, line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append(f"{candidate.relative_to(root).as_posix()}:{number}:{line.strip()}")
                        if len(matches) >= MAX_TOOL_RESULTS:
                            return matches
            except (OSError, UnicodeError):
                continue
        return matches

    def schema(properties: dict[str, dict[str, object]], required: list[str] | None = None) -> dict[str, object]:
        return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}

    registry = ToolRegistry()
    definitions = (
        ToolDefinition("list_files", "Lista una carpeta.", schema({"path": {"type": "string", "default": "."}}), list_files, False),
        ToolDefinition("find_files", "Busca archivos por glob.", schema({"pattern": {"type": "string"}}, ["pattern"]), find_files, False),
        ToolDefinition("read_file", "Lee un archivo de texto acotado.", schema({"path": {"type": "string"}}, ["path"]), read_file, False),
        ToolDefinition("search_text", "Busca texto con ruta y línea.", schema({"query": {"type": "string"}, "path": {"type": "string", "default": "."}}, ["query"]), search_text, False),
    )
    for definition in definitions:
        registry.register(definition)
    return registry

"""Contratos y detectores extensibles para analizar repositorios heterogéneos."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class DetectionEvidence:
    """Evidencia concreta que respalda una detección del repositorio."""

    path: str
    detail: str


@dataclass(frozen=True)
class RepositoryDetection:
    """Tecnología o capacidad detectada junto con evidencia y comandos."""

    category: str
    name: str
    evidence: tuple[DetectionEvidence, ...]
    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositorySnapshot:
    """Vista acotada: inventario completo de rutas y contenido sólo seleccionado."""

    files: tuple[str, ...]
    directories: tuple[str, ...]
    contents: Mapping[str, str]

    def has_file(self, name: str) -> tuple[str, ...]:
        """Localiza archivos por nombre, sin asumir su posición en el árbol."""
        return tuple(path for path in self.files if Path(path).name == name)


class RepositoryDetector(ABC):
    """Interfaz común para detectores independientes y combinables."""

    @abstractmethod
    def detect(self, snapshot: RepositorySnapshot) -> tuple[RepositoryDetection, ...]:
        """Devuelve únicamente detecciones respaldadas por evidencia concreta."""


@dataclass(frozen=True)
class LanguageRule:
    name: str
    extensions: tuple[str, ...]
    marker_files: tuple[str, ...] = ()


DEFAULT_LANGUAGE_RULES: tuple[LanguageRule, ...] = (
    LanguageRule("Python", (".py", ".pyi"), ("pyproject.toml",)),
    LanguageRule("JavaScript", (".js", ".jsx", ".mjs", ".cjs"), ("package.json",)),
    LanguageRule("TypeScript", (".ts", ".tsx"), ("tsconfig.json",)),
    LanguageRule("Java", (".java",), ("pom.xml",)),
    LanguageRule("Kotlin", (".kt", ".kts"), ()),
    LanguageRule("Go", (".go",), ("go.mod",)),
    LanguageRule("Rust", (".rs",), ("Cargo.toml",)),
    LanguageRule("Ruby", (".rb",), ("Gemfile",)),
    LanguageRule("PHP", (".php",), ("composer.json",)),
    LanguageRule("C#", (".cs",), ()),
    LanguageRule("C/C++", (".c", ".h", ".cc", ".cpp", ".hpp"), ("CMakeLists.txt",)),
)


class LanguageDetector(RepositoryDetector):
    """Detecta lenguajes mediante reglas declarativas de extensiones y marcadores."""

    def __init__(self, rules: Sequence[LanguageRule] = DEFAULT_LANGUAGE_RULES) -> None:
        self._rules = tuple(rules)

    @property
    def source_extensions(self) -> frozenset[str]:
        return frozenset(ext for rule in self._rules for ext in rule.extensions)

    def detect(self, snapshot: RepositorySnapshot) -> tuple[RepositoryDetection, ...]:
        detections: list[RepositoryDetection] = []
        for rule in self._rules:
            extension_matches = tuple(
                path for path in snapshot.files if Path(path).suffix in rule.extensions
            )
            marker_matches = tuple(
                path
                for marker in rule.marker_files
                for path in snapshot.has_file(marker)
            )
            if not extension_matches and not marker_matches:
                continue
            evidence = tuple(
                DetectionEvidence(path, f"extensión {Path(path).suffix}")
                for path in extension_matches[:10]
            ) + tuple(
                DetectionEvidence(path, "archivo marcador del lenguaje")
                for path in marker_matches
            )
            detections.append(RepositoryDetection("language", rule.name, evidence))
        return tuple(detections)


ManifestParser = Callable[[str, str], tuple[tuple[str, ...], tuple[str, ...]]]


@dataclass(frozen=True)
class BuildRule:
    name: str
    manager: str
    marker_files: tuple[str, ...]
    default_commands: tuple[str, ...]
    parser: ManifestParser | None = None


def _parse_package_json(path: str, content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return (), ()
    if not isinstance(payload, dict):
        return (), ()
    dependencies: set[str] = set()
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        values = payload.get(section, {})
        if isinstance(values, dict):
            dependencies.update(str(name) for name in values)
    scripts = payload.get("scripts", {})
    commands = tuple(
        f"npm run {name}"
        for name in ("build", "test", "lint", "start", "dev")
        if isinstance(scripts, dict) and name in scripts
    )
    return tuple(sorted(dependencies)), commands


_PYPROJECT_DEPENDENCY_RE = re.compile(
    r"(?:dependencies\s*=\s*\[|[a-zA-Z0-9_.-]+\s*=)\s*([^\n]+)"
)


def _parse_pyproject(path: str, content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dependencies = tuple(
        sorted({match.strip(" []\"',") for match in _PYPROJECT_DEPENDENCY_RE.findall(content)})
    )
    commands: list[str] = []
    if "pytest" in content.casefold():
        commands.append("python -m pytest")
    if "ruff" in content.casefold():
        commands.append("ruff check .")
    if "[build-system]" in content:
        commands.append("python -m build")
    return dependencies, tuple(commands)


_MAVEN_DEPENDENCY_RE = re.compile(
    r"<dependency>.*?<groupId>([^<]+)</groupId>.*?<artifactId>([^<]+)</artifactId>",
    re.DOTALL,
)


def _parse_maven(path: str, content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dependencies = tuple(
        sorted(f"{group}:{artifact}" for group, artifact in _MAVEN_DEPENDENCY_RE.findall(content))
    )
    return dependencies, ()


def _parse_gradle(path: str, content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dependency_re = re.compile(
        r"(?:api|implementation|testImplementation|runtimeOnly)\s*\(?\s*[\"']([^\"']+)"
    )
    return tuple(sorted(set(dependency_re.findall(content)))), ()


DEFAULT_BUILD_RULES: tuple[BuildRule, ...] = (
    BuildRule("Python packaging", "pip", ("pyproject.toml",), (), _parse_pyproject),
    BuildRule("npm", "npm", ("package.json",), (), _parse_package_json),
    BuildRule("Yarn", "yarn", ("yarn.lock",), ("yarn test",)),
    BuildRule("pnpm", "pnpm", ("pnpm-lock.yaml",), ("pnpm test",)),
    BuildRule("Maven", "Maven", ("pom.xml",), ("mvn test", "mvn package"), _parse_maven),
    BuildRule(
        "Gradle",
        "Gradle",
        ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"),
        ("./gradlew test", "./gradlew build"),
        _parse_gradle,
    ),
    BuildRule("Cargo", "Cargo", ("Cargo.toml",), ("cargo test", "cargo build")),
    BuildRule("Go modules", "Go", ("go.mod",), ("go test ./...", "go build ./...")),
    BuildRule("Bundler", "Bundler", ("Gemfile",), ("bundle exec rake test",)),
    BuildRule("Composer", "Composer", ("composer.json",), ("composer test",)),
    BuildRule("CMake", "CMake", ("CMakeLists.txt",), ("cmake --build build",)),
)


class BuildSystemDetector(RepositoryDetector):
    """Detecta build y dependencias mediante reglas con parsers intercambiables."""

    def __init__(self, rules: Sequence[BuildRule] = DEFAULT_BUILD_RULES) -> None:
        self._rules = tuple(rules)

    def detect(self, snapshot: RepositorySnapshot) -> tuple[RepositoryDetection, ...]:
        detections: list[RepositoryDetection] = []
        for rule in self._rules:
            matches = tuple(
                path
                for marker in rule.marker_files
                for path in snapshot.has_file(marker)
            )
            if not matches:
                continue
            dependencies: set[str] = set()
            commands = set(rule.default_commands)
            if rule.parser is not None:
                for path in matches:
                    parsed_dependencies, parsed_commands = rule.parser(
                        path, snapshot.contents.get(path, "")
                    )
                    dependencies.update(parsed_dependencies)
                    commands.update(parsed_commands)
            detections.append(
                RepositoryDetection(
                    "build_system",
                    rule.name,
                    tuple(
                        DetectionEvidence(path, f"gestor/build: {rule.manager}")
                        for path in matches
                    ),
                    tuple(sorted(commands)),
                )
            )
            detections.extend(
                RepositoryDetection(
                    "dependency",
                    dependency,
                    tuple(DetectionEvidence(path, "declaración de dependencia") for path in matches),
                )
                for dependency in sorted(dependencies)
                if dependency
            )
        return tuple(detections)


@dataclass(frozen=True)
class TechnologyRule:
    category: str
    name: str
    tokens: tuple[str, ...]


DEFAULT_TECHNOLOGY_RULES: tuple[TechnologyRule, ...] = (
    TechnologyRule("framework", "Django", ("django",)),
    TechnologyRule("framework", "FastAPI", ("fastapi",)),
    TechnologyRule("framework", "Flask", ("flask",)),
    TechnologyRule("framework", "React", ("react",)),
    TechnologyRule("framework", "Vue", ("vue",)),
    TechnologyRule("framework", "Express", ("express",)),
    TechnologyRule("framework", "Spring", ("spring-boot", "springframework")),
    TechnologyRule("test_framework", "pytest", ("pytest",)),
    TechnologyRule("test_framework", "Jest", ("jest",)),
    TechnologyRule("test_framework", "JUnit", ("junit",)),
    TechnologyRule("tool", "Ruff", ("ruff",)),
    TechnologyRule("tool", "ESLint", ("eslint",)),
)


class TechnologyDetector(RepositoryDetector):
    """Reconoce frameworks y herramientas desde contenido ya seleccionado."""

    def __init__(
        self, rules: Sequence[TechnologyRule] = DEFAULT_TECHNOLOGY_RULES
    ) -> None:
        self._rules = tuple(rules)

    def detect(self, snapshot: RepositorySnapshot) -> tuple[RepositoryDetection, ...]:
        detections: list[RepositoryDetection] = []
        for rule in self._rules:
            evidence = tuple(
                DetectionEvidence(path, f"referencia a {rule.name}")
                for path, content in snapshot.contents.items()
                if any(token in content.casefold() for token in rule.tokens)
            )
            if evidence:
                detections.append(
                    RepositoryDetection(rule.category, rule.name, evidence)
                )
        return tuple(detections)

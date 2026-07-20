"""Builders de repositorios temporales mínimos para pruebas integrales."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TemporaryRepository:
    """Repositorio aislado con snapshot inicial fácilmente inspeccionable."""

    kind: str
    root: Path
    initial_files: Mapping[str, str]
    memory_root: Path | None = None

    def files(self) -> tuple[str, ...]:
        return tuple(
            sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())
        )

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def modified_files(self) -> tuple[str, ...]:
        return tuple(
            path for path, content in self.initial_files.items()
            if not (self.root / path).is_file()
            or (self.root / path).read_text(encoding="utf-8") != content
        )


REPOSITORY_CONTENTS: dict[str, dict[str, str]] = {
    "analysis": {
        "README.md": "# Sample component\nArchitecture notes only.\n",
        "src/component.py": "VALUE = 'stable'\n",
    },
    "simple_change": {
        "README.md": "# Small change\n",
        "src/value.py": "VALUE = 'before'\n",
        "tests/test_value.py": "def test_value(): pass\n",
    },
    "rag_docs": {
        "README.md": "# Documented component\n",
        "docs/contract.md": "# Contract\nThe component returns a normalized value.\n",
        "src/component.py": "def value(): return 'normalized'\n",
    },
    "persistent_memory": {
        "README.md": "# Component with project decisions\n",
        "docs/architecture.md": "# Architecture\nA small isolated component.\n",
        "docs/validation.md": "# Validation\nUse the configured project check.\n",
        "pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-q'\n",
        "src/component.py": "VALUE = 1\n",
    },
    "failed_command": {
        "README.md": "# Component with deterministic validation\n",
        "src/component.py": "VALUE = 1\n",
        "scripts/check.sh": "project-check\n",
    },
    "blocked_operation": {
        "README.md": "# Protected component\n",
        "protected/locked.txt": "unchanged\n",
    },
}


def build_repository(base: Path, kind: str) -> TemporaryRepository:
    """Materializa una especificación conocida debajo del tmp_path recibido."""
    if kind not in REPOSITORY_CONTENTS:
        raise ValueError(f"Tipo de repositorio desconocido: {kind}.")
    root = base / kind
    contents = dict(REPOSITORY_CONTENTS[kind])
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    memory_root = base / "persistent-store" if kind == "persistent_memory" else None
    return TemporaryRepository(kind, root, contents, memory_root)

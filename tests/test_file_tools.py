"""Tests de lectura, escritura y listado de archivos."""

from pathlib import Path

import pytest

from tools.file_tools import list_files, read_file, write_file


@pytest.fixture()
def isolated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("security.paths.WORKSPACE_ROOT", workspace)
    return workspace


def test_write_creates_parents_and_read_returns_content(
    isolated_workspace: Path,
) -> None:
    result = write_file("docs/nota.txt", "contenido")

    assert result == "Archivo escrito correctamente: docs/nota.txt"
    assert read_file("docs/nota.txt") == "contenido"


def test_write_replaces_existing_content(isolated_workspace: Path) -> None:
    write_file("nota.txt", "primero")
    write_file("nota.txt", "segundo")

    assert read_file("nota.txt") == "segundo"


def test_read_reports_missing_file(isolated_workspace: Path) -> None:
    assert read_file("inexistente.txt") == (
        "Error: el archivo 'inexistente.txt' no existe."
    )


def test_read_rejects_directory(isolated_workspace: Path) -> None:
    (isolated_workspace / "docs").mkdir()

    assert read_file("docs") == "Error: 'docs' no es un archivo."


def test_list_distinguishes_files_and_directories(isolated_workspace: Path) -> None:
    (isolated_workspace / "archivo.txt").write_text("texto", encoding="utf-8")
    (isolated_workspace / "docs").mkdir()

    assert list_files() == "[archivo] archivo.txt\n[directorio] docs"


@pytest.mark.parametrize("operation", [read_file, list_files])
def test_read_tools_control_path_escape(
    isolated_workspace: Path, operation: object
) -> None:
    result = operation("../secreto.txt")  # type: ignore[operator]

    assert result.startswith("Error")


def test_write_controls_path_escape(isolated_workspace: Path) -> None:
    result = write_file("../secreto.txt", "no escribir")

    assert result.startswith("Error al escribir")
    assert not (isolated_workspace.parent / "secreto.txt").exists()

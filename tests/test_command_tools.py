"""Tests de ejecución controlada de comandos."""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.command_tools import run_command


@pytest.fixture()
def isolated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("tools.command_tools.WORKSPACE_ROOT", workspace)
    return workspace


def test_returns_exit_code_stdout_and_stderr(isolated_workspace: Path) -> None:
    script = isolated_workspace / "result.py"
    script.write_text(
        "import sys\nprint('salida')\nprint('error', file=sys.stderr)\nsys.exit(3)\n",
        encoding="utf-8",
    )

    result = run_command(f"{sys.executable} result.py")

    assert result == {"exit_code": 3, "stdout": "salida\n", "stderr": "error\n"}


def test_runs_with_workspace_as_current_directory(
    isolated_workspace: Path,
) -> None:
    script = isolated_workspace / "cwd.py"
    script.write_text("import os\nprint(os.getcwd())\n", encoding="utf-8")

    result = run_command(f"{sys.executable} cwd.py")

    assert result["exit_code"] == 0
    assert Path(result["stdout"].strip()) == isolated_workspace


def test_handles_unknown_command(isolated_workspace: Path) -> None:
    result = run_command("comando-que-no-existe")

    assert result["exit_code"] == -1
    assert "Comando no encontrado" in result["stderr"]


def test_handles_invalid_command_syntax(isolated_workspace: Path) -> None:
    result = run_command("comando 'sin cierre")

    assert result["exit_code"] == -1
    assert "Comando inválido" in result["stderr"]


def test_handles_timeout(
    isolated_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="demo", timeout=30)

    monkeypatch.setattr("tools.command_tools.subprocess.run", raise_timeout)

    result = run_command("demo")

    assert result["exit_code"] == -1
    assert "timeout" in result["stderr"]


def test_blocks_destructive_command(isolated_workspace: Path) -> None:
    result = run_command("rm archivo.txt")

    assert result["exit_code"] == -1
    assert "destructivo" in result["stderr"]


@pytest.mark.parametrize(
    "command",
    ["cat ../secreto.txt", "cat ../.env", "cat /etc/passwd"],
)
def test_blocks_path_arguments_outside_workspace(
    isolated_workspace: Path, command: str
) -> None:
    result = run_command(command)

    assert result["exit_code"] == -1
    assert "ruta" in result["stderr"].lower()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git push", "git push"),
        ("git reset --hard", "git reset --hard"),
        ("rm -rf .", "destructivo"),
        ('bash -c "rm -rf ."', "bash -c"),
        ('sh -c "echo hola"', "sh -c"),
        ('python -c "open(\'/etc/passwd\').read()"', "python -c"),
        ('python3 -c "print(\'hola\')"', "python3 -c"),
    ],
)
def test_blocks_forbidden_command_patterns(
    isolated_workspace: Path, command: str, expected: str
) -> None:
    result = run_command(command)

    assert result["exit_code"] == -1
    assert expected in result["stderr"]


@pytest.mark.parametrize("sensitive_name", [".env", "secrets.json"])
def test_blocks_direct_sensitive_file_read(
    isolated_workspace: Path, sensitive_name: str
) -> None:
    (isolated_workspace / sensitive_name).write_text(
        "SECRET=value", encoding="utf-8"
    )

    result = run_command(f"cat {sensitive_name}")

    assert result["exit_code"] == -1
    assert "sensible" in result["stderr"]


def test_blocks_relative_symlink_that_escapes_workspace(
    isolated_workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("privado", encoding="utf-8")
    (isolated_workspace / "link.txt").symlink_to(outside)

    result = run_command("cat link.txt")

    assert result["exit_code"] == -1
    assert "fuera del workspace" in result["stderr"]


def test_allows_command_with_file_inside_workspace(isolated_workspace: Path) -> None:
    (isolated_workspace / "visible.txt").write_text("contenido", encoding="utf-8")

    result = run_command("cat visible.txt")

    assert result == {"exit_code": 0, "stdout": "contenido", "stderr": ""}


def test_executes_python_file_inside_workspace(isolated_workspace: Path) -> None:
    (isolated_workspace / "hello.py").write_text(
        "print('hola desde workspace')\n", encoding="utf-8"
    )

    result = run_command(f"{sys.executable} hello.py")

    assert result == {
        "exit_code": 0,
        "stdout": "hola desde workspace\n",
        "stderr": "",
    }

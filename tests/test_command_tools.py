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
    command = (
        f'{sys.executable} -c "import sys; print(\'salida\'); '
        "print('error', file=sys.stderr); sys.exit(3)\""
    )

    result = run_command(command)

    assert result == {"exit_code": 3, "stdout": "salida\n", "stderr": "error\n"}


def test_runs_with_workspace_as_current_directory(
    isolated_workspace: Path,
) -> None:
    result = run_command(f'{sys.executable} -c "import os; print(os.getcwd())"')

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


@pytest.mark.parametrize("command", ["cat ../secreto.txt", "cat /etc/passwd"])
def test_blocks_path_arguments_outside_workspace(
    isolated_workspace: Path, command: str
) -> None:
    result = run_command(command)

    assert result["exit_code"] == -1
    assert "ruta" in result["stderr"].lower()

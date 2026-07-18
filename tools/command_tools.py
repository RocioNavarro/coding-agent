"""Tool para ejecutar comandos controlados dentro del workspace."""

import shlex
import subprocess
from pathlib import Path
from typing import TypedDict

from core.settings import DEFAULT_SETTINGS
from security.paths import PathSecurityError, WORKSPACE_ROOT, resolve_workspace_path


COMMAND_TIMEOUT_SECONDS = DEFAULT_SETTINGS.command_timeout_seconds
DESTRUCTIVE_COMMANDS = {
    "dd",
    "mkfs",
    "rmdir",
    "rm",
    "shred",
}


class CommandResult(TypedDict):
    """Resultado estable de la ejecución de un comando."""

    exit_code: int
    stdout: str
    stderr: str


def _error_result(message: str) -> CommandResult:
    """Construye un resultado de error sin lanzar excepciones al llamador."""
    return {"exit_code": -1, "stdout": "", "stderr": message}


def _validate_command(arguments: list[str]) -> None:
    """Rechaza comandos destructivos y argumentos que escapen del workspace."""
    executable = Path(arguments[0]).name.lower()
    if executable in DESTRUCTIVE_COMMANDS:
        raise ValueError(f"El comando destructivo '{executable}' no está permitido.")

    for argument in arguments[1:]:
        if argument.startswith("-"):
            continue

        candidate = Path(argument)
        if candidate.is_absolute() or ".." in candidate.parts:
            resolve_workspace_path(candidate)


def run_command(command: str) -> CommandResult:
    """Ejecuta un comando sin shell, con timeout y ``workspace/`` como cwd."""
    try:
        arguments = shlex.split(command)
    except ValueError as error:
        return _error_result(f"Comando inválido: {error}")

    if not arguments:
        return _error_result("El comando está vacío.")

    try:
        _validate_command(arguments)
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        completed_process = subprocess.run(
            arguments,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        return {
            "exit_code": completed_process.returncode,
            "stdout": completed_process.stdout,
            "stderr": completed_process.stderr,
        }
    except (PathSecurityError, ValueError) as error:
        return _error_result(str(error))
    except FileNotFoundError:
        return _error_result(f"Comando no encontrado: {arguments[0]}")
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        return {
            "exit_code": -1,
            "stdout": stdout if isinstance(stdout, str) else stdout.decode(errors="replace"),
            "stderr": (
                stderr if isinstance(stderr, str) else stderr.decode(errors="replace")
            )
            + f"Comando cancelado por timeout de {COMMAND_TIMEOUT_SECONDS} segundos.",
        }
    except OSError as error:
        return _error_result(f"Error al ejecutar el comando: {error}")

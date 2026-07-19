"""Política defensiva para validar comandos antes de ejecutarlos.

Esta política reduce riesgos conocidos para el trabajo práctico, pero no reemplaza
un sandbox fuerte provisto por el sistema operativo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Sequence


DESTRUCTIVE_EXECUTABLES = frozenset({"dd", "mkfs", "rmdir", "rm", "shred"})
COMMAND_WRAPPERS = frozenset(
    {
        "busybox",
        "command",
        "env",
        "find",
        "nice",
        "nohup",
        "setsid",
        "sudo",
        "su",
        "timeout",
        "xargs",
    }
)
SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
EVAL_FLAGS: dict[str, frozenset[str]] = {
    "bash": frozenset({"-c"}),
    "dash": frozenset({"-c"}),
    "node": frozenset({"-e", "--eval"}),
    "perl": frozenset({"-e"}),
    "php": frozenset({"-r"}),
    "powershell": frozenset({"-command", "-encodedcommand"}),
    "pwsh": frozenset({"-command", "-encodedcommand"}),
    "python": frozenset({"-c"}),
    "python3": frozenset({"-c"}),
    "ruby": frozenset({"-e"}),
    "sh": frozenset({"-c"}),
    "zsh": frozenset({"-c"}),
}


class CommandPolicyError(ValueError):
    """El comando infringe una regla explícita de seguridad."""


def validate_command(
    arguments: Sequence[str],
    workspace_root: Path,
    sensitive_names: Collection[str] = SENSITIVE_FILE_NAMES,
) -> None:
    """Valida el comando completo contra la política de la Parte 1."""
    if not arguments:
        raise CommandPolicyError("El comando está vacío.")

    executable = Path(arguments[0]).name.lower()
    lowered = [argument.lower() for argument in arguments[1:]]

    if executable in DESTRUCTIVE_EXECUTABLES:
        raise CommandPolicyError(
            f"El comando destructivo '{executable}' no está permitido."
        )
    if executable in COMMAND_WRAPPERS:
        raise CommandPolicyError(
            f"El wrapper de comandos '{executable}' no está permitido."
        )
    if executable == "git":
        if "push" in lowered:
            raise CommandPolicyError("La operación 'git push' no está permitida.")
        if "reset" in lowered and "--hard" in lowered:
            raise CommandPolicyError("La operación 'git reset --hard' no está permitida.")

    forbidden_flags = EVAL_FLAGS.get(executable, frozenset())
    if executable.startswith("python"):
        forbidden_flags = EVAL_FLAGS["python"]
    used_flags = forbidden_flags.intersection(lowered)
    if used_flags:
        flag = sorted(used_flags)[0]
        raise CommandPolicyError(
            f"No se permite ejecutar código inline mediante '{executable} {flag}'."
        )

    normalized_sensitive = {name.casefold() for name in sensitive_names}
    for argument in arguments[1:]:
        _validate_argument_paths(argument, workspace_root, normalized_sensitive)


def _validate_argument_paths(
    argument: str, workspace_root: Path, sensitive_names: set[str]
) -> None:
    """Revisa rutas directas y valores de opciones como ``--file=/ruta``."""
    candidate_text = argument.split("=", 1)[1] if "=" in argument else argument
    if not candidate_text or (argument.startswith("-") and "=" not in argument):
        return

    candidate = Path(candidate_text)

    resolved_workspace = workspace_root.resolve()
    resolved_candidate = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_workspace / candidate).resolve()
    )
    try:
        resolved_candidate.relative_to(resolved_workspace)
    except ValueError as error:
        raise CommandPolicyError(
            f"La ruta '{candidate_text}' está fuera del workspace."
        ) from error

    if any(part.casefold() in sensitive_names for part in candidate.parts):
        raise CommandPolicyError(
            f"El acceso al archivo sensible '{candidate_text}' no está permitido."
        )

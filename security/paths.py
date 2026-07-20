"""Validación de rutas confinadas al workspace del proyecto."""

from pathlib import Path

from security.command_policy import SENSITIVE_FILE_EXTENSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
FORBIDDEN_NAMES = {".env"}


class PathSecurityError(ValueError):
    """Indica que una ruta no cumple las restricciones del workspace."""


def resolve_workspace_path(path: str | Path) -> Path:
    """Resuelve una ruta relativa y comprueba que permanezca en el workspace.

    Args:
        path: Ruta relativa a ``workspace/``.

    Returns:
        La ruta absoluta y normalizada dentro del workspace.

    Raises:
        PathSecurityError: Si la ruta es absoluta, escapa del workspace o
            intenta acceder a un archivo ``.env``.
    """
    relative_path = Path(path)

    if relative_path.is_absolute():
        raise PathSecurityError("Las rutas absolutas no están permitidas.")

    if any(part == ".env" for part in relative_path.parts):
        raise PathSecurityError("El acceso a archivos .env no está permitido.")

    resolved_path = (WORKSPACE_ROOT / relative_path).resolve()

    try:
        resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise PathSecurityError(
            f"La ruta '{path}' está fuera del workspace."
        ) from error

    if resolved_path.name in FORBIDDEN_NAMES:
        raise PathSecurityError("El acceso a archivos .env no está permitido.")

    if resolved_path.suffix.casefold() in SENSITIVE_FILE_EXTENSIONS:
        raise PathSecurityError(
            f"El acceso al archivo sensible '{resolved_path.name}' no está permitido."
        )

    return resolved_path


def relative_to_workspace(path: Path) -> Path:
    """Devuelve una ruta relativa al workspace después de validarla."""
    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise PathSecurityError(
            f"La ruta '{path}' está fuera del workspace."
        ) from error

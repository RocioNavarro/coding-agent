"""Tools para operar con archivos dentro del workspace."""

from security.paths import (
    PathSecurityError,
    relative_to_workspace,
    resolve_workspace_path,
)


def read_file(path: str) -> str:
    """Lee como UTF-8 un archivo relativo al workspace."""
    try:
        file_path = resolve_workspace_path(path)

        if not file_path.exists():
            return f"Error: el archivo '{path}' no existe."
        if not file_path.is_file():
            return f"Error: '{path}' no es un archivo."

        return file_path.read_text(encoding="utf-8")
    except (PathSecurityError, OSError, UnicodeError) as error:
        return f"Error al leer '{path}': {error}"


def write_file(path: str, content: str) -> str:
    """Reemplaza con texto UTF-8 el contenido de un archivo del workspace."""
    try:
        file_path = resolve_workspace_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        relative_path = relative_to_workspace(file_path)
        return f"Archivo escrito correctamente: {relative_path.as_posix()}"
    except (PathSecurityError, OSError, UnicodeError) as error:
        return f"Error al escribir '{path}': {error}"


def list_files(path: str = ".") -> str:
    """Lista archivos y directorios de una ruta relativa al workspace."""
    try:
        directory = resolve_workspace_path(path)

        if not directory.exists():
            return f"Error: el directorio '{path}' no existe."
        if not directory.is_dir():
            return f"Error: '{path}' no es un directorio."

        entries = []
        for item in sorted(directory.iterdir(), key=lambda entry: entry.name):
            item_type = "directorio" if item.is_dir() else "archivo"
            relative_path = relative_to_workspace(item)
            entries.append(f"[{item_type}] {relative_path.as_posix()}")

        return "\n".join(entries) if entries else "El directorio está vacío."
    except (PathSecurityError, OSError) as error:
        return f"Error al listar '{path}': {error}"

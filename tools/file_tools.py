from pathlib import Path

# Carpeta raíz donde el agente puede trabajar
WORKSPACE = Path("workspace").resolve()


def resolve_safe_path(relative_path: str) -> Path:
    """
    Convierte un path relativo en un path absoluto dentro de workspace.

    También evita que el agente pueda acceder a archivos
    ubicados fuera de la carpeta workspace.
    """
    target = (WORKSPACE / relative_path).resolve()

    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise PermissionError(
            f"Acceso bloqueado: '{relative_path}' está fuera de workspace."
        )

    return target


def read_file(path: str) -> str:
    file_path = resolve_safe_path(path)

    if not file_path.exists():
        return f"Error: el archivo '{path}' no existe."

    if not file_path.is_file():
        return f"Error: '{path}' no es un archivo."

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as error:
        return f"Error al leer el archivo: {error}"


def write_file(path: str, content: str) -> str:
    file_path = resolve_safe_path(path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        return f"Archivo escrito correctamente: {path}"

    except Exception as error:
        return f"Error al escribir el archivo: {error}"


def list_files(path: str = ".") -> str:
    directory = resolve_safe_path(path)

    if not directory.exists():
        return f"Error: el directorio '{path}' no existe."

    if not directory.is_dir():
        return f"Error: '{path}' no es un directorio."

    try:
        results = []

        for item in sorted(directory.iterdir()):
            item_type = "directorio" if item.is_dir() else "archivo"
            relative_item = item.relative_to(WORKSPACE)

            results.append(
                f"[{item_type}] {relative_item}"
            )

        if not results:
            return "El directorio está vacío."

        return "\n".join(results)

    except Exception as error:
        return f"Error al listar archivos: {error}"

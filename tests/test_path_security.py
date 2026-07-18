"""Tests de confinamiento de rutas dentro del workspace."""

from pathlib import Path

import pytest

from security.paths import PathSecurityError, WORKSPACE_ROOT, resolve_workspace_path


def test_resolves_relative_path_inside_workspace() -> None:
    resolved = resolve_workspace_path("carpeta/archivo.txt")

    assert resolved == (WORKSPACE_ROOT / "carpeta/archivo.txt").resolve()


@pytest.mark.parametrize("path", ["../archivo", "../../.env", "/etc/passwd"])
def test_rejects_paths_outside_workspace(path: str) -> None:
    with pytest.raises(PathSecurityError):
        resolve_workspace_path(path)


def test_rejects_env_file_inside_workspace() -> None:
    with pytest.raises(PathSecurityError, match=r"\.env"):
        resolve_workspace_path("config/.env")


def test_rejects_symlink_that_escapes_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to(tmp_path)
    monkeypatch.setattr("security.paths.WORKSPACE_ROOT", workspace.resolve())

    with pytest.raises(PathSecurityError):
        resolve_workspace_path("escape/secret.txt")

"""Memoria persistente, local, auditable y aislada por workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.researcher import EvidenceFragment, ProjectMemoryProvider
from core.task_state import TaskState
from core.observability import NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation


MEMORY_SCHEMA_VERSION = 1
SECRET_KEY = re.compile(
    r"(?:api[_-]?key|password|passwd|secret|token|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|token|authorization|credential)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
SECRET_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[opsu]_[A-Za-z0-9]{12,}|"
    r"AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._~+/=-]{12,})\b",
    re.IGNORECASE,
)
SENSITIVE_PATH = re.compile(
    r"(?:^|/)(?:\.env(?:\..*)?|[^/]*(?:secret|credential)[^/]*)$",
    re.IGNORECASE,
)


class ProjectMemoryError(RuntimeError):
    """Error controlado de persistencia de memoria."""


class MemoryCorruptionError(ProjectMemoryError):
    """El archivo existe pero no cumple el contrato de memoria."""


def _empty_data(workspace: str, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "workspace": workspace,
        "workspace_id": workspace_id,
        "project_summary": "",
        "technologies": [],
        "architecture": "",
        "modules": [],
        "important_files": [],
        "dependencies": [],
        "known_commands": [],
        "conventions": [],
        "decisions": [],
        "bugs": [],
        "frequent_errors": [],
        "previous_tasks": [],
        "modified_files": [],
        "session_summaries": [],
        "updated_at": None,
    }


class ProjectMemory(ProjectMemoryProvider):
    """Almacén JSON por workspace y proveedor consultable por Researcher."""

    _LIST_FIELDS = frozenset(
        {
            "technologies", "modules", "important_files", "dependencies",
            "known_commands", "conventions", "decisions", "bugs",
            "frequent_errors", "previous_tasks", "modified_files",
            "session_summaries",
        }
    )

    def __init__(
        self,
        workspace: str | Path,
        *,
        identifier: str | None = None,
        storage_root: str | Path | None = None,
        observability: ObservabilityClient | None = None,
    ) -> None:
        root = Path(workspace).resolve()
        if not root.is_dir():
            raise ValueError("workspace debe ser un directorio existente.")
        self.workspace = root
        canonical = root.as_posix()
        stable_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        self.workspace_id = self._safe_identifier(identifier) if identifier else stable_hash
        base = Path(storage_root).resolve() if storage_root else root / ".coding-agent" / "memory"
        self.storage_root = base
        self.path = base / f"{self.workspace_id}.json"
        self._data = _empty_data(canonical, self.workspace_id)
        self._loaded = False
        self.observability = observability or NoOpObservabilityClient()

    @staticmethod
    def _safe_identifier(identifier: str) -> str:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier no puede estar vacío.")
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", identifier.strip()).strip(".-")
        if not normalized:
            raise ValueError("identifier no contiene caracteres utilizables.")
        return normalized[:80]

    @property
    def data(self) -> dict[str, Any]:
        """Devuelve una copia para auditoría sin permitir mutaciones desordenadas."""
        return deepcopy(self._data)

    def load(self) -> ProjectMemory:
        """Carga la memoria; un archivo ausente equivale a memoria vacía."""
        if not self.path.exists():
            self._data = _empty_data(self.workspace.as_posix(), self.workspace_id)
            self._loaded = True
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_data(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise MemoryCorruptionError(
                f"La memoria '{self.path}' está corrupta: {error}"
            ) from error
        expected_workspace = self._sanitize(self.workspace.as_posix())
        if raw["workspace"] != expected_workspace or raw["workspace_id"] != self.workspace_id:
            raise MemoryCorruptionError("La identidad de workspace de la memoria no coincide.")
        self._data = raw
        self._loaded = True
        return self

    def save(self) -> None:
        """Persiste JSON redactado mediante reemplazo atómico en el mismo directorio."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        safe = self._sanitize(self._data)
        safe["updated_at"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.workspace_id}.", suffix=".tmp", dir=self.storage_root
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            temporary = None
        except OSError as error:
            raise ProjectMemoryError(f"No se pudo guardar la memoria: {error}") from error
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass
        self._data = safe
        self._loaded = True

    def update_project_summary(self, summary: str) -> None:
        self._set_text("project_summary", summary)

    def update_architecture(self, architecture: str) -> None:
        self._set_text("architecture", architecture)

    def add_technology(self, technology: str) -> None:
        self._append_unique("technologies", technology)

    def add_decision(self, decision: str) -> None:
        self._append_record("decisions", {"decision": decision})

    def add_known_command(self, command: str, evidence: str | None = None) -> None:
        record = {"command": command}
        if evidence:
            record["evidence"] = evidence
        self._append_record("known_commands", record)

    def add_bug(self, bug: str, resolution: str | None = None) -> None:
        record = {"bug": bug}
        if resolution:
            record["resolution"] = resolution
        self._append_record("bugs", record)

    def save_task_summary(self, state: TaskState) -> None:
        """Registra una tarea terminada y los artefactos auditables asociados."""
        if not isinstance(state, TaskState):
            raise TypeError("state debe ser TaskState.")
        task = {
            "task_id": state.task_id,
            "request": state.original_request,
            "status": state.current_status,
            "result": state.final_result,
            "modified_files": list(state.files_modified),
            "errors": [error.message for error in state.errors],
            "warnings": list(state.warnings),
        }
        self._append_record("previous_tasks", task, identity="task_id")
        for path in state.files_modified:
            self._append_unique("modified_files", path, path=True)
        for error in state.errors:
            self._append_unique("frequent_errors", error.message)
        self._append_record(
            "session_summaries",
            {
                "task_id": state.task_id,
                "summary": state.final_result or state.current_status,
                "agents": [result.subagent_id for result in state.subagent_results],
            },
            identity="task_id",
        )
        emit_observation(
            self.observability,
            ObservabilityEvent(
                "agent", "memory-summary-write", task_id=state.task_id,
                payload={"status": state.current_status,
                         "modified_file_count": len(state.files_modified),
                         "error_count": len(state.errors)},
            ),
        )

    def search_relevant_memory(
        self, query: str, *, limit: int = 5
    ) -> tuple[EvidenceFragment, ...]:
        """Recupera entradas por coincidencia léxica simple y trazable."""
        if limit < 1:
            return ()
        if not self._loaded:
            self.load()
        tokens = self._tokens(query)
        candidates: list[tuple[int, str, str]] = []
        for field in ("project_summary", "architecture"):
            value = self._data[field]
            if value:
                candidates.append((self._score(value, tokens), field, value))
        for field in sorted(self._LIST_FIELDS):
            for index, value in enumerate(self._data[field]):
                text = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
                candidates.append((self._score(text, tokens), f"{field}/{index}", text))
        positive = [item for item in candidates if item[0] > 0]
        selected = sorted(positive, key=lambda item: (-item[0], item[1]))[:limit]
        results = tuple(
            EvidenceFragment(
                "project_memory",
                f"memory://{self.workspace_id}/{reference}",
                content,
                min(1.0, 0.5 + score * 0.1),
            )
            for score, reference, content in selected
        )
        emit_observation(
            self.observability,
            ObservabilityEvent(
                "agent", "memory-search",
                payload={"query": query, "result_count": len(results),
                         "used_entries": [item.reference for item in results]},
            ),
        )
        return results

    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        """Implementa el contrato existente de ProjectMemoryProvider."""
        return self.search_relevant_memory(query, limit=limit)

    def add_module(self, module: str) -> None:
        self._append_unique("modules", module, path=True)

    def add_important_file(self, path: str) -> None:
        self._append_unique("important_files", path, path=True)

    def add_dependency(self, dependency: str) -> None:
        self._append_unique("dependencies", dependency)

    def add_convention(self, convention: str) -> None:
        self._append_unique("conventions", convention)

    def _set_text(self, field: str, value: str) -> None:
        self._data[field] = self._safe_text(value)

    def _append_unique(self, field: str, value: str, *, path: bool = False) -> None:
        safe = self._safe_text(value)
        if path and SENSITIVE_PATH.search(safe.replace("\\", "/")):
            return
        if safe not in self._data[field]:
            self._data[field].append(safe)

    def _append_record(
        self, field: str, record: Mapping[str, Any], *, identity: str | None = None
    ) -> None:
        safe = self._sanitize(dict(record))
        if identity:
            self._data[field] = [
                item for item in self._data[field]
                if not isinstance(item, dict) or item.get(identity) != safe.get(identity)
            ]
        if safe not in self._data[field]:
            self._data[field].append(safe)

    @classmethod
    def _sanitize(cls, value: Any, key: str = "") -> Any:
        if key and SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(item_key): cls._sanitize(item, str(item_key)) for item_key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, str):
            redacted = SECRET_TOKEN.sub("[REDACTED]", value)
            return SECRET_ASSIGNMENT.sub(
                lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
                redacted,
            )
        return value

    @classmethod
    def _safe_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("El valor de memoria no puede estar vacío.")
        return cls._sanitize(value.strip())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[\w.-]{3,}", text.casefold())}

    @classmethod
    def _score(cls, text: str, tokens: set[str]) -> int:
        candidate = cls._tokens(text)
        return len(tokens.intersection(candidate)) if tokens else 1

    @classmethod
    def _validate_data(cls, data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("la raíz debe ser un objeto")
        required = set(_empty_data("", ""))
        if set(data) != required:
            raise ValueError("el esquema no contiene exactamente los campos requeridos")
        if data["schema_version"] != MEMORY_SCHEMA_VERSION:
            raise ValueError("versión de esquema no soportada")
        for field in cls._LIST_FIELDS:
            if not isinstance(data[field], list):
                raise ValueError(f"{field} debe ser una lista")
        for field in ("workspace", "workspace_id", "project_summary", "architecture"):
            if not isinstance(data[field], str):
                raise ValueError(f"{field} debe ser texto")
        if data["updated_at"] is not None and not isinstance(data["updated_at"], str):
            raise ValueError("updated_at debe ser texto o null")

"""Implementer genérico para propuestas y reemplazos localizados con políticas."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal, Mapping, Sequence

from agents.base import AgentContext, AgentExecutionError, AgentInput, BaseAgent
from agents.context_manager import ContextManager, StateContextManager
from core.llm_client import LLMClient
from core.task_state import ErrorRecord, SourceReference, SubagentResult, TaskState
from core.models import EvidenceAssessment
from security.evidence_policy import EvidenceContext, EvidenceSufficiencyPolicy
from tools.registry import ToolRegistry


ImplementationMode = Literal["propose_only", "apply_changes"]
LOCK_FILE_NAMES = frozenset(
    {
        "Cargo.lock", "Gemfile.lock", "composer.lock", "package-lock.json",
        "pnpm-lock.yaml", "poetry.lock", "yarn.lock",
    }
)
DEFAULT_PROTECTED_PATTERNS = (
    ".env", ".env.*", ".git/*", "**/.git/*", "*credentials*", "*secret*"
)
MAX_FRAGMENT_CHARACTERS = 12_000


IMPLEMENTER_SYSTEM_PROMPT = """Sos Implementer, un agente de cambios localizados.
Trabajá únicamente con el pedido, plan aprobado, fragmentos seleccionados,
convenciones, evidencia técnica y políticas incluidas. No inventes APIs ni supongas
contenido ausente. Proponé reemplazos exactos y mínimos sobre archivos autorizados.
No reescribas archivos completos, no modifiques locks sin autorización, no crees
templates por lenguaje y no ejecutes tests: esa responsabilidad pertenece a Tester.
Respondé sólo con JSON: summary, proposed_change, conventions_check (lista), changes
(lista de objetos path, old_text, new_text, explanation), findings, recommendations,
sources, files_relevant, blockers y confidence."""


class ImplementerBlockedError(AgentExecutionError):
    """Indica que faltan precondiciones para proponer o aplicar cambios."""


@dataclass(frozen=True)
class WriteDecision:
    allowed: bool
    reason: str


class WritePolicy(ABC):
    """Contrato de autorización previo a cualquier propuesta o escritura."""

    @abstractmethod
    def evaluate(self, path: str, allowed_files: Sequence[str]) -> WriteDecision:
        """Decide si una ruta pertenece al alcance y puede modificarse."""

    @abstractmethod
    def resolve(self, path: str) -> Path:
        """Resuelve una ruta ya validada dentro del workspace configurado."""


class ScopedWritePolicy(WritePolicy):
    """Política confinada por raíz, alcance, protegidos y locks autorizados."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
        authorized_lock_files: Sequence[str] = (),
    ) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root debe ser un directorio existente.")
        self._root = root
        self._protected_patterns = tuple(protected_patterns)
        self._authorized_locks = frozenset(authorized_lock_files)

    def evaluate(self, path: str, allowed_files: Sequence[str]) -> WriteDecision:
        try:
            self.resolve(path)
        except ValueError as error:
            return WriteDecision(False, str(error))
        normalized = Path(path).as_posix()
        if normalized not in set(allowed_files):
            return WriteDecision(False, "El archivo está fuera del alcance seleccionado.")
        if any(fnmatch(normalized, pattern) for pattern in self._protected_patterns):
            return WriteDecision(False, "El archivo está protegido por política.")
        if Path(normalized).name in LOCK_FILE_NAMES and normalized not in self._authorized_locks:
            return WriteDecision(False, "El lock file requiere autorización explícita.")
        return WriteDecision(True, "Escritura autorizada.")

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or any(part in {"..", ".env"} for part in candidate.parts):
            raise ValueError("La ruta no está permitida.")
        resolved = (self._root / candidate).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise ValueError("La ruta escapa del repositorio.") from error
        return resolved


@dataclass(frozen=True)
class FileFragment:
    path: str
    content: str


@dataclass(frozen=True)
class LocalizedChange:
    path: str
    old_text: str
    new_text: str
    explanation: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.path, self.old_text, self.explanation)
        ) or not isinstance(self.new_text, str):
            raise ValueError("Los campos del cambio localizado no pueden estar vacíos.")
        if self.old_text == self.new_text:
            raise ValueError("El cambio debe modificar el texto seleccionado.")


@dataclass(frozen=True)
class ImplementerResult:
    mode: ImplementationMode
    proposed_change: str
    conventions_check: tuple[str, ...]
    changes: tuple[LocalizedChange, ...]
    files_read: tuple[str, ...]
    files_modified: tuple[str, ...]
    summary: str
    subagent_result: SubagentResult


class ImplementerAgent(BaseAgent):
    """Genera y opcionalmente aplica reemplazos exactos en archivos autorizados."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        write_policy: WritePolicy,
        context_manager: ContextManager | None = None,
        minimum_evidence_confidence: float = 0.5,
        evidence_policy: EvidenceSufficiencyPolicy | None = None,
        name: str = "implementer",
    ) -> None:
        super().__init__(
            name=name,
            role="Localized Change Implementer",
            system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
            allowed_tools=(),
            llm_client=llm_client,
        )
        if not 0 <= minimum_evidence_confidence <= 1:
            raise ValueError("minimum_evidence_confidence debe estar entre 0 y 1.")
        self.write_policy = write_policy
        self.context_manager = context_manager or StateContextManager()
        self.minimum_evidence_confidence = minimum_evidence_confidence
        self.evidence_policy = evidence_policy or EvidenceSufficiencyPolicy()

    def specialization_prompt(self) -> str:
        return "No produzcas código fuera de los fragmentos y archivos autorizados."

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
        *,
        mode: ImplementationMode = "propose_only",
    ) -> ImplementerResult:
        if mode not in {"propose_only", "apply_changes"}:
            raise ValueError("mode debe ser propose_only o apply_changes.")
        try:
            self._verify_preconditions(task_state, mode)
            selected = self.context_manager.select(instruction, task_state, context)
            if not selected.files:
                raise ImplementerBlockedError(
                    "Explorer y ContextManager no seleccionaron archivos relevantes."
                )
            fragments = self._read_fragments(selected.files, instruction, task_state)
            response = self.llm_client.complete(
                self._build_implementation_context(
                    instruction, task_state, selected, fragments, mode
                ),
                (),
            )
            if response.tool_calls:
                raise AgentExecutionError("Implementer no acepta tool calls del LLM.")
            parsed = self._parse_response(response.text)
            changes = parsed["changes"]
            self._validate_changes(changes, selected.files, fragments)
            modified = (
                self._apply_changes(changes, fragments, task_state)
                if mode == "apply_changes"
                else ()
            )
            subagent_result = self._to_subagent_result(
                instruction, parsed, changes, modified
            )
        except Exception as error:
            controlled = error if isinstance(error, AgentExecutionError) else AgentExecutionError(
                f"El agente '{self.name}' no pudo implementar: {error}"
            )
            task_state.record_error(
                ErrorRecord(str(controlled), task_state.current_phase, self.name, True)
            )
            if controlled is error:
                raise
            raise controlled from error

        task_state.add_subagent_result(subagent_result)
        task_state.add_observation(
            f"Implementer modo={mode}; cambios propuestos={len(changes)}; "
            f"archivos modificados={len(modified)}."
        )
        return ImplementerResult(
            mode=mode,
            proposed_change=parsed["proposed_change"],
            conventions_check=parsed["conventions_check"],
            changes=changes,
            files_read=tuple(fragment.path for fragment in fragments),
            files_modified=modified,
            summary=parsed["summary"],
            subagent_result=subagent_result,
        )

    def assess_evidence(
        self,
        instruction: str,
        state: TaskState,
        *,
        validation_available: bool,
    ) -> EvidenceAssessment:
        """Recolecta hechos disponibles y delega la clasificación a la política."""
        selected = self.context_manager.select(instruction, state)
        targets = selected.files
        existing: list[str] = []
        policy_risks: list[str] = []
        permissions_granted = bool(targets)
        for path in targets:
            decision = self.write_policy.evaluate(path, targets)
            if not decision.allowed:
                permissions_granted = False
                policy_risks.append(decision.reason)
            try:
                if self.write_policy.resolve(path).is_file():
                    existing.append(path)
            except ValueError as error:
                permissions_granted = False
                policy_risks.append(str(error))

        evidence_text = tuple(
            filter(
                None,
                (
                    *state.repository_findings,
                    *(result.summary for result in state.subagent_results),
                    *(item for result in state.subagent_results for item in result.findings),
                    *(item for result in state.subagent_results for item in result.blockers),
                    *state.warnings,
                ),
            )
        )
        lowered = tuple((item, item.casefold()) for item in evidence_text)
        conventions = tuple(
            item for item, text in lowered
            if any(marker in text for marker in ("convención", "convention", "arquitectura"))
        )
        impacts = tuple(item for item, text in lowered if "impact" in text)
        ambiguities = tuple(item for item, text in lowered if "ambig" in text)
        contradictions = tuple(item for item, text in lowered if "contradic" in text)
        detected_risks = tuple(item for item, text in lowered if "riesgo" in text)
        excessive = any("excesiv" in item.casefold() for item in detected_risks)
        context = EvidenceContext(
            component=", ".join(targets),
            expected_behavior="\n".join(
                filter(None, (state.original_request, state.approved_plan or ""))
            ),
            conventions=conventions,
            impact=impacts,
            validation_methods=("Tester configurado",) if validation_available else (),
            permissions_granted=permissions_granted,
            target_files=targets,
            existing_files=tuple(existing),
            supporting_sources=tuple(source.reference for source in state.sources),
            ambiguities=ambiguities,
            contradictions=contradictions,
            risks=(*detected_risks, *policy_risks),
            risk_level="excessive" if excessive else "moderate" if detected_risks else "low",
        )
        return self.evidence_policy.evaluate(context)

    def _verify_preconditions(self, state: TaskState, mode: ImplementationMode) -> None:
        if not state.approved_plan:
            raise ImplementerBlockedError("Implementer requiere un plan aprobado.")
        if mode == "apply_changes" and not state.has_current_sufficient_evidence:
            raise ImplementerBlockedError(
                "Implementer requiere un EvidenceAssessment sufficient vigente."
            )
        research_results = tuple(
            result for result in state.subagent_results if result.subagent_id == "researcher"
        )
        sufficient = any(
            result.status == "completed"
            and result.confidence is not None
            and result.confidence >= self.minimum_evidence_confidence
            and bool(result.sources)
            for result in research_results
        )
        if not sufficient:
            raise ImplementerBlockedError("Implementer requiere evidencia técnica suficiente.")

    def _read_fragments(
        self, paths: Sequence[str], instruction: str, state: TaskState
    ) -> tuple[FileFragment, ...]:
        fragments: list[FileFragment] = []
        keywords = {
            token.casefold().strip(".,:;()[]{}")
            for token in instruction.split()
            if len(token) >= 3
        }
        for relative in paths:
            decision = self.write_policy.evaluate(relative, paths)
            if not decision.allowed:
                raise AgentExecutionError(f"Política rechazó '{relative}': {decision.reason}")
            path = self.write_policy.resolve(relative)
            if not path.is_file():
                raise AgentExecutionError(f"El archivo relevante '{relative}' no existe.")
            content = path.read_text(encoding="utf-8")
            selected = self._select_fragment(content, keywords)
            fragments.append(FileFragment(relative, selected))
            state.record_file_read(relative)
        return tuple(fragments)

    @staticmethod
    def _select_fragment(content: str, keywords: set[str]) -> str:
        if len(content) <= MAX_FRAGMENT_CHARACTERS:
            return content
        lines = content.splitlines(keepends=True)
        matching = [
            index
            for index, line in enumerate(lines)
            if any(keyword in line.casefold() for keyword in keywords)
        ]
        if not matching:
            return "".join(lines[:100])[:MAX_FRAGMENT_CHARACTERS]
        indexes = sorted(
            {line for match in matching[:20] for line in range(max(0, match - 3), min(len(lines), match + 4))}
        )
        return "".join(lines[index] for index in indexes)[:MAX_FRAGMENT_CHARACTERS]

    def _build_implementation_context(
        self,
        instruction: str,
        state: TaskState,
        selected: AgentContext,
        fragments: Sequence[FileFragment],
        mode: ImplementationMode,
    ) -> list:
        architecture_and_conventions = tuple(
            finding
            for finding in state.repository_findings
            if any(
                marker in finding.casefold()
                for marker in ("arquitectura", "architecture", "convention", "convención")
            )
        )
        evidence = tuple(
            source.to_dict()
            for source in state.sources
            if source.origin in {"project_memory", "rag", "web", "repository"}
        )
        facts = (
            f"Modo: {mode}",
            f"Pedido original: {state.original_request}",
            f"Plan aprobado: {state.approved_plan}",
            "Arquitectura y convenciones detectadas: "
            f"{json.dumps(architecture_and_conventions, ensure_ascii=False)}",
            f"Evidencia técnica: {json.dumps(evidence, ensure_ascii=False)}",
            f"Fragmentos autorizados: {json.dumps([fragment.__dict__ for fragment in fragments], ensure_ascii=False)}",
            f"Política efectiva: sólo se permiten cambios en {json.dumps(list(selected.files), ensure_ascii=False)}.",
        )
        agent_input = AgentInput(
            instruction,
            state.task_id,
            AgentContext(
                facts=(*selected.facts, *facts),
                sources=selected.sources,
                files=selected.files,
                constraints=(
                    *selected.constraints,
                    "Usar reemplazos exactos, mínimos y dentro de los fragmentos.",
                ),
            ),
        )
        return self.build_context(agent_input)

    @staticmethod
    def _parse_response(text: str) -> dict:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise AgentExecutionError("Implementer devolvió JSON inválido.") from error
        if not isinstance(payload, dict):
            raise AgentExecutionError("La respuesta de Implementer debe ser un objeto.")
        required_text = ("summary", "proposed_change")
        for name in required_text:
            if not isinstance(payload.get(name), str) or not payload[name].strip():
                raise AgentExecutionError(f"El campo '{name}' debe ser texto no vacío.")
        conventions = payload.get("conventions_check")
        raw_changes = payload.get("changes")
        if not isinstance(conventions, list) or not all(
            isinstance(item, str) and item.strip() for item in conventions
        ) or not conventions:
            raise AgentExecutionError("conventions_check debe incluir al menos una comprobación.")
        if not isinstance(raw_changes, list):
            raise AgentExecutionError("changes debe ser una lista.")
        try:
            changes = tuple(
                LocalizedChange(
                    item["path"], item["old_text"], item["new_text"], item["explanation"]
                )
                for item in raw_changes
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AgentExecutionError(f"Cambio localizado inválido: {error}") from error
        return {
            "summary": payload["summary"].strip(),
            "proposed_change": payload["proposed_change"].strip(),
            "conventions_check": tuple(item.strip() for item in conventions),
            "changes": changes,
        }

    def _validate_changes(
        self,
        changes: Sequence[LocalizedChange],
        allowed_files: Sequence[str],
        fragments: Sequence[FileFragment],
    ) -> None:
        fragment_by_path = {fragment.path: fragment.content for fragment in fragments}
        for change in changes:
            decision = self.write_policy.evaluate(change.path, allowed_files)
            if not decision.allowed:
                raise AgentExecutionError(
                    f"Política rechazó '{change.path}': {decision.reason}"
                )
            fragment = fragment_by_path.get(change.path)
            if fragment is None or change.old_text not in fragment:
                raise AgentExecutionError(
                    f"El cambio de '{change.path}' no pertenece al fragmento autorizado."
                )
            current = self.write_policy.resolve(change.path).read_text(encoding="utf-8")
            if current.count(change.old_text) != 1:
                raise AgentExecutionError(
                    f"El texto objetivo de '{change.path}' debe aparecer exactamente una vez."
                )
            if change.old_text == current:
                raise AgentExecutionError("No se permite reescribir el archivo completo.")

    def _apply_changes(
        self,
        changes: Sequence[LocalizedChange],
        fragments: Sequence[FileFragment],
        state: TaskState,
    ) -> tuple[str, ...]:
        updated: dict[str, str] = {}
        for change in changes:
            current = updated.get(change.path)
            if current is None:
                current = self.write_policy.resolve(change.path).read_text(encoding="utf-8")
            updated[change.path] = current.replace(change.old_text, change.new_text, 1)
        for path, content in updated.items():
            self.write_policy.resolve(path).write_text(content, encoding="utf-8")
            state.record_file_modified(path)
        return tuple(updated)

    def _to_subagent_result(
        self,
        instruction: str,
        parsed: Mapping[str, object],
        changes: Sequence[LocalizedChange],
        modified: Sequence[str],
    ) -> SubagentResult:
        summary = str(parsed["summary"])
        return SubagentResult(
            self.name,
            instruction,
            "completed",
            result=summary,
            summary=summary,
            findings=tuple(change.explanation for change in changes),
            recommendations=(str(parsed["proposed_change"]),),
            files_relevant=tuple(dict.fromkeys(change.path for change in changes)),
            confidence=None,
        )

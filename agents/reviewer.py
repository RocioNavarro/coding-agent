"""Reviewer genérico basado en diff, evidencia, alcance y validaciones previas."""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from agents.base import AgentContext, AgentExecutionError, AgentInput, BaseAgent
from core.llm_client import LLMClient
from core.task_state import SourceReference, SubagentResult, TaskState
from tools.registry import ToolRegistry


ReviewDecision = Literal[
    "approved", "changes_requested", "blocked", "insufficient_evidence"
]
ReviewSeverity = Literal["none", "low", "medium", "high", "critical"]
SEVERITY_ORDER: dict[ReviewSeverity, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
MAX_DIFF_CHARACTERS = 40_000


REVIEWER_SYSTEM_PROMPT = """Sos Reviewer, un revisor técnico independiente del lenguaje.
Evaluá sólo el pedido, plan, arquitectura, convenciones, diff, alcance, fuentes y
validaciones provistos. No agregues reglas propias de un ecosistema. Identificá
cambios innecesarios, riesgos, regresiones y evidencia faltante. No modifiques
archivos ni solicites tools. Respondé únicamente JSON con decision (approved,
changes_requested, blocked o insufficient_evidence), summary, issues (lista de
objetos code, message, severity, evidence), severity, required_changes,
optional_suggestions y confidence."""


@dataclass(frozen=True)
class DiffSnapshot:
    patch: str
    files: tuple[str, ...]
    available: bool = True
    error: str | None = None


class DiffProvider(ABC):
    """Proveedor de diff de sólo lectura."""

    @abstractmethod
    def get_diff(self, modified_files: Sequence[str]) -> DiffSnapshot:
        """Obtiene el diff limitado a archivos declarados como modificados."""


class GitDiffProvider(DiffProvider):
    """Obtiene un diff sin hooks, shell ni operaciones mutadoras."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        timeout_seconds: float = 10.0,
        max_characters: int = MAX_DIFF_CHARACTERS,
    ) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root debe ser un directorio existente.")
        self._root = root
        self._timeout = timeout_seconds
        self._max_characters = max_characters

    def get_diff(self, modified_files: Sequence[str]) -> DiffSnapshot:
        if not modified_files:
            return DiffSnapshot("", ())
        try:
            completed = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--no-color", "--", *modified_files],
                cwd=self._root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return DiffSnapshot("", tuple(modified_files), False, str(error))
        if completed.returncode != 0:
            return DiffSnapshot(
                "", tuple(modified_files), False,
                completed.stderr.strip() or "git diff falló."
            )
        patch = completed.stdout
        if len(patch) > self._max_characters:
            patch = patch[: self._max_characters] + "\n… [diff truncado]"
        return DiffSnapshot(patch, tuple(modified_files))


@dataclass(frozen=True)
class ReviewIssue:
    code: str
    message: str
    severity: ReviewSeverity
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError("severity inválida.")
        if not self.code.strip() or not self.message.strip() or not self.evidence:
            raise ValueError("Cada issue requiere código, mensaje y evidencia.")


@dataclass(frozen=True)
class ReviewerResult:
    decision: ReviewDecision
    summary: str
    issues: tuple[ReviewIssue, ...]
    severity: ReviewSeverity
    required_changes: tuple[str, ...]
    optional_suggestions: tuple[str, ...]
    confidence: float
    diff: DiffSnapshot
    subagent_result: SubagentResult


class ReviewerAgent(BaseAgent):
    """Revisa el resultado completo sin modificar archivos ni ejecutar validaciones."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        diff_provider: DiffProvider,
        name: str = "reviewer",
    ) -> None:
        super().__init__(
            name=name,
            role="Evidence-based Change Reviewer",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=(),
            llm_client=llm_client,
        )
        self.diff_provider = diff_provider

    def specialization_prompt(self) -> str:
        return "No apruebes si falta evidencia o si Tester no ejecutó checks confiables."

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
    ) -> ReviewerResult:
        modified = task_state.files_modified
        diff = self.diff_provider.get_diff(modified)
        prerequisite = self._prerequisite_issue(task_state, diff)
        if prerequisite is not None:
            return self._finish_without_llm(
                task_state, instruction, diff, prerequisite
            )

        deterministic = self._deterministic_issues(task_state, diff)
        response = self.llm_client.complete(
            self._build_review_context(instruction, task_state, context, diff), ()
        )
        if response.tool_calls:
            raise AgentExecutionError("Reviewer no puede solicitar tools.")
        parsed = self._parse_response(response.text)
        issues = tuple((*deterministic, *parsed["issues"]))
        decision = self._enforce_decision(parsed["decision"], issues, task_state)
        required = tuple(
            dict.fromkeys(
                (*parsed["required_changes"], *(issue.message for issue in deterministic))
            )
        )
        severity = self._max_severity(issues, parsed["severity"])
        summary = parsed["summary"]
        return self._finish(
            task_state,
            instruction,
            diff,
            decision,
            summary,
            issues,
            severity,
            required,
            parsed["optional_suggestions"],
            parsed["confidence"],
        )

    @staticmethod
    def _prerequisite_issue(
        state: TaskState, diff: DiffSnapshot
    ) -> ReviewIssue | None:
        if not state.approved_plan:
            return ReviewIssue(
                "missing_plan", "No existe un plan aprobado para comparar.",
                "high", ("TaskState.approved_plan",)
            )
        if not state.files_modified:
            return ReviewIssue(
                "missing_changes", "No hay archivos modificados para revisar.",
                "medium", ("TaskState.files_modified",)
            )
        if not diff.available or not diff.patch.strip():
            return ReviewIssue(
                "missing_diff", "El diff no está disponible para la revisión.",
                "high", (diff.error or "Diff vacío",)
            )
        explorer = any(
            result.subagent_id == "explorer" and result.files_relevant
            for result in state.subagent_results
        )
        research = any(
            result.subagent_id == "researcher"
            and result.status == "completed"
            and result.sources
            for result in state.subagent_results
        )
        tester = any(result.subagent_id == "tester" for result in state.subagent_results)
        if not explorer or not research or not tester or not state.sources:
            missing = ", ".join(
                name
                for name, present in (
                    ("Explorer", explorer), ("Researcher", research),
                    ("Tester", tester), ("fuentes", bool(state.sources))
                )
                if not present
            )
            return ReviewIssue(
                "insufficient_evidence",
                f"Falta evidencia obligatoria: {missing}.",
                "high",
                ("TaskState.subagent_results", "TaskState.sources"),
            )
        return None

    @staticmethod
    def _deterministic_issues(
        state: TaskState, diff: DiffSnapshot
    ) -> tuple[ReviewIssue, ...]:
        issues: list[ReviewIssue] = []
        explorer_scope = {
            path
            for result in state.subagent_results
            if result.subagent_id == "explorer"
            for path in result.files_relevant
        }
        implementer_scope = {
            path
            for result in state.subagent_results
            if result.subagent_id == "implementer"
            for path in result.files_relevant
        }
        outside = tuple(path for path in state.files_modified if path not in explorer_scope)
        if outside:
            issues.append(
                ReviewIssue(
                    "out_of_scope",
                    "Hay archivos modificados fuera del alcance identificado por Explorer.",
                    "high",
                    outside,
                )
            )
        unrelated = tuple(
            path
            for path in state.files_modified
            if path in explorer_scope and path not in implementer_scope
        )
        if unrelated:
            issues.append(
                ReviewIssue(
                    "unrelated_files",
                    "Hay archivos modificados que Implementer no declaró necesarios.",
                    "medium",
                    unrelated,
                )
            )
        diff_missing = tuple(path for path in state.files_modified if path not in diff.files)
        if diff_missing:
            issues.append(
                ReviewIssue(
                    "diff_scope_mismatch",
                    "El diff no cubre todos los archivos registrados como modificados.",
                    "high",
                    diff_missing,
                )
            )
        tester_results = tuple(
            result for result in state.subagent_results if result.subagent_id == "tester"
        )
        latest = tester_results[-1]
        if latest.status == "failed":
            issues.append(
                ReviewIssue(
                    "tests_failed", "Tester informó validaciones fallidas.",
                    "high", latest.findings or (latest.summary or "Tester failed",)
                )
            )
        elif latest.status in {"blocked", "unavailable", "skipped"}:
            issues.append(
                ReviewIssue(
                    "validation_incomplete",
                    f"La validación terminó con estado {latest.status}.",
                    "high", latest.blockers or (latest.summary or latest.status,)
                )
            )
        return tuple(issues)

    def _build_review_context(
        self,
        instruction: str,
        state: TaskState,
        context: AgentContext | None,
        diff: DiffSnapshot,
    ) -> list:
        agent_summaries = [
            {
                "agent": result.subagent_id,
                "status": result.status,
                "summary": result.summary,
                "findings": list(result.findings),
                "files_relevant": list(result.files_relevant),
                "blockers": list(result.blockers),
                "confidence": result.confidence,
            }
            for result in state.subagent_results
        ]
        checks = [
            call.to_dict()
            for call in state.tool_calls
            if call.tool_name == "validation_command"
        ]
        selected = context or AgentContext()
        facts = (
            f"Pedido original: {state.original_request}",
            f"Plan aprobado: {state.approved_plan}",
            f"Arquitectura y convenciones: {json.dumps(state.repository_findings, ensure_ascii=False)}",
            f"Archivos modificados: {json.dumps(state.files_modified, ensure_ascii=False)}",
            f"Diff: {diff.patch}",
            f"Resultados de subagentes: {json.dumps(agent_summaries, ensure_ascii=False)}",
            f"Fuentes: {json.dumps([source.to_dict() for source in state.sources], ensure_ascii=False)}",
            f"Checks ejecutados: {json.dumps(checks, ensure_ascii=False)}",
        )
        return self.build_context(
            AgentInput(
                instruction,
                state.task_id,
                AgentContext(
                    facts=(*selected.facts, *facts),
                    sources=selected.sources,
                    files=state.files_modified,
                    constraints=(
                        *selected.constraints,
                        "No introducir reglas que no estén en las convenciones provistas.",
                    ),
                ),
            )
        )

    @staticmethod
    def _parse_response(text: str) -> dict[str, object]:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise AgentExecutionError("Reviewer devolvió JSON inválido.") from error
        if not isinstance(payload, dict):
            raise AgentExecutionError("La respuesta de Reviewer debe ser un objeto.")
        decision = payload.get("decision")
        severity = payload.get("severity")
        confidence = payload.get("confidence")
        if decision not in {"approved", "changes_requested", "blocked", "insufficient_evidence"}:
            raise AgentExecutionError("decision de Reviewer inválida.")
        if severity not in SEVERITY_ORDER:
            raise AgentExecutionError("severity de Reviewer inválida.")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise AgentExecutionError("confidence de Reviewer inválida.")
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise AgentExecutionError("summary de Reviewer no puede estar vacío.")
        issues = ReviewerAgent._parse_issues(payload.get("issues"))
        required = ReviewerAgent._text_list(payload, "required_changes")
        optional = ReviewerAgent._text_list(payload, "optional_suggestions")
        return {
            "decision": decision,
            "summary": summary.strip(),
            "issues": issues,
            "severity": severity,
            "required_changes": required,
            "optional_suggestions": optional,
            "confidence": float(confidence),
        }

    @staticmethod
    def _parse_issues(value: object) -> tuple[ReviewIssue, ...]:
        if not isinstance(value, list):
            raise AgentExecutionError("issues debe ser una lista.")
        try:
            return tuple(
                ReviewIssue(
                    item["code"], item["message"], item["severity"],
                    tuple(item["evidence"]),
                )
                for item in value
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AgentExecutionError(f"Issue de Reviewer inválido: {error}") from error

    @staticmethod
    def _text_list(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
        value = payload.get(name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise AgentExecutionError(f"{name} debe ser una lista de textos.")
        return tuple(item.strip() for item in value)

    @staticmethod
    def _enforce_decision(
        proposed: object,
        issues: Sequence[ReviewIssue],
        state: TaskState,
    ) -> ReviewDecision:
        codes = {issue.code for issue in issues}
        if "validation_incomplete" in codes:
            latest = next(
                result for result in reversed(state.subagent_results)
                if result.subagent_id == "tester"
            )
            return "blocked" if latest.status == "blocked" else "insufficient_evidence"
        if codes:
            return "changes_requested"
        return proposed  # type: ignore[return-value]

    @staticmethod
    def _max_severity(
        issues: Sequence[ReviewIssue], proposed: object
    ) -> ReviewSeverity:
        severities = [issue.severity for issue in issues]
        if proposed in SEVERITY_ORDER:
            severities.append(proposed)  # type: ignore[arg-type]
        return max(severities, key=SEVERITY_ORDER.get) if severities else "none"  # type: ignore[arg-type]

    def _finish_without_llm(
        self,
        state: TaskState,
        instruction: str,
        diff: DiffSnapshot,
        issue: ReviewIssue,
    ) -> ReviewerResult:
        decision: ReviewDecision = (
            "blocked" if issue.code == "missing_plan" else "insufficient_evidence"
        )
        return self._finish(
            state, instruction, diff, decision, issue.message, (issue,),
            issue.severity, (issue.message,), (), 0.0
        )

    def _finish(
        self,
        state: TaskState,
        instruction: str,
        diff: DiffSnapshot,
        decision: ReviewDecision,
        summary: str,
        issues: Sequence[ReviewIssue],
        severity: ReviewSeverity,
        required: Sequence[str],
        optional: Sequence[str],
        confidence: float,
    ) -> ReviewerResult:
        subagent = SubagentResult(
            self.name,
            instruction,
            decision,
            result=summary,
            summary=summary,
            findings=tuple(
                f"{issue.code}: {issue.message} Evidencia: {', '.join(issue.evidence)}"
                for issue in issues
            ),
            recommendations=tuple((*required, *optional)),
            blockers=tuple(required) if decision != "approved" else (),
            files_relevant=tuple(state.files_modified),
            confidence=confidence,
        )
        state.add_subagent_result(subagent)
        state.add_source(
            SourceReference("inference", f"reviewer:{state.task_id}", summary)
        )
        state.add_observation(
            f"Reviewer decision={decision}; severity={severity}; issues={len(issues)}."
        )
        return ReviewerResult(
            decision,
            summary,
            tuple(issues),
            severity,
            tuple(required),
            tuple(optional),
            confidence,
            diff,
            subagent,
        )

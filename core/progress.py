"""Detección configurable de loops y falta de progreso tecnológico-neutral."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


ProgressRecommendation = Literal[
    "retry_with_new_strategy", "replan", "ask_user", "stop"
]


@dataclass(frozen=True)
class ProgressLimits:
    command_error_repeats: int = 3
    read_repeats: int = 3
    search_repeats: int = 3
    modification_repeats: int = 2
    diff_repeats: int = 2
    agent_cycle_repeats: int = 3
    no_evidence_iterations: int = 3
    max_cycle_length: int = 4

    def __post_init__(self) -> None:
        for name in (
            "command_error_repeats", "read_repeats", "search_repeats",
            "modification_repeats", "diff_repeats", "agent_cycle_repeats",
            "no_evidence_iterations",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} debe ser al menos 2.")
        if self.max_cycle_length < 2:
            raise ValueError("max_cycle_length debe ser al menos 2.")


@dataclass(frozen=True)
class ProgressAssessment:
    detected: bool
    kind: str | None = None
    recommendation: ProgressRecommendation | None = None
    reason: str | None = None
    repetitions: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "kind": self.kind,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "repetitions": self.repetitions,
        }


class ProgressMonitor:
    """Observa acciones normalizadas sin conocer lenguajes ni frameworks."""

    def __init__(self, limits: ProgressLimits | None = None) -> None:
        self.limits = limits or ProgressLimits()
        self._last_signatures: dict[str, str] = {}
        self._repetitions: dict[str, int] = {}
        self._agents: list[str] = []
        self._known_evidence: set[str] = set()
        self._iterations_without_evidence = 0

    def record_tool_call(
        self,
        agent: str,
        tool: str,
        parameters: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        justification: str | None = None,
    ) -> ProgressAssessment:
        """Registra una tool call ya ejecutada y evalúa patrones repetidos."""
        actor = self._agent(agent)
        if tool == "run_command":
            error = self._execution_error(result)
            if error is not None:
                signature = self._fingerprint(actor, parameters.get("command"), error)
                return self._repeat(
                    "command_error", signature, self.limits.command_error_repeats,
                    "retry_with_new_strategy",
                    "El mismo comando produjo nuevamente el mismo error.",
                )
            self._reset("command_error")
        elif tool == "read_file":
            signature = self._fingerprint(
                actor, parameters.get("path"), justification or "", result.get("result")
            )
            return self._repeat(
                "repeated_read", signature, self.limits.read_repeats,
                "retry_with_new_strategy",
                "La misma lectura se repitió sin nueva justificación ni evidencia.",
            )
        elif tool == "web_search":
            signature = self._fingerprint(actor, parameters.get("query"))
            return self._repeat(
                "repeated_search", signature, self.limits.search_repeats,
                "retry_with_new_strategy", "La misma búsqueda se repitió.",
            )
        elif tool == "write_file":
            signature = self._fingerprint(
                actor, parameters.get("path"), parameters.get("content")
            )
            return self._repeat(
                "repeated_modification", signature,
                self.limits.modification_repeats, "replan",
                "La misma modificación se intentó nuevamente.",
            )
        return ProgressAssessment(False)

    def record_diff(
        self,
        agent: str,
        diff: str,
        *,
        evidence: Sequence[str] = (),
    ) -> ProgressAssessment:
        actor = self._agent(agent)
        normalized = "\n".join(line.rstrip() for line in diff.strip().splitlines())
        signature = self._fingerprint(actor, normalized)
        assessment = self._repeat(
            "repeated_diff", signature, self.limits.diff_repeats, "replan",
            "El diff no cambió entre intentos.",
        )
        self._remember_evidence(evidence)
        return assessment

    def record_agent(
        self, agent: str, *, evidence: Sequence[str] = ()
    ) -> ProgressAssessment:
        self._agents.append(self._agent(agent))
        maximum = self.limits.max_cycle_length * self.limits.agent_cycle_repeats
        if len(self._agents) > maximum:
            self._agents = self._agents[-maximum:]
        self._remember_evidence(evidence)
        for length in range(2, self.limits.max_cycle_length + 1):
            required = length * self.limits.agent_cycle_repeats
            if len(self._agents) < required:
                continue
            suffix = self._agents[-required:]
            pattern = suffix[:length]
            if pattern * self.limits.agent_cycle_repeats == suffix:
                return ProgressAssessment(
                    True,
                    "agent_cycle",
                    "ask_user",
                    "Se detectó un ciclo entre agentes: " + " → ".join(pattern),
                    self.limits.agent_cycle_repeats,
                )
        return ProgressAssessment(False)

    def record_iteration(
        self, *, evidence: Sequence[str] = ()
    ) -> ProgressAssessment:
        new_evidence = self._remember_evidence(evidence)
        if new_evidence:
            self._iterations_without_evidence = 0
            return ProgressAssessment(False)
        self._iterations_without_evidence += 1
        if self._iterations_without_evidence >= self.limits.no_evidence_iterations:
            return ProgressAssessment(
                True,
                "no_new_evidence",
                "stop",
                "Se alcanzó el límite de iteraciones sin nueva evidencia.",
                self._iterations_without_evidence,
            )
        return ProgressAssessment(False)

    def _repeat(
        self,
        kind: str,
        signature: str,
        threshold: int,
        recommendation: ProgressRecommendation,
        reason: str,
    ) -> ProgressAssessment:
        if self._last_signatures.get(kind) == signature:
            repetitions = self._repetitions.get(kind, 1) + 1
        else:
            repetitions = 1
        self._last_signatures[kind] = signature
        self._repetitions[kind] = repetitions
        if repetitions >= threshold:
            return ProgressAssessment(True, kind, recommendation, reason, repetitions)
        return ProgressAssessment(False)

    def _reset(self, kind: str) -> None:
        self._last_signatures.pop(kind, None)
        self._repetitions.pop(kind, None)

    def _remember_evidence(self, evidence: Sequence[str]) -> bool:
        fingerprints = {
            self._fingerprint(item)
            for item in evidence
            if isinstance(item, str) and item.strip()
        }
        new = fingerprints - self._known_evidence
        self._known_evidence.update(fingerprints)
        return bool(new)

    @staticmethod
    def _execution_error(result: Mapping[str, Any]) -> str | None:
        if result.get("success") is False:
            return str(result.get("error") or "unknown error")
        payload = result.get("result")
        if isinstance(payload, Mapping):
            exit_code = payload.get("exit_code")
            if isinstance(exit_code, int) and exit_code != 0:
                return str(payload.get("stderr") or f"exit_code={exit_code}")
        return None

    @staticmethod
    def _agent(agent: str) -> str:
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError("agent no puede estar vacío.")
        return agent.strip()

    @staticmethod
    def _fingerprint(*values: object) -> str:
        serialized = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

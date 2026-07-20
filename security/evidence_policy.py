"""Política genérica para decidir si la evidencia permite modificar archivos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.models import EvidenceAssessment
from core.observability import NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation


RiskLevel = Literal["low", "moderate", "high", "excessive"]


def _texts(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in values
    ):
        raise ValueError(f"{field_name} debe contener textos no vacíos.")
    return tuple(item.strip() for item in values)


@dataclass(frozen=True)
class EvidenceContext:
    """Hechos disponibles para evaluar un cambio, sin inferir datos ausentes."""

    component: str
    expected_behavior: str
    conventions: tuple[str, ...] = ()
    impact: tuple[str, ...] = ()
    validation_methods: tuple[str, ...] = ()
    permissions_granted: bool = False
    target_files: tuple[str, ...] = ()
    existing_files: tuple[str, ...] = ()
    supporting_sources: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    risk_level: RiskLevel = "low"

    def __post_init__(self) -> None:
        for field_name in ("component", "expected_behavior"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} debe ser texto.")
            object.__setattr__(self, field_name, value.strip())
        for field_name in (
            "conventions", "impact", "validation_methods", "target_files",
            "existing_files", "supporting_sources", "ambiguities",
            "contradictions", "risks",
        ):
            object.__setattr__(self, field_name, _texts(getattr(self, field_name), field_name))
        if not isinstance(self.permissions_granted, bool):
            raise ValueError("permissions_granted debe ser booleano.")
        if self.risk_level not in {"low", "moderate", "high", "excessive"}:
            raise ValueError("risk_level inválido.")


class EvidenceSufficiencyPolicy:
    """Clasifica evidencia y recomienda una acción sin ejecutar modificaciones."""

    def __init__(self, observability: ObservabilityClient | None = None) -> None:
        self._observability = observability or NoOpObservabilityClient()

    def evaluate(self, context: EvidenceContext) -> EvidenceAssessment:
        if not isinstance(context, EvidenceContext):
            raise TypeError("context debe ser una instancia de EvidenceContext.")

        missing: list[str] = []
        blockers: list[str] = []
        request_help = False

        if not context.component:
            missing.append("componente a modificar")
            blockers.append("El componente a modificar no está identificado.")
        if not context.expected_behavior:
            missing.append("comportamiento esperado")
            blockers.append("El comportamiento esperado no está definido.")
        if context.ambiguities:
            blockers.extend(f"Ambigüedad: {item}" for item in context.ambiguities)
            request_help = True

        existing = set(context.existing_files)
        missing_files = tuple(path for path in context.target_files if path not in existing)
        if missing_files:
            missing.append("archivos objetivo existentes")
            blockers.extend(f"El archivo objetivo no existe: {path}" for path in missing_files)
        if context.contradictions:
            blockers.extend(
                f"Fuentes contradictorias: {item}" for item in context.contradictions
            )
            request_help = True
        if not context.permissions_granted:
            missing.append("permisos de modificación")
            blockers.append("No hay permisos suficientes para modificar los archivos.")
            request_help = True
        if not context.validation_methods:
            missing.append("método de validación")
            blockers.append("No hay validación disponible para comprobar el cambio.")
        if context.risk_level == "excessive":
            blockers.append("El riesgo evaluado es excesivo.")

        if blockers:
            assessment = EvidenceAssessment(
                status="insufficient",
                supporting_sources=context.supporting_sources,
                missing_information=tuple(dict.fromkeys(missing)),
                risks=tuple((*context.risks, *blockers)),
                recommended_action="request_help" if request_help else "stop",
                confidence=1.0,
            )
            self._record(assessment)
            return assessment

        if not context.conventions:
            missing.append("convenciones del componente")
        if not context.impact:
            missing.append("impacto del cambio")
        if not context.supporting_sources:
            missing.append("fuentes de respaldo")
        if not context.target_files:
            missing.append("archivos objetivo")
        if missing:
            assessment = EvidenceAssessment(
                status="partial",
                supporting_sources=context.supporting_sources,
                missing_information=tuple(missing),
                risks=context.risks,
                recommended_action="gather_more_evidence",
                confidence=0.5,
            )
            self._record(assessment)
            return assessment

        assessment = EvidenceAssessment(
            status="sufficient",
            supporting_sources=context.supporting_sources,
            missing_information=(),
            risks=context.risks,
            recommended_action="proceed",
            confidence=1.0,
        )
        self._record(assessment)
        return assessment

    def _record(self, assessment: EvidenceAssessment) -> None:
        emit_observation(
            self._observability,
            ObservabilityEvent(
                "agent", "evidence-assessment",
                payload={"decision": assessment.status,
                         "reasons": assessment.missing_information,
                         "risks": assessment.risks,
                         "recommended_action": assessment.recommended_action,
                         "confidence": assessment.confidence},
            ),
        )

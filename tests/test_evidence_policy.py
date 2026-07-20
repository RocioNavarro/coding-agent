"""Tests unitarios de la política genérica de suficiencia de evidencia."""

from dataclasses import FrozenInstanceError

import pytest

from core.models import EvidenceAssessment
from security.evidence_policy import EvidenceContext, EvidenceSufficiencyPolicy


def complete_context(**overrides: object) -> EvidenceContext:
    values: dict[str, object] = {
        "component": "security.evidence_policy",
        "expected_behavior": "Bloquear cambios sin evidencia suficiente.",
        "conventions": ("Usar dataclasses inmutables.",),
        "impact": ("Afecta la autorización previa a escrituras.",),
        "validation_methods": ("pytest tests/test_evidence_policy.py",),
        "permissions_granted": True,
        "target_files": ("security/evidence_policy.py",),
        "existing_files": ("security/evidence_policy.py",),
        "supporting_sources": ("AGENTS.md", "core/models.py"),
        "risks": ("Una clasificación incorrecta podría bloquear un cambio válido.",),
    }
    values.update(overrides)
    return EvidenceContext(**values)  # type: ignore[arg-type]


def assess(**overrides: object):
    return EvidenceSufficiencyPolicy().evaluate(complete_context(**overrides))


def test_evidence_is_sufficient_when_every_dimension_is_supported() -> None:
    result = assess()

    assert result.status == "sufficient"
    assert result.recommended_action == "proceed"
    assert result.missing_information == ()
    assert result.confidence == 1.0


def test_evidence_is_partial_when_non_blocking_information_is_missing() -> None:
    result = assess(conventions=(), impact=())

    assert result.status == "partial"
    assert result.recommended_action == "gather_more_evidence"
    assert result.missing_information == (
        "convenciones del componente", "impacto del cambio"
    )
    assert result.confidence == 0.5


def test_evidence_is_insufficient_when_required_behavior_is_missing() -> None:
    result = assess(expected_behavior="")

    assert result.status == "insufficient"
    assert result.recommended_action == "stop"
    assert "comportamiento esperado" in result.missing_information


def test_ambiguous_request_requires_help() -> None:
    result = assess(ambiguities=("No se define qué salida debe conservarse.",))

    assert result.status == "insufficient"
    assert result.recommended_action == "request_help"
    assert any("Ambigüedad" in risk for risk in result.risks)


def test_nonexistent_target_file_stops_the_change() -> None:
    result = assess(existing_files=())

    assert result.status == "insufficient"
    assert result.recommended_action == "stop"
    assert any("no existe" in risk for risk in result.risks)


def test_contradictory_sources_require_help() -> None:
    result = assess(contradictions=("README y código describen contratos distintos.",))

    assert result.status == "insufficient"
    assert result.recommended_action == "request_help"
    assert any("contradictorias" in risk for risk in result.risks)


def test_insufficient_permissions_require_help() -> None:
    result = assess(permissions_granted=False)

    assert result.status == "insufficient"
    assert result.recommended_action == "request_help"
    assert "permisos de modificación" in result.missing_information


def test_absent_validation_stops_the_change() -> None:
    result = assess(validation_methods=())

    assert result.status == "insufficient"
    assert result.recommended_action == "stop"
    assert "método de validación" in result.missing_information


def test_excessive_risk_stops_the_change() -> None:
    result = assess(risk_level="excessive")

    assert result.status == "insufficient"
    assert result.recommended_action == "stop"
    assert any("excesivo" in risk for risk in result.risks)


def test_evidence_assessment_is_immutable() -> None:
    result = assess()

    with pytest.raises(FrozenInstanceError):
        result.status = "partial"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "action", "confidence"),
    [
        ("sufficient", "stop", 1.0),
        ("partial", "proceed", 0.5),
        ("insufficient", "stop", 1.1),
    ],
)
def test_evidence_assessment_rejects_invalid_invariants(
    status: str, action: str, confidence: float
) -> None:
    with pytest.raises(ValueError):
        EvidenceAssessment(
            status=status,  # type: ignore[arg-type]
            supporting_sources=(),
            missing_information=(),
            risks=(),
            recommended_action=action,  # type: ignore[arg-type]
            confidence=confidence,
        )

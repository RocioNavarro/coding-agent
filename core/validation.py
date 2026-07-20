"""Validadores puros y reutilizables, sin dependencias del resto del proyecto.

Antes vivían duplicados (con variaciones menores) en core/profiles.py,
core/config.py y agents/web_research.py. Cada módulo sigue lanzando su propia
excepción específica; sólo la normalización en sí está unificada acá.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def normalize_domain(value: str) -> str:
    """Normaliza un dominio: minúsculas, sin esquema ni 'www.'.

    Lanza ValueError si, tras normalizar, no queda un dominio válido.
    """
    candidate = value.strip().casefold()
    if "://" in candidate:
        candidate = urlsplit(candidate).hostname or ""
    candidate = candidate.removeprefix("www.").strip(".")
    if not candidate or "/" in candidate or " " in candidate:
        raise ValueError(f"Dominio inválido: {value!r}")
    return candidate

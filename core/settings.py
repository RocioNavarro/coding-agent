"""Configuración central del coding agent."""

from dataclasses import dataclass


@dataclass
class AgentSettings:
    """Opciones de ejecución configurables de la Parte 1."""

    supervision_enabled: bool = True
    plan_mode_enabled: bool = True
    max_iterations: int = 20
    command_timeout_seconds: int = 60


DEFAULT_SETTINGS = AgentSettings()

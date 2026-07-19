"""Configuración central del coding agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import AgentConfig


@dataclass
class AgentSettings:
    """Opciones de ejecución configurables de la Parte 1."""

    supervision_enabled: bool = True
    plan_mode_enabled: bool = True
    max_iterations: int = 20
    command_timeout_seconds: int = 60
    web_search_enabled: bool = True
    web_search_config: dict[str, Any] | None = None
    agent_config: AgentConfig | None = None

    @classmethod
    def from_environment(cls) -> "AgentSettings":
        """Carga opciones no secretas desde el entorno ya inicializado."""
        enabled = os.getenv("CODING_AGENT_WEB_SEARCH_ENABLED", "true").strip().casefold()
        if enabled not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError("CODING_AGENT_WEB_SEARCH_ENABLED debe ser booleano.")
        raw_config = os.getenv("CODING_AGENT_WEB_SEARCH_CONFIG", "{}").strip() or "{}"
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError as error:
            raise ValueError("CODING_AGENT_WEB_SEARCH_CONFIG debe contener JSON válido.") from error
        if not isinstance(config, dict):
            raise ValueError("CODING_AGENT_WEB_SEARCH_CONFIG debe ser un objeto JSON.")
        configured_path = os.getenv("CODING_AGENT_CONFIG")
        default_path = Path("agent.config.yaml")
        from core.config import load_agent_config

        agent_config = (
            load_agent_config(configured_path)
            if configured_path
            else load_agent_config(default_path) if default_path.is_file() else None
        )
        return cls(
            web_search_enabled=enabled in {"true", "1", "yes", "on"},
            web_search_config=config,
            agent_config=agent_config,
        )


DEFAULT_SETTINGS = AgentSettings()

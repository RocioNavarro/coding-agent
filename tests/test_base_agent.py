"""Tests del runtime común de subagentes sin llamadas reales a APIs."""

import json
from collections.abc import Sequence
from typing import Any

import pytest

from agents.base import AgentContext, AgentExecutionError, StubAgent
from core.models import LLMResponse, LLMUsage, Message, ToolCall
from core.task_state import SourceReference, TaskState
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry


class FakeLLMClient:
    """Cliente determinista que captura exactamente el contexto y schemas recibidos."""

    def __init__(
        self, response: LLMResponse | None = None, error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.messages: list[Message] = []
        self.schemas: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.messages = list(messages)
        self.schemas = list(tools)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def structured_response(
    *, tool_calls: list[ToolCall] | None = None
) -> LLMResponse:
    payload = {
        "summary": "Revisión terminada",
        "findings": ["El estado está aislado de la CLI."],
        "recommendations": ["Mantener contratos pequeños."],
        "sources": [
            {
                "origin": "repository",
                "reference": "core/task_state.py",
                "summary": "Modelo compartido",
            }
        ],
        "files_relevant": ["core/task_state.py"],
        "blockers": [],
        "confidence": 0.9,
    }
    calls = tool_calls or []
    return LLMResponse(
        assistant_message=Message(
            role="assistant", content=json.dumps(payload), tool_calls=calls
        ),
        text=json.dumps(payload),
        tool_calls=calls,
        model="fake-model",
        usage=LLMUsage(5, 5, 10),
        latency_ms=1.0,
    )


def registry_with_tools() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("read_file", "write_file"):
        registry.register(
            ToolDefinition(
                name=name,
                description=f"Tool simulada {name}.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                executor=lambda path: path,
                modifies_system=name == "write_file",
            )
        )
    return registry


def build_agent(client: FakeLLMClient, *allowed_tools: str) -> StubAgent:
    return StubAgent(
        name="stub-reviewer",
        role="Revisor simulado",
        system_prompt="Analizá solamente la información provista.",
        allowed_tools=allowed_tools,
        llm_client=client,
    )


def test_runs_stub_agent_with_only_explicit_context() -> None:
    client = FakeLLMClient(structured_response())
    agent = build_agent(client, "read_file")
    state = TaskState.create(
        "Pedido original que no debe copiarse", task_id="task-agent"
    )
    state.add_repository_finding("Hallazgo global no seleccionado")
    context = AgentContext(
        facts=("Dato seleccionado",),
        files=("core/task_state.py",),
        constraints=("No modificar archivos",),
    )

    result = agent.run(
        "Revisá el modelo", state, context, registry_with_tools()
    )

    assert result.summary == "Revisión terminada"
    assert result.findings == ("El estado está aislado de la CLI.",)
    assert result.confidence == 0.9
    assert [message.role for message in client.messages] == ["system", "user"]
    sent_input = json.loads(client.messages[1].content)
    assert sent_input == {
        "task_id": "task-agent",
        "instruction": "Revisá el modelo",
        "context": {
            "facts": ["Dato seleccionado"],
            "sources": [],
            "files": ["core/task_state.py"],
            "constraints": ["No modificar archivos"],
        },
    }
    assert "Pedido original" not in client.messages[1].content
    assert "Hallazgo global" not in client.messages[1].content
    assert [schema["function"]["name"] for schema in client.schemas] == [
        "read_file"
    ]


def test_rejects_tool_call_not_allowed_for_agent() -> None:
    forbidden_call = ToolCall(
        id="call-write",
        name="write_file",
        arguments={"path": "archivo.py"},
    )
    client = FakeLLMClient(structured_response(tool_calls=[forbidden_call]))
    agent = build_agent(client, "read_file")
    state = TaskState.create("No escribir", task_id="task-forbidden")

    with pytest.raises(AgentExecutionError, match="no está permitida"):
        agent.run("Inspeccioná", state, available_tools=registry_with_tools())

    assert state.subagent_results == ()
    assert len(state.errors) == 1
    assert state.errors[0].component == "stub-reviewer"


def test_converts_result_and_sources_into_shared_state() -> None:
    read_call = ToolCall(
        id="call-read",
        name="read_file",
        arguments={"path": "core/task_state.py"},
    )
    client = FakeLLMClient(structured_response(tool_calls=[read_call]))
    agent = build_agent(client, "read_file")
    state = TaskState.create("Revisar estado", task_id="task-state")

    result = agent.run(
        "Buscá riesgos", state, available_tools=registry_with_tools()
    )

    assert state.subagent_results == (result,)
    assert result.requested_tool_calls == (read_call,)
    assert result.files_relevant == ("core/task_state.py",)
    assert state.sources == (
        SourceReference(
            origin="repository",
            reference="core/task_state.py",
            summary="Modelo compartido",
        ),
    )


def test_wraps_llm_error_and_records_it_in_shared_state() -> None:
    client = FakeLLMClient(error=RuntimeError("proveedor caído"))
    agent = build_agent(client)
    state = TaskState.create("Intentar análisis", task_id="task-error")
    state.set_phase("delegation")

    with pytest.raises(AgentExecutionError, match="proveedor caído") as captured:
        agent.run("Analizá", state)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert state.subagent_results == ()
    assert len(state.errors) == 1
    assert state.errors[0].phase == "delegation"
    assert state.errors[0].recoverable is True


def test_agent_result_preserves_json_round_trip() -> None:
    client = FakeLLMClient(structured_response())
    state = TaskState.create("Serializar resultado", task_id="task-json-agent")

    build_agent(client).run("Revisá", state)
    restored = TaskState.from_json(state.to_json())

    assert restored == state

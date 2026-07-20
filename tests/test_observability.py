"""Tests de contratos y degradación segura de observabilidad."""

import logging
from dataclasses import FrozenInstanceError

import pytest

from core.observability import (
    LangfuseObservabilityClient,
    NoOpObservabilityClient,
    ObservabilityClient,
    ObservabilityEvent,
    build_observability_client,
    sanitize_observability_data,
)
from core.llm_client import ObservedLLMClient
from core.models import LLMResponse, LLMUsage, Message
from core.settings import AgentSettings
from core.supervision import SupervisedToolExecutor
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry
from security.evidence_policy import EvidenceContext, EvidenceSufficiencyPolicy
from core.progress import ProgressLimits, ProgressMonitor


class FakeObservation:
    def __init__(self, index: int = 0) -> None:
        self.ended = False
        self.trace_id = "0" * 31 + "1"
        self.observation_id = f"{index + 1:016x}"

    def end(self) -> None:
        self.ended = True


class FakeLangfuse:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []
        self.observations: list[FakeObservation] = []
        self.flushed = False

    def start_observation(self, **arguments: object) -> FakeObservation:
        if self.fail:
            raise RuntimeError("provider unavailable")
        self.calls.append(arguments)
        observation = FakeObservation(len(self.observations))
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        if self.fail:
            raise RuntimeError("flush unavailable")
        self.flushed = True


class FakeObservabilityClient:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def record(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None


def test_noop_client_accepts_events_and_flushes() -> None:
    client = NoOpObservabilityClient()

    client.record(ObservabilityEvent("task", "task-started"))
    client.flush()


def test_disabled_observability_selects_noop() -> None:
    client = build_observability_client(
        environ={"CODING_AGENT_OBSERVABILITY_ENABLED": "false"}
    )

    assert isinstance(client, NoOpObservabilityClient)


def test_missing_credentials_selects_noop_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        client = build_observability_client(
            environ={"CODING_AGENT_OBSERVABILITY_ENABLED": "true"}
        )

    assert isinstance(client, NoOpObservabilityClient)
    assert "faltan credenciales" in caplog.text


def test_fake_langfuse_factory_is_injectable_without_exposing_credentials() -> None:
    created: dict[str, object] = {}
    fake = FakeLangfuse()

    def factory(**arguments: object) -> FakeLangfuse:
        created.update(arguments)
        return fake

    client = build_observability_client(
        environ={
            "CODING_AGENT_OBSERVABILITY_ENABLED": "true",
            "LANGFUSE_PUBLIC_KEY": "public-value",
            "LANGFUSE_SECRET_KEY": "secret-value",
            "LANGFUSE_HOST": "https://langfuse.test",
        },
        langfuse_factory=factory,
    )

    assert isinstance(client, LangfuseObservabilityClient)
    assert created == {
        "public_key": "public-value",
        "secret_key": "secret-value",
        "host": "https://langfuse.test",
    }


def test_recursive_sanitization_redacts_keys_and_embedded_values() -> None:
    original = {
        "headers": {"Authorization": "Bearer private", "safe": "value"},
        "items": [
            {"api_key": "key"}, "password=hunter2", "usar sk-lf-private1234"
        ],
    }

    sanitized = sanitize_observability_data(original)

    assert sanitized == {
        "headers": {"Authorization": "***", "safe": "value"},
        "items": [{"api_key": "***"}, "password=***", "usar ***"],
    }
    assert original["headers"]["Authorization"] == "Bearer private"


def test_provider_error_does_not_interrupt_main_flow(caplog) -> None:
    client = LangfuseObservabilityClient(FakeLangfuse(fail=True))

    with caplog.at_level(logging.WARNING):
        client.record(ObservabilityEvent("result", "finished", payload={"ok": True}))
        client.flush()

    assert "provider unavailable" in caplog.text
    assert "flush unavailable" in caplog.text


def test_typed_event_is_immutable_and_validates_fields() -> None:
    event = ObservabilityEvent("agent", "explorer", agent="explorer")

    assert isinstance(FakeObservabilityClient(), ObservabilityClient)
    with pytest.raises(FrozenInstanceError):
        event.name = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        ObservabilityEvent("invalid", "event")  # type: ignore[arg-type]


def test_langfuse_v4_contract_ends_observation_and_omits_unknown_cost() -> None:
    fake = FakeLangfuse()
    client = LangfuseObservabilityClient(fake)
    event = ObservabilityEvent(
        "llm_call", "completion", model="model-a", input_tokens=4,
        output_tokens=2, total_tokens=6, latency_ms=12.5,
    )

    client.record(event)

    assert event.estimated_cost is None
    assert fake.calls[0]["as_type"] == "generation"
    assert fake.calls[0]["usage_details"] == {
        "input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
    }
    assert "cost_details" not in fake.calls[0]
    assert fake.observations[0].ended is True


def test_parent_child_relationship_uses_langfuse_trace_context() -> None:
    fake = FakeLangfuse()
    client = LangfuseObservabilityClient(fake)

    client.record(ObservabilityEvent("task", "root", event_id="root"))
    client.record(
        ObservabilityEvent("agent", "child", event_id="child", parent_event_id="root")
    )

    assert fake.calls[1]["trace_context"] == {
        "trace_id": fake.observations[0].trace_id,
        "parent_span_id": fake.observations[0].observation_id,
    }


class StaticLLM:
    def complete(self, messages, tools=()):
        return LLMResponse(
            Message("assistant", "token=secret-value"), "token=secret-value", [],
            "fake-model", LLMUsage(3, 2, 5), 4.0,
        )


def test_common_llm_wrapper_records_tokens_cost_and_sanitized_io() -> None:
    observed = FakeObservabilityClient()
    client = ObservedLLMClient(StaticLLM(), observed)

    client.complete([Message("user", "password=hunter2")])

    assert len(observed.events) == 1
    event = observed.events[0]
    assert event.event_type == "llm_call"
    assert event.total_tokens == 5
    assert event.estimated_cost is None
    assert "hunter2" not in str(event.payload)
    assert "secret-value" not in str(event.payload)


def test_central_tool_executor_emits_one_policy_and_one_execution_event() -> None:
    observed = FakeObservabilityClient()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "read_file", "read", {
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            }, lambda path: "ok", False,
        )
    )
    executor = SupervisedToolExecutor(
        registry, AgentSettings(supervision_enabled=False), observability=observed
    )

    result = executor.execute("read_file", {"path": "safe.txt"})

    assert result["success"] is True
    assert [event.name for event in observed.events] == ["policy-decision", "read_file"]


def test_evidence_and_progress_policies_emit_decisions_once() -> None:
    observed = FakeObservabilityClient()
    evidence = EvidenceSufficiencyPolicy(observed)
    evidence.evaluate(
        EvidenceContext(
            component="app.py", expected_behavior="change", conventions=("existing",),
            impact=("localized",), validation_methods=("pytest",),
            permissions_granted=True, target_files=("app.py",),
            existing_files=("app.py",), supporting_sources=("app.py",),
        )
    )
    progress = ProgressMonitor(
        ProgressLimits(read_repeats=2), observability=observed
    )
    progress.record_tool_call("main", "read_file", {"path": "a"}, {"result": "x"})
    progress.record_tool_call("main", "read_file", {"path": "a"}, {"result": "x"})

    assert [event.name for event in observed.events].count("evidence-assessment") == 1
    progress_events = [event for event in observed.events if event.name == "progress-assessment"]
    assert len(progress_events) == 2
    assert progress_events[-1].payload["repetition_detected"] is True

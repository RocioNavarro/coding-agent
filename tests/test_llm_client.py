"""Tests unitarios del adaptador OpenAI sin llamadas reales de red."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from core.llm_client import (
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    OpenAILLMClient,
)
from core.models import LLMUsage, Message, ToolCall


@pytest.fixture(autouse=True)
def clear_llm_tuning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que la configuración local vuelva no deterministas estos tests."""
    monkeypatch.delenv("CODING_AGENT_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CODING_AGENT_LLM_MAX_RETRIES", raising=False)


def make_response(*, output: list[object] | None = None, text: str = "Hola") -> object:
    """Construye una respuesta mínima semejante a la entregada por el SDK."""
    return SimpleNamespace(
        status="completed",
        model="gpt-test",
        output_text=text,
        output=output or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
    )


def make_client(response: object) -> tuple[OpenAILLMClient, Mock]:
    """Inyecta un SDK simulado en el adaptador."""
    sdk = Mock()
    sdk.responses.create.return_value = response
    return OpenAILLMClient(api_key="test-key", model="gpt-test", client=sdk), sdk


def test_text_response_is_normalized_and_request_uses_plain_data() -> None:
    client, sdk = make_client(make_response())

    with patch("core.llm_client.perf_counter", side_effect=[1.0, 1.125]):
        result = client.complete([Message(role="user", content="Hola")])

    assert result.assistant_message == Message(role="assistant", content="Hola")
    assert result.text == "Hola"
    assert result.tool_calls == []
    assert result.model == "gpt-test"
    assert result.usage == LLMUsage(10, 4, 14)
    assert result.latency_ms == pytest.approx(125.0)
    sdk.responses.create.assert_called_once_with(
        model="gpt-test", input=[{"role": "user", "content": "Hola"}]
    )


def test_multiple_tool_calls_are_normalized_and_schema_is_converted() -> None:
    calls = [
        SimpleNamespace(
            type="function_call", call_id="call-1", name="read_file",
            arguments='{"path": "README.md"}',
        ),
        SimpleNamespace(
            type="function_call", call_id="call-2", name="list_files",
            arguments='{"path": "."}',
        ),
    ]
    client, sdk = make_client(make_response(output=calls, text=""))
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee un archivo.",
            "parameters": {"type": "object"},
        },
    }

    result = client.complete([Message(role="user", content="Leé")], [tool])

    assert result.tool_calls == [
        ToolCall("call-1", "read_file", {"path": "README.md"}),
        ToolCall("call-2", "list_files", {"path": "."}),
    ]
    assert result.assistant_message.tool_calls == result.tool_calls
    assert sdk.responses.create.call_args.kwargs["tools"] == [{
        "type": "function",
        "name": "read_file",
        "description": "Lee un archivo.",
        "parameters": {"type": "object"},
    }]


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_invalid_tool_arguments_raise_controlled_error(arguments: str) -> None:
    call = SimpleNamespace(
        type="function_call", call_id="call-1", name="read_file",
        arguments=arguments,
    )
    client, _ = make_client(make_response(output=[call], text=""))

    with pytest.raises(LLMInvalidResponseError):
        client.complete([Message(role="user", content="Leé")])


def test_incomplete_response_raises_controlled_error() -> None:
    response = make_response()
    response.status = "incomplete"  # type: ignore[attr-defined]
    client, _ = make_client(response)

    with pytest.raises(LLMInvalidResponseError, match="no se completó"):
        client.complete([Message(role="user", content="Hola")])


def test_malformed_response_raises_controlled_error() -> None:
    client, _ = make_client(SimpleNamespace(status="completed"))

    with pytest.raises(LLMInvalidResponseError, match="formato inválido"):
        client.complete([Message(role="user", content="Hola")])


def test_provider_error_is_translated() -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = APIConnectionError(request=Mock())
    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=sdk)

    with pytest.raises(LLMProviderError, match="No fue posible conectar"):
        client.complete([Message(role="user", content="Hola")])


def test_default_timeout_and_retries() -> None:
    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=Mock())

    assert client.timeout_seconds == 60
    assert client.max_retries == 1


def test_timeout_and_retries_are_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODING_AGENT_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("CODING_AGENT_LLM_MAX_RETRIES", "0")

    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=Mock())

    assert client.timeout_seconds == 12.5
    assert client.max_retries == 0


def test_empty_tuning_values_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_AGENT_LLM_TIMEOUT_SECONDS", "")
    monkeypatch.setenv("CODING_AGENT_LLM_MAX_RETRIES", "")

    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=Mock())

    assert client.timeout_seconds == 60
    assert client.max_retries == 1


@pytest.mark.parametrize("value", ["invalid", "0", "-2"])
def test_invalid_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CODING_AGENT_LLM_TIMEOUT_SECONDS", value)

    with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT_SECONDS"):
        OpenAILLMClient(api_key="test-key", model="gpt-test", client=Mock())


@pytest.mark.parametrize("value", ["invalid", "1.5", "-1"])
def test_invalid_max_retries_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CODING_AGENT_LLM_MAX_RETRIES", value)

    with pytest.raises(LLMConfigurationError, match="LLM_MAX_RETRIES"):
        OpenAILLMClient(api_key="test-key", model="gpt-test", client=Mock())


def test_sdk_receives_timeout_and_retries() -> None:
    with patch("core.llm_client.OpenAI") as sdk_class:
        OpenAILLMClient(
            api_key="test-key",
            model="gpt-test",
            timeout_seconds=23,
            max_retries=4,
        )

    sdk_class.assert_called_once_with(
        api_key="test-key", timeout=23.0, max_retries=4
    )


def provider_exception(kind: str, secret: str = "secret-api-key") -> OpenAIError:
    """Construye excepciones reales del SDK sin hacer solicitudes de red."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    if kind == "timeout":
        return APITimeoutError(request)
    if kind == "connection":
        return APIConnectionError(message=secret, request=request)
    response = httpx.Response(
        401 if kind == "authentication" else 403,
        request=request,
        headers={"x-request-id": "req-safe-test"},
    )
    classes = {
        "authentication": AuthenticationError,
        "permission": PermissionDeniedError,
        "rate": RateLimitError,
        "status": APIStatusError,
    }
    if kind == "rate":
        response.status_code = 429
    if kind == "status":
        response.status_code = 500
    return classes[kind](secret, response=response, body=None)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("timeout", "límite configurado de 60 segundos"),
        ("connection", "No fue posible conectar"),
        ("authentication", "API key de OpenAI fue rechazada"),
        ("permission", "no tiene acceso al modelo"),
        ("rate", "límite de uso, cuota o saldo"),
        ("status", "HTTP 500; request ID: req-safe-test"),
    ],
)
def test_sdk_errors_are_sanitized(kind: str, expected: str) -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = provider_exception(kind)
    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=sdk)

    with pytest.raises(LLMProviderError, match=expected) as error:
        client.complete([Message(role="user", content="Hola")])

    assert "secret-api-key" not in str(error.value)


def test_fallback_openai_error_is_sanitized() -> None:
    sdk = Mock()
    sdk.responses.create.side_effect = OpenAIError("secret-api-key")
    client = OpenAILLMClient(api_key="test-key", model="gpt-test", client=sdk)

    with pytest.raises(LLMProviderError, match="error inesperado") as error:
        client.complete([Message(role="user", content="Hola")])

    assert "secret-api-key" not in str(error.value)


def test_reads_configuration_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    sdk = Mock()
    sdk.responses.create.return_value = make_response()
    client = OpenAILLMClient(client=sdk)

    client.complete([Message(role="user", content="Hola")])

    assert sdk.responses.create.call_args.kwargs["model"] == "env-model"


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_missing_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.delenv(missing)

    with pytest.raises(LLMConfigurationError, match=missing) as error:
        OpenAILLMClient(client=Mock())

    message = str(error.value)
    assert "env-key" not in message
    assert "env-model" not in message


def test_invalid_tool_schema_is_rejected_before_sdk_call() -> None:
    client, sdk = make_client(make_response())

    with pytest.raises(LLMConfigurationError, match="definición"):
        client.complete(
            [Message(role="user", content="Hola")],
            [{"type": "function", "function": {"name": "broken"}}],
        )

    sdk.responses.create.assert_not_called()


def test_tool_history_is_serialized_without_sdk_types() -> None:
    client, sdk = make_client(make_response())
    call = ToolCall("call-1", "read_file", {"path": "README.md"})
    history = [
        Message(role="user", content="Leé el archivo"),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content='{"success": true}', tool_call_id="call-1"),
    ]

    client.complete(history)

    assert sdk.responses.create.call_args.kwargs["input"] == [
        {"role": "user", "content": "Leé el archivo"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"success": true}',
        },
    ]

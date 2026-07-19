"""Tests unitarios de las invariantes de la fase de planificación."""

from collections.abc import Sequence
from typing import Any

import pytest

from core.harness import PlanningError, run_planning_loop
from core.models import LLMResponse, LLMUsage, Message, PlanReview, ToolCall


class PlanningLLM:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.tools: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.tools.append(list(tools))
        return self.response


def response(text: str, calls: list[ToolCall] | None = None) -> LLMResponse:
    tool_calls = calls or []
    return LLMResponse(
        assistant_message=Message("assistant", text, tool_calls=tool_calls),
        text=text, tool_calls=tool_calls, model="fake",
        usage=LLMUsage(1, 1, 2), latency_ms=1.0,
    )


@pytest.mark.parametrize(
    "invalid_response",
    [response(""), response("Plan", [ToolCall("call", "write_file", {})])],
)
def test_invalid_plans_never_reach_execution(invalid_response: LLMResponse) -> None:
    llm = PlanningLLM(invalid_response)

    with pytest.raises(PlanningError):
        run_planning_loop(
            llm, [Message("user", "Tarea")], lambda plan: PlanReview("approve"),
            max_revisions=2, output=lambda text: None,
        )

    assert llm.tools == [[]]

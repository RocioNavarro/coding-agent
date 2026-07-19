"""Detección neutral de loops, estancamiento y falsos positivos."""

import pytest

from core.progress import ProgressLimits, ProgressMonitor


def monitor(**overrides: int) -> ProgressMonitor:
    return ProgressMonitor(ProgressLimits(**overrides))


def test_detects_same_command_with_same_error() -> None:
    progress = monitor(command_error_repeats=2)

    first = progress.record_tool_call(
        "tester", "run_command", {"command": "tool test"},
        {"success": False, "error": "exit 1"},
    )
    second = progress.record_tool_call(
        "tester", "run_command", {"command": "tool test"},
        {"success": False, "error": "exit 1"},
    )

    assert first.detected is False
    assert second.detected is True
    assert second.recommendation == "retry_with_new_strategy"


def test_command_success_or_new_error_is_not_same_failure_loop() -> None:
    progress = monitor(command_error_repeats=2)

    progress.record_tool_call(
        "tester", "run_command", {"command": "tool test"},
        {"success": False, "error": "first"},
    )
    changed = progress.record_tool_call(
        "tester", "run_command", {"command": "tool test"},
        {"success": False, "error": "second"},
    )
    success = progress.record_tool_call(
        "tester", "run_command", {"command": "tool test"},
        {"success": True, "result": {"exit_code": 0}},
    )

    assert changed.detected is False
    assert success.detected is False


def test_detects_same_read_without_new_justification() -> None:
    progress = monitor(read_repeats=2)
    result = {"success": True, "result": "same content"}

    progress.record_tool_call("explorer", "read_file", {"path": "doc.txt"}, result)
    repeated = progress.record_tool_call(
        "explorer", "read_file", {"path": "doc.txt"}, result
    )

    assert repeated.detected is True
    assert repeated.kind == "repeated_read"


def test_new_read_justification_avoids_false_positive() -> None:
    progress = monitor(read_repeats=2)
    result = {"success": True, "result": "same content"}

    progress.record_tool_call(
        "explorer", "read_file", {"path": "doc.txt"}, result,
        justification="inspect architecture",
    )
    assessment = progress.record_tool_call(
        "explorer", "read_file", {"path": "doc.txt"}, result,
        justification="verify changed contract",
    )

    assert assessment.detected is False


def test_detects_repeated_search_but_not_distinct_queries() -> None:
    progress = monitor(search_repeats=2)
    result = {"success": True, "result": ["source"]}

    progress.record_tool_call("researcher", "web_search", {"query": "topic a"}, result)
    duplicate = progress.record_tool_call(
        "researcher", "web_search", {"query": "topic a"}, result
    )
    distinct = progress.record_tool_call(
        "researcher", "web_search", {"query": "topic b"}, result
    )

    assert duplicate.recommendation == "retry_with_new_strategy"
    assert distinct.detected is False


def test_detects_same_modification_and_allows_changed_content() -> None:
    progress = monitor(modification_repeats=2)
    ok = {"success": True, "result": "written"}

    progress.record_tool_call(
        "implementer", "write_file", {"path": "app.txt", "content": "one"}, ok
    )
    duplicate = progress.record_tool_call(
        "implementer", "write_file", {"path": "app.txt", "content": "one"}, ok
    )
    changed = progress.record_tool_call(
        "implementer", "write_file", {"path": "app.txt", "content": "two"}, ok
    )

    assert duplicate.recommendation == "replan"
    assert changed.detected is False


def test_detects_same_diff_and_ignores_changed_diff() -> None:
    progress = monitor(diff_repeats=2)

    progress.record_diff("reviewer", "- old\n+ new")
    duplicate = progress.record_diff("reviewer", "- old\n+ new")
    changed = progress.record_diff("reviewer", "- old\n+ newer")

    assert duplicate.recommendation == "replan"
    assert changed.detected is False


def test_detects_cycle_between_agents() -> None:
    progress = monitor(agent_cycle_repeats=3, max_cycle_length=3)
    assessment = None
    for agent in ("implementer", "tester") * 3:
        assessment = progress.record_agent(agent)

    assert assessment is not None
    assert assessment.detected is True
    assert assessment.kind == "agent_cycle"
    assert assessment.recommendation == "ask_user"


def test_normal_pipeline_is_not_an_agent_cycle() -> None:
    progress = monitor(agent_cycle_repeats=2)

    assessments = [
        progress.record_agent(agent)
        for agent in ("explorer", "researcher", "implementer", "tester", "reviewer")
    ]

    assert not any(item.detected for item in assessments)


def test_same_action_from_different_agents_is_not_a_false_loop() -> None:
    progress = monitor(search_repeats=2)
    result = {"success": True, "result": ["source"]}

    progress.record_tool_call(
        "researcher-a", "web_search", {"query": "shared topic"}, result
    )
    assessment = progress.record_tool_call(
        "researcher-b", "web_search", {"query": "shared topic"}, result
    )

    assert assessment.detected is False


def test_detects_iterations_without_new_evidence_and_resets_on_evidence() -> None:
    progress = monitor(no_evidence_iterations=3)

    assert progress.record_iteration(evidence=("source-a",)).detected is False
    assert progress.record_iteration(evidence=("source-a",)).detected is False
    assert progress.record_iteration(evidence=()).detected is False
    stopped = progress.record_iteration(evidence=("source-a",))
    reset = progress.record_iteration(evidence=("source-b",))

    assert stopped.recommendation == "stop"
    assert reset.detected is False


@pytest.mark.parametrize(
    "arguments",
    [
        {"command_error_repeats": 1},
        {"read_repeats": 0},
        {"agent_cycle_repeats": 1},
        {"max_cycle_length": 1},
    ],
)
def test_rejects_unsafe_limits(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ProgressLimits(**arguments)

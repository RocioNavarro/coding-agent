"""Pruebas de la infraestructura compartida para escenarios integrales."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.project_memory import ProjectMemory
from agents.researcher import EvidenceFragment
from agents.tester import CommandOutcome
from core.harness import run_planning_loop
from core.models import Message, PlanReview
from core.observability import ObservabilityEvent, emit_observation
from core.task_state import TaskState
from tests.integration.fakes import (
    FakeMemory,
    FakeObservability,
    FakeRAG,
    FakeWeb,
    ScriptedCommandRunner,
    ScriptedLLM,
    ScriptedPlanReview,
    ScriptedReviewer,
)
from tests.integration.repositories import REPOSITORY_CONTENTS, TemporaryRepository


def test_all_repository_fixtures_are_minimal_and_owned_by_tmp_path(
    tmp_path: Path,
    analysis_repository: TemporaryRepository,
    simple_change_repository: TemporaryRepository,
    rag_repository: TemporaryRepository,
    persistent_memory_repository: TemporaryRepository,
    failed_command_repository: TemporaryRepository,
    blocked_operation_repository: TemporaryRepository,
) -> None:
    repositories = (
        analysis_repository,
        simple_change_repository,
        rag_repository,
        persistent_memory_repository,
        failed_command_repository,
        blocked_operation_repository,
    )

    assert len({repository.root for repository in repositories}) == 6
    for repository in repositories:
        assert repository.root.is_relative_to(tmp_path)
        assert repository.files() == tuple(sorted(repository.initial_files))
        assert repository.initial_files == REPOSITORY_CONTENTS[repository.kind]
        assert repository.modified_files() == ()


def test_repositories_are_isolated(
    analysis_repository: TemporaryRepository,
    simple_change_repository: TemporaryRepository,
) -> None:
    target = simple_change_repository.root / "src/value.py"
    target.write_text("VALUE = 'after'\n", encoding="utf-8")

    assert simple_change_repository.modified_files() == ("src/value.py",)
    assert analysis_repository.read("src/component.py") == "VALUE = 'stable'\n"
    assert analysis_repository.modified_files() == ()


def test_only_persistent_fixture_exposes_reusable_memory_storage(
    persistent_memory_repository: TemporaryRepository,
    analysis_repository: TemporaryRepository,
) -> None:
    assert persistent_memory_repository.memory_root is not None
    assert analysis_repository.memory_root is None

    first = ProjectMemory(
        persistent_memory_repository.root,
        identifier="integration",
        storage_root=persistent_memory_repository.memory_root,
    ).load()
    first.add_decision("Keep the public contract stable.")
    first.save()

    second = ProjectMemory(
        persistent_memory_repository.root,
        identifier="integration",
        storage_root=persistent_memory_repository.memory_root,
    ).load()
    assert second.data["decisions"] == [
        {"decision": "Keep the public contract stable."}
    ]
    assert not (analysis_repository.root / ".coding-agent").exists()


def test_scripted_llm_supports_phase_sequences_errors_and_prompt_log() -> None:
    llm = ScriptedLLM(
        ("fallback",),
        by_phase={"explorer": ("first", "second", RuntimeError("llm failed"))},
    )
    prompt = [Message("user", "inspect")]

    llm.set_phase("explorer")
    assert llm.complete(prompt).text == "first"
    assert llm.complete(prompt).text == "second"
    with pytest.raises(RuntimeError, match="llm failed"):
        llm.complete(prompt)
    llm.set_phase("reviewer")
    assert llm.complete(prompt).text == "fallback"
    assert [phase for phase, _ in llm.prompts] == [
        "explorer", "explorer", "explorer", "reviewer"
    ]


def test_plan_review_fake_supports_approve_reject_modify_and_sequences() -> None:
    assert ScriptedPlanReview.approve()("plan").action == "approve"
    assert ScriptedPlanReview.reject()("plan").action == "reject"

    review = ScriptedPlanReview.modify("add validation")
    first = review("plan one")
    second = review("plan two")
    assert first == PlanReview("modify", "add validation")
    assert second == PlanReview("approve")
    assert review.plans == ["plan one", "plan two"]


def test_command_runner_records_attempts_failures_and_timeout() -> None:
    runner = ScriptedCommandRunner(
        {
            "project-check": (
                CommandOutcome(1, 2.0, stderr="failed"),
                CommandOutcome(0, 1.0, stdout="ok"),
            ),
            "slow-check": (
                CommandOutcome(None, 50.0, stderr="timeout", timed_out=True),
            ),
        }
    )

    assert runner.execute("project-check", timeout_seconds=5).exit_code == 1
    assert runner.execute("project-check", timeout_seconds=5).stdout == "ok"
    timeout = runner.execute("slow-check", timeout_seconds=0.05)
    assert timeout.timed_out is True
    assert runner.calls == [
        ("project-check", 5), ("project-check", 5), ("slow-check", 0.05)
    ]


def test_fake_observability_tracks_hierarchy_lifecycle_and_provider_errors() -> None:
    observed = FakeObservability()
    root = ObservabilityEvent("task", "task-start", event_id="root", task_id="task-1")
    child = ObservabilityEvent(
        "agent", "explorer", event_id="child", parent_event_id="root", task_id="task-1"
    )
    end = ObservabilityEvent("result", "task-end", parent_event_id="root", task_id="task-1")

    observed.record(root)
    observed.record(child)
    observed.record(end)
    observed.flush()
    assert observed.children_of("root") == (child, end)
    assert observed.open_traces == set()
    assert observed.closed_traces == {"root"}
    assert observed.flushed is True

    failing = FakeObservability(fail=True)
    emit_observation(failing, root)
    assert failing.events == []


def test_reviewer_rag_memory_and_web_fakes_are_sequential_and_auditable() -> None:
    fragment = EvidenceFragment("rag", "docs/contract.md", "contract", 0.9)
    reviewer = ScriptedReviewer(("changes_requested", "approved"))
    state = TaskState.create("review the result", task_id="task-review")
    rag = FakeRAG((fragment,))
    memory = FakeMemory((fragment,))
    web = FakeWeb((fragment,))

    assert reviewer.run("first review", state).decision == "changes_requested"
    assert reviewer.run("second review", state).decision == "approved"
    assert rag.retrieve_filtered("contract", filters={"tag": "stable"}, limit=1) == (fragment,)
    assert memory.search("decision", limit=1) == (fragment,)
    assert web.search_context("reference", limit=1) == (fragment,)
    assert rag.calls == [("contract", {"tag": "stable"}, 1)]
    assert memory.queries == [("decision", 1)]
    assert web.calls == [("reference", 1)]
    assert reviewer.calls == ["first review", "second review"]
    assert [result.status for result in state.subagent_results] == [
        "changes_requested", "approved"
    ]


def test_fixture_contents_do_not_reference_a_specific_evaluation_repository() -> None:
    forbidden = (
        "Print" + "Script",
        "Kot" + "lin",
        "Gra" + "dle",
        "final" + "-evaluation-repository",
    )
    corpus = "\n".join(
        content for files in REPOSITORY_CONTENTS.values() for content in files.values()
    )
    assert not any(term.casefold() in corpus.casefold() for term in forbidden)


def test_llm_and_plan_fakes_are_compatible_with_existing_planning_loop() -> None:
    llm = ScriptedLLM(("1. Inspect\n2. Validate",))
    review = ScriptedPlanReview.approve()

    result = run_planning_loop(
        llm,
        [Message("user", "analyze")],
        review,
        max_revisions=1,
        output=lambda _message: None,
    )

    assert result.approved is True
    assert result.plan == "1. Inspect\n2. Validate"
    assert len(llm.prompts) == 1
    assert review.plans == [result.plan]

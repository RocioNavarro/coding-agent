"""Escenario integral: memoria, RAG suficiente y ausencia de fallback web."""

import json

from agents.orchestrator import MainAgent
from agents.researcher import EvidenceFragment
from core.models import PlanReview
from tests.integration.fakes import (
    FakeMemory, FakeObservability, FakeRAG, FakeWeb, ScriptedPlanner,
    StaticTaskAnalyzer,
)
from tests.integration.scenario_support import explorer_for, researcher_for


def test_rag_evidence_avoids_web_and_remains_traceable(rag_repository) -> None:
    observed = FakeObservability()
    memory = FakeMemory(observability=observed)
    rag = FakeRAG((
        EvidenceFragment(
            "rag", "docs/contract.md#contract",
            "The component returns a normalized value.", 0.98,
        ),
    ), observability=observed)
    web = FakeWeb(observability=observed)
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("analysis", research_required=True),
        plan_generator=ScriptedPlanner(("1. Summarize the recovered contract",)),
        explorer=explorer_for(
            rag_repository.root, ("docs/contract.md",), observability=observed
        ),
        researcher=researcher_for(memory, rag, web),
        implementer=None,
        tester=None,
        reviewer=None,
        observability=observed,
    )

    result = agent.run(
        "Analyze the documented contract",
        lambda _plan: PlanReview("approve"),
        task_id="rag-scenario",
    )

    state = result.task_state
    assert result.status == "completed"
    assert len(memory.queries) == 1
    assert memory.fragments == ()
    assert len(rag.calls) == 1
    assert web.calls == []
    assert any(source.origin == "rag" for source in state.sources)
    assert "[utilizado:rag] docs/contract.md#contract" in result.final_response
    trace_text = next(
        item.removeprefix("RAG trace: ")
        for item in state.observations if item.startswith("RAG trace: ")
    )
    trace = json.loads(trace_text)
    assert [item["chunk_id"] for item in trace["retrieved"]] == ["chunk-1"]
    assert [item["chunk_id"] for item in trace["used"]] == ["chunk-1"]
    names = [event.name for event in observed.events]
    assert "fake-memory-query" in names
    assert "fake-rag-retrieval" in names
    assert "researcher" in names
    assert "fake-web-search" not in names

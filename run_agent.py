"""Composición mínima de MainAgent para un primer análisis de sólo lectura.

Uso:
    python run_agent.py "Explicame la arquitectura de este repositorio"

Lee agent.config.yaml de la raíz del repo para resolver workspace, perfil, RAG y
memoria (hoy configurado para PrintScript, en modo sólo lectura). Arma Explorer y
Researcher; no incluye Implementer/Tester/Reviewer todavía, así que sólo puede
resolver tareas de análisis, nunca de cambio de archivos.

Requiere OPENAI_API_KEY y OPENAI_MODEL en el entorno o en .env.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from agents.explorer import ExplorerAgent
from agents.orchestrator import LLMPlanGenerator, LLMTaskAnalyzer, MainAgent
from agents.project_memory import ProjectMemory
from agents.researcher import ResearcherAgent
from core.llm_client import LLMClientError, ObservedLLMClient, OpenAILLMClient
from core.models import PlanReview
from core.observability import ObservabilityClient, build_observability_client
from core.settings import AgentSettings
from rag.embeddings import HashEmbeddingProvider
from rag.retriever import RagRetriever
from rag.vector_store import JsonVectorStore


def build_main_agent(
    settings: AgentSettings, observability: ObservabilityClient
) -> MainAgent:
    """Arma un MainAgent de sólo lectura (Explorer + Researcher) desde AgentConfig."""
    config = settings.agent_config
    if config is None:
        raise SystemExit(
            "No se encontró agent.config.yaml en la raíz del repo. "
            "Copiá agent.config.example.yaml y adaptalo antes de correr este script."
        )

    llm_client = ObservedLLMClient(OpenAILLMClient(), observability)

    project_memory = None
    if config.memory.enabled:
        project_memory = ProjectMemory(
            config.workspace.path,
            identifier=config.memory.identifier,
            storage_root=config.memory.path,
            observability=observability,
        )

    explorer = ExplorerAgent(
        repository_root=config.workspace.path,
        llm_client=llm_client,
        project_memory=project_memory,
        profile=config.profile,
        observability=observability,
    )

    researcher = None
    if config.rag.enabled and config.rag.sources:
        retriever = RagRetriever(
            embedding_provider=HashEmbeddingProvider(),
            vector_store=JsonVectorStore(config.rag.index_path),
            observability=observability,
        )
        researcher = ResearcherAgent.from_settings(
            llm_client=llm_client,
            project_memory=project_memory,
            knowledge_retriever=retriever,
            settings=settings,
            profile=config.profile,
        )

    return MainAgent(
        task_analyzer=LLMTaskAnalyzer(llm_client),
        plan_generator=LLMPlanGenerator(llm_client),
        explorer=explorer,
        researcher=researcher,
        implementer=None,
        tester=None,
        reviewer=None,
        project_memory=project_memory,
        observability=observability,
        profile=config.profile,
    )


def interactive_plan_review(plan: str) -> PlanReview:
    """Diálogo de aprobación por consola, igual en espíritu al de main.py."""
    print("\n--- Plan propuesto ---")
    print(plan)
    while True:
        answer = input("\n[a]probar / [r]echazar / [m]odificar: ").strip().lower()
        if answer in {"a", "aprobar"}:
            return PlanReview("approve")
        if answer in {"r", "rechazar"}:
            return PlanReview("reject")
        if answer in {"m", "modificar"}:
            modification = input("Modificación solicitada: ").strip()
            if modification:
                return PlanReview("modify", modification)
            print("La modificación no puede estar vacía.")
            continue
        print("Opción inválida. Usá a, r o m.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Corre un análisis de sólo lectura del MainAgent sobre el workspace "
            "configurado en agent.config.yaml."
        )
    )
    parser.add_argument(
        "request", help="Pedido en texto, ej. 'Explicame la arquitectura de este repositorio'."
    )
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        settings = AgentSettings.from_environment()
        observability = build_observability_client()
        agent = build_main_agent(settings, observability)
    except LLMClientError as error:
        print(f"Error de configuración del LLM: {error}", file=sys.stderr)
        return 1

    result = agent.run(args.request, interactive_plan_review)

    print("\n=== Resultado ===")
    print(f"Estado: {result.status}")
    print(f"Agentes usados: {', '.join(result.selected_agents) or 'ninguno'}")
    print(f"Iteraciones: {result.iterations}\n")
    print(result.final_response)

    observability.flush()
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

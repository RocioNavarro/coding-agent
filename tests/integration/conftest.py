"""Fixtures pytest de repositorios integrales aislados."""

from pathlib import Path

import pytest

from tests.integration.repositories import TemporaryRepository, build_repository


@pytest.fixture()
def repository_factory(tmp_path: Path):
    def create(kind: str) -> TemporaryRepository:
        return build_repository(tmp_path, kind)
    return create


@pytest.fixture()
def analysis_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("analysis")


@pytest.fixture()
def simple_change_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("simple_change")


@pytest.fixture()
def rag_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("rag_docs")


@pytest.fixture()
def persistent_memory_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("persistent_memory")


@pytest.fixture()
def failed_command_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("failed_command")


@pytest.fixture()
def blocked_operation_repository(repository_factory) -> TemporaryRepository:
    return repository_factory("blocked_operation")

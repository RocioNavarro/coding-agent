"""Tests de la interfaz acotada de análisis real."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_agent


def test_keyboard_interrupt_is_clean(capsys) -> None:
    agent = Mock()
    agent.run.side_effect = KeyboardInterrupt
    with (
        patch.object(run_agent, "load_dotenv"),
        patch.object(run_agent.AgentSettings, "from_environment", return_value=Mock()),
        patch.object(run_agent, "build_observability_client", return_value=Mock()),
        patch.object(run_agent, "build_main_agent", return_value=agent),
    ):
        exit_code = run_agent.main(["pedido"])

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "Operación cancelada por el usuario." in captured.err
    assert "Traceback" not in captured.err


def test_progress_task_analyzer_is_visible(capsys) -> None:
    delegate = Mock()
    delegate.analyze.return_value = "resultado"

    result = run_agent.ProgressTaskAnalyzer(delegate).analyze("pedido")

    assert result == "resultado"
    assert "[LLM] Analizando la tarea..." in capsys.readouterr().out


def test_progress_plan_generator_is_visible(capsys) -> None:
    delegate = Mock()
    delegate.generate.return_value = "plan"

    result = run_agent.ProgressPlanGenerator(delegate).generate(
        SimpleNamespace(), feedback=("ajuste",)
    )

    assert result == "plan"
    delegate.generate.assert_called_once()
    assert "[LLM] Generando el plan..." in capsys.readouterr().out


def test_progress_runner_is_visible(capsys) -> None:
    delegate = Mock()
    delegate.run.return_value = "evidencia"

    result = run_agent.ProgressRunner(
        delegate, "[Explorer] Inspeccionando el repositorio..."
    ).run("pedido")

    assert result == "evidencia"
    assert "[Explorer] Inspeccionando el repositorio..." in capsys.readouterr().out

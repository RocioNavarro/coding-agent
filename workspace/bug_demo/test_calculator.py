"""Tests de la calculadora de demostración."""

from calculator import divide


def test_divide() -> None:
    assert divide(10, 2) == 5

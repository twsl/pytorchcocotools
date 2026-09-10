"""Conftest for profiling tests."""

from typing import cast

from _pytest.terminal import TerminalReporter
import pytest


@pytest.fixture
def terminal_writer(request: pytest.FixtureRequest) -> TerminalReporter:
    """Return the pytest TerminalReporter for direct output bypassing capture."""
    return cast(TerminalReporter, request.config.pluginmanager.get_plugin("terminalreporter"))

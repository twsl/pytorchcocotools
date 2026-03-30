"""Conftest for profiling tests."""

from _pytest.terminal import TerminalReporter
import pytest


@pytest.fixture
def terminal_writer(request: pytest.FixtureRequest) -> TerminalReporter:
    """Return the pytest TerminalReporter for direct output bypassing capture."""
    return request.config.pluginmanager.get_plugin("terminalreporter")

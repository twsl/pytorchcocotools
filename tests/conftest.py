import pytest
from pytest import FixtureRequest
import torch


@pytest.fixture(
    scope="session",
    params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def device(request: FixtureRequest) -> str:
    """Return the device to use for the tests (parametrized)."""
    return request.param

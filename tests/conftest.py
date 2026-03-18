from typing import Any

import pytest
from pytest import FixtureRequest
from pytest_park.pytest_benchmark import default_pytest_benchmark_group_stats
import torch


@pytest.fixture(
    scope="session",
    params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def device(request: FixtureRequest) -> str:
    """Return the device to use for the tests (parametrized)."""
    return request.param


def pytest_benchmark_group_stats(
    config: Any, benchmarks: list[Any], group_by: str
) -> list[tuple[str | None, list[Any]]]:
    """Grouping for pytest-benchmark using pytest-park.

    Groups benchmarks by clustering methods ending with _np and _pt together
    using pytest-park's default grouping method.
    """
    return default_pytest_benchmark_group_stats(
        config,
        benchmarks,
        group_by,
        original_postfix="_np",
        reference_postfix="_pt",
        group_values_by_postfix={
            "_np": "numpy",
            "_pt": "pytorch",
            "_tm": "torchmetrics",
            "none": "unlabeled",
        },
        ignore_params=["device"],
    )

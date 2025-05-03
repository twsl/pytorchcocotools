import pytest

from pytorchcocotools.__about__ import __version__


def test_version() -> None:
    assert __version__ is not None


@pytest.mark.xfail(reason="expected to fail")
def test_failed(device: str) -> None:
    assert device is not None
    assert 1 == 2

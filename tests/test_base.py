from pytorchcocotools.__about__ import __version__


def test_version():
    assert __version__ is not None


def test_failed():
    assert 1 == 2

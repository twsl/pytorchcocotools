import functools

import torch

from pytorchcocotools.utils.callable import resolve_actual_function


def test_resolve_actual_function_returns_original_function() -> None:
    def identity(value: int) -> int:
        return value

    assert resolve_actual_function(identity) is identity


def test_resolve_actual_function_unwraps_functools_wrapped_function() -> None:
    def base(value: int) -> int:
        return value + 1

    @functools.wraps(base)
    def wrapped(value: int) -> int:
        return base(value)

    assert resolve_actual_function(wrapped) is base


def test_resolve_actual_function_unwraps_torch_decorated_function() -> None:
    @torch.inference_mode()
    def decorated(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    resolved = resolve_actual_function(decorated)

    assert resolved is not decorated
    assert resolved.__name__ == "decorated"


def test_resolve_actual_function_unwraps_torch_compiled_function() -> None:
    @torch.compile(dynamic=True, mode="reduce-overhead")
    def compiled(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    resolved = resolve_actual_function(compiled)

    assert resolved is not compiled
    assert resolved.__name__ == "compiled"

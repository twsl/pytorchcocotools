from collections.abc import Callable
import inspect
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_TORCH_WRAPPED_CALLABLE_ATTRIBUTES = ("_torchdynamo_orig_callable",)


def resolve_actual_function(func: F) -> F:
    """Resolve the innermost Python function for a decorated callable."""
    resolved: object = inspect.unwrap(func)
    seen = {id(resolved)}

    while True:
        next_resolved = None
        for attr_name in _TORCH_WRAPPED_CALLABLE_ATTRIBUTES:
            candidate = getattr(resolved, attr_name, None)
            if callable(candidate):
                next_resolved = inspect.unwrap(candidate)
                break

        if next_resolved is None or id(next_resolved) in seen:
            return resolved  # pyright: ignore[reportReturnType]

        seen.add(id(next_resolved))
        resolved = next_resolved

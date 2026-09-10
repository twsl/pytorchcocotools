"""Callable protocols describing the signatures of the shared fixtures in this package.

``collections.abc.Callable`` cannot express default values, keyword arguments or
keyword-only parameters, so fixture return types are declared as explicit protocols
instead of the untyped ``Callable[..., T]`` shorthand.
"""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class ComputeFn(Protocol):
    """Runs ``update`` + ``compute`` on a freshly constructed metric.

    Implemented by the ``tm_compute`` and ``pt_compute`` fixtures. Keyword arguments are
    forwarded to the metric constructor.
    """

    def __call__(
        self,
        preds: list[dict[str, Tensor]],
        target: list[dict[str, Tensor]],
        **kwargs: Any,
    ) -> dict[str, Tensor]: ...


class AssertMapClose(Protocol):
    """Asserts that the scalar entries of two metric result dicts match.

    Implemented by the ``assert_map_close`` fixture.
    """

    def __call__(
        self,
        result: dict[str, Tensor],
        reference: dict[str, Tensor],
        atol: float = 1e-4,
    ) -> None: ...


class MakeRandomBoxes(Protocol):
    """Builds *n* random non-degenerate xyxy boxes inside an ``img_w`` x ``img_h`` canvas.

    Implemented by the ``make_random_boxes`` fixture.
    """

    def __call__(
        self,
        n: int,
        img_w: float = 640.0,
        img_h: float = 480.0,
        seed: int = 0,
    ) -> Tensor: ...


class MakeStressBatch(Protocol):
    """Builds ``preds`` + ``target`` lists with *n_boxes_per_image* boxes in each image.

    Implemented by the ``make_stress_batch`` fixture.
    """

    def __call__(
        self,
        n_images: int,
        n_boxes_per_image: int,
        n_classes: int = 10,
        seed: int = 42,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]: ...

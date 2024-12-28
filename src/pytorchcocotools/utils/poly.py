from collections.abc import Mapping, Sequence
from typing import Any, Optional, Self, Union

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import TVTensor


# TODO: migrate once https://github.com/pytorch/vision/pull/8817 is merged
class Polygons(TVTensor):
    """:class:`torch.Tensor` subclass for polygons with shape ``[N, L, 2]``.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    canvas_size: tuple[int, int]

    @property
    def num_coordinates(self) -> int:
        return self.shape[0]

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        canvas_size: tuple[int, int],
        check_dims: bool = True,
    ) -> Self:  # type: ignore[override]
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 3:
                raise ValueError(f"Expected a 1D, 2D or 3D tensor, got {tensor.ndim}D")
            if tensor.shape[-1] != 2:
                raise ValueError(f"Expected a tensor of shape (N, L, 2), not {tensor.shape}")
        polygon = tensor.as_subclass(cls)
        polygon.canvas_size = canvas_size
        return polygon

    def __new__(
        cls,
        data: Tensor | list[Tensor] | tuple[Tensor, ...],
        *,
        canvas_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        data_nested = torch.nested.as_nested_tensor(data, dtype=dtype)  # TODO: add device
        tensor = cls._to_tensor(data_nested, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        # If there are BoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, Polygon))
        canvas_size = first_bbox_from_args.canvas_size

        if isinstance(output, torch.Tensor) and not isinstance(output, Polygons):
            output = Polygon._wrap(output, canvas_size=canvas_size, check_dims=False)
        elif isinstance(output, tuple | list):
            output = type(output)(Polygons._wrap(part, canvas_size=canvas_size, check_dims=False) for part in output)  # pyright: ignore[reportCallIssue]
        return output  # pyright: ignore[reportReturnType]

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(canvas_size=self.canvas_size)


class Polygon(TVTensor):
    """:class:`torch.Tensor` subclass for polygon with shape ``[N, 2]``.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    canvas_size: tuple[int, int]

    @property
    def num_coordinates(self) -> int:
        return self.shape[0]

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        canvas_size: tuple[int, int],
        check_dims: bool = True,
    ) -> Self:  # type: ignore[override]
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")
            if tensor.shape[-1] != 2:
                raise ValueError(f"Expected a tensor of shape (..., 2), not {tensor.shape}")
        polygon = tensor.as_subclass(cls)
        polygon.canvas_size = canvas_size
        return polygon

    def __new__(
        cls,
        data: Any,
        *,
        canvas_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Self:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size)

    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        # If there are BoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, Polygon))
        canvas_size = first_bbox_from_args.canvas_size

        if isinstance(output, torch.Tensor) and not isinstance(output, Polygon):
            output = Polygon._wrap(output, canvas_size=canvas_size, check_dims=False)
        elif isinstance(output, tuple | list):
            output = type(output)(Polygon._wrap(part, canvas_size=canvas_size, check_dims=False) for part in output)  # pyright: ignore[reportCallIssue]
        return output  # pyright: ignore[reportReturnType]

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(canvas_size=self.canvas_size)

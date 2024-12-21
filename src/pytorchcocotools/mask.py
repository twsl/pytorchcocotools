from typing import Literal

from jaxtyping import Num
import torch
from torch import Tensor
from torchvision import tv_tensors as tv

import pytorchcocotools._mask as _mask
from pytorchcocotools.internal.entities import Bool, IoUObject, PyObj, RleObj, RleObjs, TorchDevice


def iou(
    dt: IoUObject,
    gt: IoUObject,
    pyiscrowd: list[Bool],
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Compute intersection over union between masks.

    Note:
    Finally, a note about the intersection over union (iou) computation.
    The standard iou of a ground truth (gt) and detected (dt) object is
    .. code-block:: python
        iou(gt,dt) = area(intersect(gt,dt)) / area(union(gt,dt))

    For "crowd" regions, we use a modified criteria. If a gt object is
    marked as "iscrowd", we allow a dt to match any subregion of the gt.
    Choosing gt' in the crowd gt that best matches the dt can be done using
    gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)
    For crowd gt regions we use this modified criteria above for the iou.

    Args:
        dt: The detected objects.
        gt: The ground truth objects.
        pyiscrowd: A list of booleans indicating whether the ground truth objects are crowds.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The intersection over union between the detected and ground truth objects.
    """
    is_crowd = [bool(is_c) for is_c in pyiscrowd]
    return _mask.iou(dt, gt, is_crowd, device=device, requires_grad=requires_grad)


def merge(
    rleObjs: RleObjs,  # noqa: N803
    intersect: bool = False,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RleObj:
    """Compute union or intersection of encoded masks.

    Args:
        rleObjs: The masks to merge.
        intersect: Whether to compute the intersection.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The merged mask.
    """
    return _mask.merge(rleObjs, intersect, device=device, requires_grad=requires_grad)


def frPyObjects(  # noqa: N802
    pyobj: PyObj,
    h: int,
    w: int,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RleObjs | RleObj:
    """Convert (list of) polygon, bbox, or uncompressed RLE to encoded RLE mask.

    Args:
        pyobj: The object to convert.
        h: The height of the mask.
        w: The width of the mask.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The encoded mask.
    """
    return _mask.frPyObjects(pyobj, h, w, device=device, requires_grad=requires_grad)


def encode(
    bimask: tv.Mask,
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> RleObjs:
    """Encode binary masks using RLE.

    Note:
    Requires channel last order input.

    Warning:
    This functions differs from the original implementation and always returns a list of encoded masks.

    Args:
        bimask: The binary mask to encode.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The encoded mask.
    """
    if len(bimask.shape) == 3:
        return _mask.encode(bimask, device=device, requires_grad=requires_grad)
    elif len(bimask.shape) == 2:
        # tv.wrap(bimask, like=bimask)
        bimask_unsq = tv.wrap(bimask.unsqueeze(0), like=bimask, device=device, requires_grad=requires_grad)
        # bimask = torch.unsqueeze(bimask, dim=-1)  # masks expected to be in format [h,w,n]
        # return _mask.encode(bimask)[0]  # original implementation
        return _mask.encode(bimask_unsq, device=device, requires_grad=requires_grad)  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError("encode expects a binary mask or batch of binary masks")


def decode(
    rleObjs: RleObj | RleObjs,  # noqa: N803
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.Mask:
    """Decode binary masks encoded via RLE.

    Note:
    Returns channel last order output.

    Warning:
    This functions differs from the original implementation and always returns a Tensor batch of decoded masks.

    Args:
        rleObjs: The encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The decoded mask.
    """
    if isinstance(rleObjs, list):
        return _mask.decode(rleObjs, device=device, requires_grad=requires_grad)
    else:
        # return _mask.decode([rleObjs])[:, :, 0]  # original implementation
        return _mask.decode([rleObjs], device=device, requires_grad=requires_grad)


def area(
    rleObjs: RleObj | RleObjs,  # noqa: N803
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> list[int]:
    """Compute area of encoded masks.

    Warning:
    This functions differs from the original implementation and always returns a list of areas.

    Args:
        rleObjs: The encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The areas of the masks.
    """
    if isinstance(rleObjs, list):
        return _mask.area(rleObjs, device=device, requires_grad=requires_grad)
    else:
        # return _mask.area([rleObjs])[0]  # original implementation
        return _mask.area([rleObjs], device=device, requires_grad=requires_grad)


def toBbox(  # noqa: N803, N802
    rleObjs: RleObj | RleObjs,  # noqa: N803, N802
    *,
    device: TorchDevice | None = None,
    requires_grad: bool | None = None,
) -> tv.BoundingBoxes:
    """Get bounding boxes surrounding encoded masks.

    Warning:
    This functions differs from the original implementation and always returns a Tensor batch of bounding boxes.

    Args:
        rleObjs: The encoded masks.
        device: The desired device of the bounding boxes.
        requires_grad: Whether the bounding boxes require gradients.

    Returns:
        The bounding boxes.
    """
    if isinstance(rleObjs, list):
        return _mask.toBbox(rleObjs, device=device, requires_grad=requires_grad)
    else:
        # return _mask.toBbox([rleObjs])[0]  # original implementation
        return _mask.toBbox([rleObjs], device=device, requires_grad=requires_grad)

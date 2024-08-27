from typing import Literal

import torch
from torch import Tensor

import pytorchcocotools._mask as _mask
from pytorchcocotools.internal.entities import RleObj, RleObjs


def iou(dt: Tensor, gt: Tensor, pyiscrowd: list[bool | Literal[0, 1]]) -> Tensor:  # TODO: add better type hints
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

    Returns:
        The intersection over union between the detected and ground truth objects.
    """
    is_crowd = [bool(is_c) for is_c in pyiscrowd]
    return _mask.iou(dt, gt, is_crowd)


def merge(rleObjs: RleObjs, intersect: bool = False) -> RleObj:  # noqa: N803
    """Compute union or intersection of encoded masks.

    Args:
        rleObjs: The masks to merge.
        intersect: Whether to compute the intersection.

    Returns:
        The merged mask.
    """
    return _mask.merge(rleObjs, intersect)


def frPyObjects(pyobj: Tensor | list | dict, h: int, w: int) -> RleObjs | RleObj:  # noqa: N802
    """Convert (list of) polygon, bbox, or uncompressed RLE to encoded RLE mask.

    Args:
        pyobj: The object to convert.
        h: The height of the mask.
        w: The width of the mask.

    Returns:
        The encoded mask.
    """
    return _mask.frPyObjects(pyobj, h, w)


def encode(bimask: Tensor) -> RleObj | RleObjs:
    """Encode binary masks using RLE.

    Note:
    Requires channel last order input.

    Args:
        bimask: The binary mask to encode.

    Returns:
        The encoded mask.
    """
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        bimask = torch.unsqueeze(bimask, dim=-1)  # masks expected to be in format [h,w,n]
        return _mask.encode(bimask)[0]
    else:
        raise ValueError("encode expects a binary mask or batch of binary masks")


def decode(rleObjs: RleObj | RleObjs) -> Tensor:  # noqa: N803
    """Decode binary masks encoded via RLE.

    Note:
    Returns channel last order output.

    Args:
        rleObjs: The encoded masks.

    Returns:
        The decoded mask.
    """
    return _mask.decode(rleObjs) if isinstance(rleObjs, list) else _mask.decode(RleObjs([rleObjs]))[:, :, 0]


def area(rleObjs: RleObj | RleObjs) -> list[int] | int:  # noqa: N803
    """Compute area of encoded masks."""
    return _mask.area(rleObjs) if isinstance(rleObjs, list) else _mask.area([rleObjs])[0]


def toBbox(rleObjs: RleObj | RleObjs) -> Tensor:  # noqa: N803, N802
    """Get bounding boxes surrounding encoded masks."""
    return _mask.toBbox(rleObjs) if isinstance(rleObjs, list) else _mask.toBbox([rleObjs])[0]

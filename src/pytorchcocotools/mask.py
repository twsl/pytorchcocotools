from typing import Literal

import torch
from torch import Tensor

import pytorchcocotools._mask as _mask
from pytorchcocotools.internal.entities import BB, IoUObject, Mask, PyObj, RleObj, RleObjs


def iou(
    dt: IoUObject, gt: IoUObject, pyiscrowd: list[bool] | list[Literal[0, 1]]
) -> Tensor:  # TODO: add better type hints
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


def frPyObjects(pyobj: PyObj, h: int, w: int) -> RleObjs | RleObj:  # noqa: N802
    """Convert (list of) polygon, bbox, or uncompressed RLE to encoded RLE mask.

    Args:
        pyobj: The object to convert.
        h: The height of the mask.
        w: The width of the mask.

    Returns:
        The encoded mask.
    """
    return _mask.frPyObjects(pyobj, h, w)


def encode(bimask: Mask) -> RleObjs:
    """Encode binary masks using RLE.

    Note:
    Requires channel last order input.

    Warning:
    This functions differs from the original implementation and always returns a list of encoded masks.

    Args:
        bimask: The binary mask to encode.

    Returns:
        The encoded mask.
    """
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        bimask = torch.unsqueeze(bimask, dim=-1)  # masks expected to be in format [h,w,n]
        # return _mask.encode(bimask)[0]  # original implementation
        return _mask.encode(bimask)
    else:
        raise ValueError("encode expects a binary mask or batch of binary masks")


def decode(rleObjs: RleObj | RleObjs) -> Mask:  # noqa: N803
    """Decode binary masks encoded via RLE.

    Note:
    Returns channel last order output.

    Warning:
    This functions differs from the original implementation and always returns a Tensor batch of decoded masks.

    Args:
        rleObjs: The encoded masks.

    Returns:
        The decoded mask.
    """
    if isinstance(rleObjs, list):
        return _mask.decode(rleObjs)
    else:
        # return _mask.decode([rleObjs])[:, :, 0]  # original implementation
        return _mask.decode([rleObjs])


def area(rleObjs: RleObj | RleObjs) -> list[int]:  # noqa: N803
    """Compute area of encoded masks.

    Warning:
    This functions differs from the original implementation and always returns a list of areas.

    Args:
        rleObjs: The encoded masks.

    Returns:
        The areas of the masks.
    """
    if isinstance(rleObjs, list):
        return _mask.area(rleObjs)
    else:
        # return _mask.area([rleObjs])[0]  # original implementation
        return _mask.area([rleObjs])


def toBbox(rleObjs: RleObj | RleObjs) -> BB:  # noqa: N803, N802
    """Get bounding boxes surrounding encoded masks.

    Warning:
    This functions differs from the original implementation and always returns a Tensor batch of bounding boxes.

    Args:
        rleObjs: The encoded masks.

    Returns:
        The bounding boxes.
    """
    if isinstance(rleObjs, list):
        return _mask.toBbox(rleObjs)
    else:
        # return _mask.toBbox([rleObjs])[0]  # original implementation
        return _mask.toBbox([rleObjs])

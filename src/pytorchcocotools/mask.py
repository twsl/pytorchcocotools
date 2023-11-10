from ctypes import ArgumentError
from typing import Union

import pytorchcocotools._mask as _mask
from pytorchcocotools._maskApi import (
    BB,
    RLE,
    Mask,
    Masks,
    RleObj,
    RleObjs,
    RLEs,
)
import torch
from torch import Tensor

# Interface for manipulating masks stored in RLE format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE
# first divides a vector (or vectorized image) into a series of piecewise
# constant regions and then for each piece simply stores the length of
# that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
# be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
# (note that the odd counts are always the numbers of zeros). Instead of
# storing the counts directly, additional compression is achieved with a
# variable bitrate representation based on a common scheme called LEB128.
#
# Compression is greatest given large piecewise constant regions.
# Specifically, the size of the RLE is proportional to the number of
# *boundaries* in M (or for an image the number of boundaries in the y
# direction). Assuming fairly simple shapes, the RLE representation is
# O(sqrt(n)) where n is number of pixels in the object. Hence space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE
# (without need for decoding). This includes computations such as area,
# union, intersection, etc. All of these operations are linear in the
# size of the RLE, in other words they are O(sqrt(n)) where n is the area
# of the object. Computing these operations on the original mask is O(n).
# Thus, using the RLE can result in substantial computational savings.
#
# Usage:
#  Rs     = encode( masks )
#  masks  = decode( Rs )
#  R      = merge( Rs, intersect=false )
#  o      = iou( dt, gt, iscrowd )
#  a      = area( Rs )
#  bbs    = toBbox( Rs )
#  Rs     = frPyObjects( [pyObjects], h, w )
#
# In the API the following formats are used:
#  Rs      - [dict] Run-length encoding of binary masks
#  R       - dict Run-length encoding of binary mask
#  masks   - [hxwxn] Binary mask(s) (must have type torch.ndarray(dtype=uint8) in column-major order)
#  iscrowd - [nx1] list of torch.ndarray. 1 indicates corresponding gt image has crowd region to ignore
#  bbs     - [nx4] Bounding box(es) stored as [x y w h]
#  poly    - Polygon stored as [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
#  dt,gt   - May be either bounding boxes or encoded masks
# Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).


def iou(dt: Tensor, gt: Tensor, pyiscrowd: list[bool]) -> Tensor:
    """Compute intersection over union between masks.

    Finally, a note about the intersection over union (iou) computation.
    The standard iou of a ground truth (gt) and detected (dt) object is
    iou(gt,dt) = area(intersect(gt,dt)) / area(union(gt,dt))
    For "crowd" regions, we use a modified criteria. If a gt object is
    marked as "iscrowd", we allow a dt to match any subregion of the gt.
    Choosing gt' in the crowd gt that best matches the dt can be done using
    gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
    iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)
    For crowd gt regions we use this modified criteria above for the iou.
    """
    return _mask.iou(dt, gt, pyiscrowd)


def merge(rleObjs: RleObjs, intersect: bool = False) -> RleObj:  # noqa: N803
    """Compute union or intersection of encoded masks."""
    return _mask.merge(rleObjs, intersect)


def frPyObjects(pyobj: Tensor | list | dict, h: int, w: int) -> RleObjs:  # noqa: N802
    """Convert polygon, bbox, and uncompressed RLE to encoded RLE mask."""
    return _mask.frPyObjects(pyobj, h, w)


def encode(bimask: Tensor) -> RleObj | RleObjs:
    """Encode binary masks using RLE. Requires channel last order input."""
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        bimask = torch.unsqueeze(bimask, dim=-1)
        return _mask.encode(bimask)[0]


def decode(rleObjs: RleObj | RleObjs) -> Tensor:  # noqa: N803
    """Decode binary masks encoded via RLE. Returns channel last order output."""
    return _mask.decode(rleObjs) if isinstance(rleObjs, list) else _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs: RleObj | RleObjs) -> Tensor:  # noqa: N803
    """Compute area of encoded masks."""
    return _mask.area(rleObjs) if isinstance(rleObjs, list) else _mask.area([rleObjs])[0]


def toBbox(rleObjs: RleObj | RleObjs) -> Tensor:  # noqa: N803, N802
    """Get bounding boxes surrounding encoded masks."""
    return _mask.toBbox(rleObjs) if isinstance(rleObjs, list) else _mask.toBbox([rleObjs])[0]

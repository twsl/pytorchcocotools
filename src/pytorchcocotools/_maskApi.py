from dataclasses import dataclass

import torch
from torch import Tensor


class BB(Tensor):
    pass


class RLE:
    def __init__(self, h: int = 0, w: int = 0, m: int = 0, cnts: Tensor = None):
        """Internal run length encoded representation.

        Args:
            h: The mask height. Defaults to 0.
            w: The mask width. Defaults to 0.
            m: The number of rle entries. Defaults to 0.
            cnts: The rle entries. Defaults to None.
        """
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts if cnts is not None else torch.zeros(m, dtype=torch.int32)


class RLEs(list[RLE]):
    def __init__(self, rles: list[RLE], n: int = None):
        self.n = n if n is not None else len(rles) if len(rles) > 0 else 0
        super().__init__(rles)


class Mask(Tensor):
    """# hxwxn binary mask, in column-major order.

    Args:
        Tensor: _description_
    """

    pass


class Masks(list[Mask]):
    def __init__(self, masks: list[Mask], h: int = None, w: int = None, n: int = None):
        self.h = h if h is not None else masks[0].shape[0] if len(masks) > 0 else 0
        self.w = w if w is not None else masks[0].shape[1] if len(masks) > 0 else 0
        self.n = n if n is not None else len(masks) if len(masks) > 0 else 0
        super().__init__(masks)


@dataclass
class RleObj:
    size: tuple[int, int]
    counts: bytes

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __delitem__(self, key):
        return delattr(self, key)

    def __iter__(self):
        for key in self.__dataclass_fields__:
            yield key, getattr(self, key)

    def __len__(self):
        return len(self.__dataclass_fields__)


class RleObjs(list[RleObj]):
    pass


def rleEncode(mask: Mask, h: int, w: int, n: int) -> RleObjs:  # noqa: N802
    """Encode binary masks using RLE.

    Args:
        mask: _description_
        h: _description_
        w: _description_
        n: _description_

    Returns:
        _description_
    """
    pass


def rleDecode(R: RLE, n: int) -> Mask:  # noqa: N802, N803
    """Decode binary masks encoded via RLE.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    pass


def rleMerge(R: RLEs, n: int, intersect: bool) -> RLEs:  # noqa: N802, N803
    """Compute union or intersection of encoded masks.

    Args:
        R: _description_
        n: _description_
        intersect: _description_

    Returns:
        _description_
    """
    pass


def rleArea(R: RLEs, n: int) -> Tensor:  # noqa: N802, N803
    """Compute area of encoded masks.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    pass


def rleIou(dt: RLEs, gt: RLEs, m: int, n: int, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between masks.

    Args:
        dt: _description_
        gt: _description_
        m: _description_
        n: _description_
        iscrowd: _description_

    Returns:
        _description_
    """
    pass


# Compute non-maximum suppression between bounding masks.
def rleNms(dt: RLE, n: int, keep: list[int], thr: float):
    pass


def bbIou(dt: BB, gt: BB, m: int, n: int, iscrowd: list[bool]) -> Tensor:  # noqa: N802
    """Compute intersection over union between bounding boxes.

    Args:
        dt: _description_
        gt: _description_
        m: _description_
        n: _description_
        iscrowd: _description_

    Returns:
        _description_
    """
    pass


# Compute non-maximum suppression between bounding boxes.
def bbNms(dt: BB, n: int, keep: list[int], thr: float):
    pass


def rleToBbox(R: RLEs, n: int) -> BB:  # noqa: N802, N803
    """Get bounding boxes surrounding encoded masks.

    Args:
        R: _description_
        n: _description_

    Returns:
        _description_
    """
    pass


def rleFrBbox(bb: BB, h: int, w: int, n: int) -> RLEs:  # noqa: N802
    """Convert bounding boxes to encoded masks.

    Args:
        bb: _description_
        h: _description_
        w: _description_
        n: _description_

    Returns:
        _description_
    """
    pass


def rleFrPoly(xy: Tensor, k: int, h: int, w: int) -> RLE:  # noqa: N802
    """Convert polygon to encoded mask.

    Args:
        xy: _description_
        k: _description_
        h: _description_
        w: _description_

    Returns:
        _description_
    """
    pass


def rleToString(R: RLE) -> bytes:
    """Get compressed string representation of encoded mask.

    Args:
        R: _description_

    Returns:
        _description_
    """
    pass


def rleFrString(s: bytes, h: int, w: int) -> RLE:
    """Convert from compressed string representation of encoded mask.

    Args:
        s: _description_
        h: _description_
        w: _description_

    Returns:
        _description_
    """
    pass

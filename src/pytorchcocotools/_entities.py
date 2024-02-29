from dataclasses import dataclass

import torch
from torch import Tensor


class BB(Tensor):
    pass


class RLE:
    def __init__(self, h: int = 0, w: int = 0, m: int = 0, cnts: torch.Tensor = None):
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
class RleObj(dict):
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

from typing import Literal, TypeAlias

import torch
from torch import Tensor

from pytorchcocotools.utils.dataclass import dataclass_dict

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]

RangeLabels: TypeAlias = list[RangeLabel]

Range: TypeAlias = tuple[int, int]

Ranges: TypeAlias = list[Range]

IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]

BB: TypeAlias = Tensor

Mask: TypeAlias = Tensor  # hxwxn binary mask, in column-major order

IsCrowd: TypeAlias = list[bool | Literal[0, 1]]

Poly: TypeAlias = list[float]


class RLE:
    def __init__(self, h: int = 0, w: int = 0, m: int = 0, cnts: torch.Tensor | None = None):
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts if cnts is not None else torch.zeros(m, dtype=torch.int32)


RLEs: TypeAlias = list[RLE]


class Masks(list[Mask]):
    def __init__(self, masks: list[Mask], h: int | None = None, w: int | None = None, n: int | None = None):
        self.h = h if h is not None else masks[0].shape[0] if len(masks) > 0 else 0
        self.w = w if w is not None else masks[0].shape[1] if len(masks) > 0 else 0
        self.n = n if n is not None else len(masks) if len(masks) > 0 else 0
        super().__init__(masks)


@dataclass_dict
class RleObj(dict):
    size: tuple[int, int]
    counts: bytes


RleObjs: TypeAlias = list[RleObj]

from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.utils.dataclass import dataclass_dict
from pytorchcocotools.utils.poly import Polygon

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]

RangeLabels: TypeAlias = list[RangeLabel]

Range: TypeAlias = tuple[int, int]

Ranges: TypeAlias = list[Range]

IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]

Poly: TypeAlias = list[float]  # list of alternating coordinates


class RLE:
    def __init__(self, h: int, w: int, cnts: Tensor) -> None:
        self.h = h
        self.w = w
        self.canvas_size = (h, w)
        self.cnts = cnts  # rle tensor [N, 2] consecutive (start, length) pairs


RLEs: TypeAlias = list[RLE]

IoUObject: TypeAlias = RLEs | list[float] | tv.BoundingBoxes


@dataclass_dict
class RleObj(dict):
    size: tuple[int, int]
    counts: bytes | str


RleObjs: TypeAlias = list[RleObj]

PyObj: TypeAlias = (
    tv.BoundingBoxes | Tensor | list[list[int]] | list[list[float]] | Poly | list[Poly] | RleObjs | RleObj
)

TorchDevice: TypeAlias = torch.device | str | int

Bool: TypeAlias = bool | Literal[0, 1]

Bools: TypeAlias = list[Bool] | list[bool]

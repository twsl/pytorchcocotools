from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torchvision import tv_tensors as tv

from pytorchcocotools.utils.dataclass import dataclass_dict

type RangeLabel = Literal["all", "small", "medium", "large"]

type RangeLabels = list[RangeLabel]

type Range = tuple[int, int]

type Ranges = list[Range]

type IoUType = Literal["segm", "bbox", "keypoints"]

type Poly = list[float]  # list of alternating coordinates


class RLE:
    def __init__(self, h: int, w: int, cnts: Tensor) -> None:
        self.h = h
        self.w = w
        self.canvas_size = (h, w)
        self.cnts = cnts  # rle tensor [N, 2] consecutive (start, length) pairs


# Called at runtime as a list constructor, e.g. `RLEs([r])`; a `type` alias would not be callable.
RLEs: TypeAlias = list[RLE]  # noqa: UP040


@dataclass_dict
class RleObj(dict):
    size: tuple[int, int]
    counts: bytes | str


# Called at runtime as a list constructor, e.g. `RleObjs(objs)`; a `type` alias would not be callable.
RleObjs: TypeAlias = list[RleObj]  # noqa: UP040

type IoUObject = RleObjs | list[float] | tv.BoundingBoxes


type PyObj = tv.BoundingBoxes | Tensor | list[list[int]] | list[list[float]] | Poly | list[Poly] | RleObjs | RleObj

type TorchDevice = torch.device | str | int

type Bool = bool | Literal[0, 1]

type Bools = list[Bool] | list[bool]

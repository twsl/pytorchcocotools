from typing import Literal, TypeAlias

from torch import Tensor

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]

RangeLabels: TypeAlias = list[RangeLabel]

Range: TypeAlias = tuple[int, int]

Ranges: TypeAlias = list[Range]

IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]

BB: TypeAlias = Tensor

Mask: TypeAlias = Tensor  # hxwxn binary mask, in column-major order

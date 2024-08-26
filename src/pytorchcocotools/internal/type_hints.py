from typing import Literal, TypeAlias

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]
RangeLabels: TypeAlias = list[RangeLabel]
Range: TypeAlias = tuple[int, int]
Ranges: TypeAlias = list[Range]
IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]

from __future__ import annotations

from dataclasses import field
from datetime import datetime
from typing import Literal, TypeAlias

import torch

from pytorchcocotools.utils.dataclass import dataclass_dict

RangeLabel: TypeAlias = Literal["all", "small", "medium", "large"]
RangeLabels: TypeAlias = list[RangeLabel]
Range: TypeAlias = tuple[int, int]
Ranges: TypeAlias = list[Range]
IoUType: TypeAlias = Literal["segm", "bbox", "keypoints"]


class Params:
    """Params for coco evaluation api."""

    def setDetParams(self):  # noqa: N802
        self.imgIds: list[int] = []
        self.catIds: list[int] = []
        # torch.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)
        self.recThrs = torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets: list[int] = [1, 10, 100]
        self.areaRng: Ranges = [
            (int(0**2), int(1e5**2)),
            (int(0**2), int(32**2)),
            (int(32**2), int(96**2)),
            (int(96**2), int(1e5**2)),
        ]
        self.areaRngLbl: RangeLabels = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):  # noqa: N802
        self.imgIds: list[int] = []
        self.catIds: list[int] = []
        # torch.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)
        self.recThrs = torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1)
        self.maxDets = [20]
        self.areaRng: Ranges = [(int(0**2), int(1e5**2)), (int(32**2), int(96**2)), (int(96**2), int(1e5**2))]
        self.areaRngLbl: RangeLabels = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            torch.Tensor(
                [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
            )
            / 10.0
        )

    def __init__(self, iouType: IoUType = "segm") -> None:  # noqa: N803
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise ValueError("iouType not supported.")
        self.iouType: IoUType = iouType


@dataclass_dict
class EvalImgResult(dict):
    image_id: int = -1
    category_id: int = -1
    aRng: Range = field(default_factory=Range)  # noqa: N815
    maxDet: int = 0  # noqa: N815
    dtIds: list[int] = field(default_factory=list[int])  # noqa: N815
    gtIds: list[int] = field(default_factory=list[int])  # noqa: N815
    dtMatches: list[int] = field(default_factory=list[int])  # noqa: N815
    gtMatches: list[int] = field(default_factory=list[int])  # noqa: N815
    dtScores: list[float] = field(default_factory=list[float])  # noqa: N815
    gtIgnore: list[int] = field(default_factory=list[int])  # noqa: N815
    dtIgnore: list[int] = field(default_factory=list[int])  # noqa: N815


@dataclass_dict
class EvalResult(dict):
    """Accumulated evaluation results."""

    params: Params = field(default_factory=Params)
    counts: list[int] = field(default_factory=list[int])
    date: datetime = datetime.now()
    precision: torch.Tensor | None = None
    recall: torch.Tensor | None = None
    scores: torch.Tensor | None = None

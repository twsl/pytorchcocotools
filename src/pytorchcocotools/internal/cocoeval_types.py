from __future__ import annotations

from dataclasses import field
from datetime import datetime
from typing import Literal, TypeAlias

from numpy import rec
import torch

from pytorchcocotools.internal.entities import IoUType, Range, RangeLabels, Ranges
from pytorchcocotools.utils.dataclass import dataclass_dict


class Params:
    """Params for coco evaluation api."""

    imgIds: list[int]  # noqa: N815
    catIds: list[int]  # noqa: N815
    iouThrs: torch.Tensor  # noqa: N815
    recThrs: torch.Tensor  # noqa: N815
    maxDets: list[int]  # noqa: N815
    areaRng: Ranges  # noqa: N815
    areaRngLbl: RangeLabels  # noqa: N815
    useCats: int  # noqa: N815
    kpt_oks_sigmas: torch.Tensor

    def __init__(self, iouType: IoUType = "segm") -> None:  # noqa: N803
        """Initialize Params with default values.

        Args:
            iouType: Type of evaluation. The default is "segm".

        Raises:
            ValueError: If iouType is not supported.
        """
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise ValueError("iouType not supported.")
        self.iouType: IoUType = iouType

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


@dataclass_dict
class EvalImgResult(dict):
    image_id: int = -1
    category_id: int = -1
    aRng: Range = field(default_factory=Range)  # noqa: N815
    maxDet: int = 0  # noqa: N815
    dtIds: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    gtIds: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    dtMatches: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    gtMatches: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    dtScores: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    gtIgnore: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815
    dtIgnore: torch.Tensor = field(default=torch.zeros(0))  # noqa: N815


@dataclass_dict
class EvalResult(dict):
    """Accumulated evaluation results."""

    params: Params = field(default_factory=Params)
    counts: torch.Tensor = field(default=torch.zeros(0))
    date: datetime = datetime.now()
    precision: torch.Tensor = field(default=torch.zeros(0))
    recall: torch.Tensor = field(default=torch.zeros(0))
    scores: torch.Tensor = field(default=torch.zeros(0))

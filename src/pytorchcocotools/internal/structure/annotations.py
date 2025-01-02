from dataclasses import field
from typing import Self, TypeAlias, cast

from pytorchcocotools.internal.entities import Poly
from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.internal.structure.rle import CocoRLE, Segmentation
from pytorchcocotools.utils.dataclass import dataclass_dict


@dataclass_dict
class CocoAnnotationObjectDetection(BaseCocoEntity):
    id: int = -1
    image_id: int = -1
    category_id: int = -1
    segmentation: Segmentation = field(default_factory=list[Poly])
    area: float = 0.0
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    iscrowd: bool = False
    score: float | None = None  # Only used in results
    ignore: bool | None = None  # Only used in results

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        iscrowd = bool(data.get("iscrowd"))
        seg = data.get("segmentation", [])
        segmentations = CocoRLE.from_dict(seg) if iscrowd and isinstance(seg, dict) else cast(list[Poly], seg)
        instance = cls(
            id=int(data.get("id", -1)),
            image_id=int(data.get("image_id", -1)),
            category_id=int(data.get("category_id", -1)),
            segmentation=segmentations,
            area=float(data.get("area", 0.0)),
            bbox=cast(list[float], data.get("bbox", [])),
            iscrowd=iscrowd,
            score=data.get("score"),
        )
        return instance


@dataclass_dict
class CocoAnnotationKeypointDetection(CocoAnnotationObjectDetection):
    keypoints: list[float] = field(default_factory=list[float])
    num_keypoints: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        iscrowd = bool(data.get("iscrowd"))
        seg = data.get("segmentation", [])
        segmentations = CocoRLE.from_dict(seg) if iscrowd and isinstance(seg, dict) else cast(list[Poly], seg)
        instance = cls(
            id=int(data.get("id", -1)),
            image_id=int(data.get("image_id", -1)),
            category_id=int(data.get("category_id", -1)),
            segmentation=segmentations,
            area=float(data.get("area", 0.0)),
            bbox=cast(list[float], data.get("bbox", [])),
            iscrowd=iscrowd,
            score=data.get("score"),
            keypoints=cast(list[float], data.get("keypoints", [])),
            num_keypoints=int(data.get("num_keypoints", 0)),
        )
        return instance


CocoAnnotationDetection: TypeAlias = CocoAnnotationObjectDetection | CocoAnnotationKeypointDetection

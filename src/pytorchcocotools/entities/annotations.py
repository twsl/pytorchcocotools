from __future__ import annotations

from dataclasses import field

from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoRLE(BaseCocoEntity):
    counts: list[float] = field(default_factory=list[float])
    size: tuple[int, int] = field(default_factory=tuple[int, int])

    @classmethod
    def from_dict(cls, data: dict) -> CocoRLE:
        return cls(counts=data.get("counts"), size=data.get("size"))


class CocoAnnotationObjectDetection(BaseCocoEntity):
    id: int = -1
    image_id: int = -1
    category_id: int = -1
    segmentation: list[CocoRLE | list[float]] = field(default_factory=list[CocoRLE | list[float]])
    area: float = 0.0
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    iscrowd: bool = False
    score: float | None = None  # only used in results

    @classmethod
    def from_dict(cls, data: dict) -> CocoAnnotationObjectDetection:
        segmentations = [
            CocoRLE.from_dict(seg) if isinstance(seg, dict) else seg for seg in data.get("segmentation", [])
        ]
        return cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            category_id=data.get("category_id"),
            segmentation=segmentations,
            area=data.get("area"),
            bbox=data.get("bbox"),
            iscrowd=bool(data.get("iscrowd")),
        )


@dataclass_dict
class CocoAnnotationKeypointDetection(CocoAnnotationObjectDetection):
    keypoints: list[float] = field(default_factory=list[float])
    num_keypoints: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> CocoAnnotationKeypointDetection:
        segmentations = [
            CocoRLE.from_dict(seg) if isinstance(seg, dict) else seg for seg in data.get("segmentation", [])
        ]

        return cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            category_id=data.get("category_id"),
            segmentation=segmentations,
            area=data.get("area"),
            bbox=data.get("bbox"),
            iscrowd=bool(data.get("iscrowd")),
            keypoints=data.get("keypoints"),
            num_keypoints=data.get("num_keypoints"),
        )

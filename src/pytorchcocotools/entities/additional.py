from __future__ import annotations

from dataclasses import field

from pytorchcocotools.entities.annotations import CocoRLE
from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.entities.categories import CocoCategoriesObjectDetection
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class ResultAnnotation(BaseCocoEntity):
    image_id: int = -1
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    score: float | None = None
    category_id: int = -1

    @classmethod
    def from_dict(cls, data: dict) -> ResultAnnotation:
        instance = cls(
            image_id=data.get("image_id"),
            bbox=data.get("bbox", []),
            score=data.get("score"),
            category_id=data.get("category_id"),
        )
        return instance


@dataclass_dict
class CocoSegmentInfo(BaseCocoEntity):
    id: int = -1
    category_id: int = -1
    area: float = 0.0
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    iscrowd: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> CocoSegmentInfo:
        instance = cls(
            id=data.get("id"),
            category_id=data.get("category_id"),
            area=data.get("area"),
            bbox=data.get("bbox", []),
            iscrowd=bool(data.get("iscrowd")),
        )
        return instance


@dataclass_dict
class CocoAnnotationPanopticSegmentation(BaseCocoEntity):
    image_id: int = -1
    file_name: str = ""
    segments_info: list[CocoSegmentInfo] = field(default_factory=list[CocoSegmentInfo])

    @classmethod
    def from_dict(cls, data: dict) -> CocoAnnotationPanopticSegmentation:
        instance = cls(
            image_id=data.get("image_id"),
            file_name=data.get("file_name"),
            segments_info=[CocoSegmentInfo.from_dict(seg) for seg in data.get("segments_info", [])],
        )
        return instance


@dataclass_dict
class CocoAnnotationImageCaptioning(BaseCocoEntity):
    id: int = -1
    image_id: int = -1
    caption: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> CocoAnnotationImageCaptioning:
        instance = cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            caption=data.get("caption"),
        )
        return instance


@dataclass_dict
class CocoAnnotationDensePose(BaseCocoEntity):
    id: int = -1
    image_id: int = -1
    category_id: int = -1
    is_crowd: bool = False
    area: float = 0.0
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    dp_I: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_U: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_V: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_x: list[float] = field(default_factory=list[float])
    dp_y: list[float] = field(default_factory=list[float])
    dp_masks: list[CocoRLE] = field(default_factory=list[CocoRLE])

    @classmethod
    def from_dict(cls, data: dict) -> CocoAnnotationDensePose:
        instance = cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            category_id=data.get("category_id"),
            is_crowd=bool(data.get("is_crowd")),
            area=data.get("area"),
            bbox=data.get("bbox", []),
            dp_I=data.get("dp_I", []),
            dp_U=data.get("dp_U", []),
            dp_V=data.get("dp_V", []),
            dp_x=data.get("dp_x", []),
            dp_y=data.get("dp_y", []),
            dp_masks=[CocoRLE.from_dict(mask) for mask in data.get("dp_masks", [])],
        )
        return instance


@dataclass_dict
class CocoCategoriesPanopticSegmentation(CocoCategoriesObjectDetection):
    isthing: bool | None = None  # noqa: N815
    color: list[int] = field(default_factory=list[int])  # [R,G,B]

    @classmethod
    def from_dict(cls, data: dict) -> CocoCategoriesPanopticSegmentation:
        instance = cls(
            id=data.get("id"),
            name=data.get("name"),
            supercategory=data.get("supercategory"),
            isthing=bool(data.get("isthing")),
            color=data.get("color", []),
        )
        return instance

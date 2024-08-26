from __future__ import annotations

from dataclasses import field

from pytorchcocotools.internal.structure.additional import (
    CocoAnnotationImageCaptioning,
    CocoAnnotationPanopticSegmentation,
    CocoCategoriesPanopticSegmentation,
)
from pytorchcocotools.internal.structure.annotations import (
    CocoAnnotationKeypointDetection,
    CocoAnnotationObjectDetection,
)
from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.internal.structure.categories import (
    CocoCategoriesKeypointDetection,
    CocoCategoriesObjectDetection,
)
from pytorchcocotools.internal.structure.images import CocoImage
from pytorchcocotools.internal.structure.info import CocoInfo
from pytorchcocotools.internal.structure.licenses import CocoLicense
from pytorchcocotools.utils.dataclass import dataclass_dict


# https://cocodataset.org/#format-data
@dataclass_dict
class CocoDetectionDataset(BaseCocoEntity):
    info: CocoInfo = field(default_factory=CocoInfo)
    licenses: list[CocoLicense] = field(default_factory=list[CocoLicense])
    images: list[CocoImage] = field(default_factory=list[CocoImage])
    annotations: list[CocoAnnotationObjectDetection | CocoAnnotationKeypointDetection] = field(
        default_factory=list[CocoAnnotationObjectDetection | CocoAnnotationKeypointDetection]
    )
    categories: list[CocoCategoriesObjectDetection | CocoCategoriesKeypointDetection] = field(
        default_factory=list[CocoCategoriesObjectDetection | CocoCategoriesKeypointDetection]
    )

    @classmethod
    def _get_annotation(cls, annotation: dict) -> CocoAnnotationKeypointDetection | CocoAnnotationObjectDetection:
        if "keypoints" in annotation:
            return CocoAnnotationKeypointDetection.from_dict(annotation)
        return CocoAnnotationObjectDetection.from_dict(annotation)

    @classmethod
    def _get_category(cls, category: dict) -> CocoCategoriesKeypointDetection | CocoCategoriesObjectDetection:
        if "keypoints" in category:
            return CocoCategoriesKeypointDetection.from_dict(category)
        return CocoCategoriesObjectDetection.from_dict(category)

    @classmethod
    def from_dict(cls, data: dict) -> CocoDetectionDataset:
        instance = cls(
            info=CocoInfo.from_dict(data.get("info", {})),
            licenses=[CocoLicense.from_dict(license) for license in data.get("licenses", [])],
            images=[CocoImage.from_dict(image) for image in data.get("images", [])],
            annotations=[cls._get_annotation(annotation) for annotation in data.get("annotations", [])],
            categories=[cls._get_category(category) for category in data.get("categories", [])],
        )
        return instance


@dataclass_dict
class CocoCaptionDataset(CocoDetectionDataset):
    annotations: list[CocoAnnotationImageCaptioning] = field(default_factory=list[CocoAnnotationImageCaptioning])
    categories: list[CocoCategoriesObjectDetection] = field(default_factory=list[CocoCategoriesObjectDetection])

    @classmethod
    def _get_annotation(cls, annotation: dict) -> CocoAnnotationImageCaptioning:
        return CocoAnnotationImageCaptioning.from_dict(annotation)

    @classmethod
    def _get_category(cls, category: dict) -> CocoCategoriesObjectDetection:
        return CocoCategoriesObjectDetection.from_dict(category)


@dataclass_dict
class CocoPanopticDataset(CocoDetectionDataset):
    annotations: list[CocoAnnotationPanopticSegmentation] = field(
        default_factory=list[CocoAnnotationPanopticSegmentation]
    )
    categories: list[CocoCategoriesPanopticSegmentation] = field(
        default_factory=list[CocoCategoriesPanopticSegmentation]
    )

    @classmethod
    def _get_annotation(cls, annotation: dict) -> CocoAnnotationPanopticSegmentation:
        return CocoAnnotationPanopticSegmentation.from_dict(annotation)

    @classmethod
    def _get_category(cls, category: dict) -> CocoCategoriesPanopticSegmentation:
        return CocoCategoriesPanopticSegmentation.from_dict(category)

from dataclasses import field
from datetime import datetime

from dateutil.parser import parse
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoInfo:
    year: int = None
    version: str = None
    description: str = None
    contributor: str = None
    url: str = None
    date_created: datetime = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoInfo":
        date = data.get("date_created")
        return cls(
            year=data.get("year"),
            version=data.get("version"),
            description=data.get("description"),
            contributor=data.get("contributor"),
            url=data.get("url"),
            date_created=parse(date) if date else None,
        )


@dataclass_dict
class CocoLicense:
    id: int = None
    name: str = None
    url: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoLicense":
        return cls(id=data.get("id"), name=data.get("name"), url=data.get("url"))


@dataclass_dict
class CocoImage:
    id: int = None
    width: int = None
    height: int = None
    file_name: str = None
    license: int = None
    date_captured: datetime = None
    flickr_url: str = None
    coco_url: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoImage":
        date = data.get("date_captured")
        return cls(
            id=data.get("id"),
            width=data.get("width"),
            height=data.get("height"),
            file_name=data.get("file_name"),
            license=data.get("license"),
            date_captured=parse(date) if date else None,
            flickr_url=data.get("flickr_url"),
            coco_url=data.get("coco_url"),
        )


@dataclass_dict
class CocoRLE:
    counts: list[int] = field(default_factory=list[float])
    size: list[int] = field(default_factory=list[float])

    @classmethod
    def from_dict(cls, data: dict) -> "CocoRLE":
        return cls(counts=data.get("counts"), size=data.get("size"))


@dataclass_dict
class CocoSegmentInfo:
    id: int = None
    category_id: int = None
    area: float = None
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    iscrowd: bool = None  # 0 or 1

    @classmethod
    def from_dict(cls, data: dict) -> "CocoSegmentInfo":
        return cls(
            id=data.get("id"),
            category_id=data.get("category_id"),
            area=data.get("area"),
            bbox=data.get("bbox"),
            iscrowd=bool(data.get("iscrowd")),
        )


@dataclass_dict
class CocoAnnotationObjectDetection:
    id: int = None
    image_id: int = None
    category_id: int = None
    segmentation: list[CocoRLE | list[float]] = field(default_factory=list[CocoRLE | list[float]])
    area: float = None
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    iscrowd: bool = None  # 0 or 1

    @classmethod
    def from_dict(cls, data: dict) -> "CocoAnnotationObjectDetection":
        segmentations = [CocoRLE.from_dict(seg) if isinstance(seg, dict) else seg for seg in data.get("segmentation")]
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
class CocoCategoriesObjectDetection:
    id: int = None
    name: str = None
    supercategory: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoCategoriesObjectDetection":
        return cls(id=data.get("id"), name=data.get("name"), supercategory=data.get("supercategory"))


@dataclass_dict
class CocoAnnotationKeypointDetection(CocoAnnotationObjectDetection):
    keypoints: list[float] = field(default_factory=list[float])
    num_keypoints: int = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoAnnotationKeypointDetection":
        segmentations = [CocoRLE.from_dict(seg) if isinstance(seg, dict) else seg for seg in data.get("segmentation")]

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


@dataclass_dict
class CocoCategoriesKeypointDetection(CocoCategoriesObjectDetection):
    keypoints: list[str]
    skeleton: list[list[int]]

    @classmethod
    def from_dict(cls, data: dict) -> "CocoCategoriesKeypointDetection":
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            supercategory=data.get("supercategory"),
            keypoints=data.get("keypoints"),
            skeleton=data.get("skeleton"),
        )


@dataclass_dict
class CocoAnnotationImageCaptioning:
    id: int = None
    image_id: int = None
    caption: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "CocoAnnotationImageCaptioning":
        return cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            caption=data.get("caption"),
        )


@dataclass_dict
class CocoAnnotationPanopticSegmentation:
    image_id: int = None
    file_name: str = None
    segments_info: list[CocoSegmentInfo] = field(default_factory=list[CocoSegmentInfo])

    @classmethod
    def from_dict(cls, data: dict) -> "CocoAnnotationPanopticSegmentation":
        return cls(
            image_id=data.get("image_id"),
            file_name=data.get("file_name"),
            segments_info=[CocoSegmentInfo.from_dict(seg) for seg in data.get("segments_info")],
        )


@dataclass_dict
class CocoCategoriesPanopticSegmentation(CocoCategoriesObjectDetection):
    isthing: bool = None  # 0 or 1
    color: list[int] = field(default_factory=list[float])  # [R,G,B]

    @classmethod
    def from_dict(cls, data: dict) -> "CocoCategoriesPanopticSegmentation":
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            supercategory=data.get("supercategory"),
            isthing=bool(data.get("isthing")),
            color=data.get("color"),
        )


@dataclass_dict
class CocoAnnotationDensePose:
    id: int = None
    image_id: int = None
    category_id: int = None
    is_crowd: bool = None  # 0 or 1
    area: float = None
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    dp_I: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_U: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_V: list[float] = field(default_factory=list[float])  # noqa: N815
    dp_x: list[float] = field(default_factory=list[float])
    dp_y: list[float] = field(default_factory=list[float])
    dp_masks: list[CocoRLE] = field(default_factory=list[CocoRLE])

    @classmethod
    def from_dict(cls, data: dict) -> "CocoAnnotationDensePose":
        return cls(
            id=data.get("id"),
            image_id=data.get("image_id"),
            category_id=data.get("category_id"),
            is_crowd=bool(data.get("is_crowd")),
            area=data.get("area"),
            bbox=data.get("bbox"),
            dp_I=data.get("dp_I"),
            dp_U=data.get("dp_U"),
            dp_V=data.get("dp_V"),
            dp_x=data.get("dp_x"),
            dp_y=data.get("dp_y"),
            dp_masks=[CocoRLE.from_dict(mask) for mask in data.get("dp_masks")],
        )


# https://cocodataset.org/#format-data
@dataclass_dict
class CocoDataset:
    info: CocoInfo = field(default_factory=CocoInfo)
    licenses: list[CocoLicense] = field(default_factory=list[CocoLicense])
    images: list[CocoImage] = field(default_factory=list[CocoImage])
    annotations: list[
        CocoAnnotationObjectDetection
        | CocoAnnotationKeypointDetection
        | CocoAnnotationPanopticSegmentation
        | CocoAnnotationDensePose
    ] = field(
        default_factory=list[
            CocoAnnotationObjectDetection
            | CocoAnnotationKeypointDetection
            | CocoAnnotationPanopticSegmentation
            | CocoAnnotationDensePose
        ]
    )
    categories: list[
        CocoCategoriesObjectDetection | CocoCategoriesKeypointDetection | CocoCategoriesPanopticSegmentation
    ] = field(
        default_factory=list[
            CocoCategoriesObjectDetection | CocoCategoriesKeypointDetection | CocoCategoriesPanopticSegmentation
        ]
    )

    @classmethod
    def _get_annotation(cls, annotation: dict) -> CocoAnnotationObjectDetection:
        if "keypoints" in annotation:
            return CocoAnnotationKeypointDetection.from_dict(annotation)
        if "segments_info" in annotation:
            return CocoAnnotationPanopticSegmentation.from_dict(annotation)
        if "dp_masks" in annotation:
            return CocoAnnotationDensePose.from_dict(annotation)
        return CocoAnnotationObjectDetection.from_dict(annotation)

    @classmethod
    def _get_category(cls, category: dict) -> CocoCategoriesObjectDetection:
        if "keypoints" in category:
            return CocoCategoriesKeypointDetection.from_dict(category)
        if "isthing" in category:
            return CocoCategoriesPanopticSegmentation.from_dict(category)
        return CocoCategoriesObjectDetection.from_dict(category)

    @classmethod
    def from_dict(cls, data: dict) -> "CocoDataset":
        return cls(
            info=CocoInfo.from_dict(data.get("info")),
            licenses=[CocoLicense.from_dict(license) for license in data.get("licenses")],
            images=[CocoImage.from_dict(image) for image in data.get("images")],
            annotations=[cls._get_annotation(annotation) for annotation in data.get("annotations")],
            categories=[cls._get_category(category) for category in data.get("categories")],
        )

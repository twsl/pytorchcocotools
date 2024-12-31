from dataclasses import field

from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.internal.structure.rle import CocoRLE
from pytorchcocotools.utils.dataclass import dataclass_dict

# https://cocodataset.org/#format-results


@dataclass_dict
class CocoObjectDetectionResult(BaseCocoEntity):
    image_id: int = -1
    category_id: int = -1
    bbox: list[float] = field(default_factory=list[float])  # [x,y,width,height]
    score: float = 0.0


@dataclass_dict
class CocoInstanceSegmentationnResult(BaseCocoEntity):
    image_id: int = -1
    category_id: int = -1
    segmentation: CocoRLE = field(default_factory=CocoRLE)
    score: float = 0.0


@dataclass_dict
class CocoKeypointDetectionResult(BaseCocoEntity):
    image_id: int = -1
    category_id: int = -1
    keypoints: list[float] = field(default_factory=list[float])
    score: float = 0.0


@dataclass_dict
class CocoStuffSegmentationResult(BaseCocoEntity):
    image_id: int = -1
    category_id: int = -1
    segmentation: CocoRLE = field(default_factory=CocoRLE)


@dataclass_dict
class CocoSegmentInfo(BaseCocoEntity):
    id: int = -1
    category_id: int = -1


@dataclass_dict
class CocoPanopticSegmentationResult(BaseCocoEntity):
    image_id: int = -1
    file_name: str = ""
    segments_info: list[CocoSegmentInfo] = field(default_factory=list[CocoSegmentInfo])


@dataclass_dict
class CocoImageCaptioningResult(BaseCocoEntity):
    image_id: int = -1
    caption: str = ""


@dataclass_dict
class CocoDensePoseResult(CocoObjectDetectionResult):
    uv_shape: tuple[int, int, int] = field(default_factory=tuple[int, int, int])
    uv_data: str = ""

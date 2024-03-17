from __future__ import annotations

from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoCategoriesObjectDetection(BaseCocoEntity):
    id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, data: dict) -> CocoCategoriesObjectDetection:
        return cls(id=data.get("id"), name=data.get("name"), supercategory=data.get("supercategory"))


@dataclass_dict
class CocoCategoriesKeypointDetection(CocoCategoriesObjectDetection):
    keypoints: list[str]
    skeleton: list[list[int]]

    @classmethod
    def from_dict(cls, data: dict) -> CocoCategoriesKeypointDetection:
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            supercategory=data.get("supercategory"),
            keypoints=data.get("keypoints"),
            skeleton=data.get("skeleton"),
        )

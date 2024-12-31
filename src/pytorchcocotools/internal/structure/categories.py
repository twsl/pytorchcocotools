from dataclasses import field
from typing import Self, TypeAlias

from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.utils.dataclass import dataclass_dict


@dataclass_dict
class CocoCategoriesObjectDetection(BaseCocoEntity):
    id: int = -1
    name: str = ""
    supercategory: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        instance = cls(id=data.get("id"), name=data.get("name"), supercategory=data.get("supercategory"))
        return instance


@dataclass_dict
class CocoCategoriesKeypointDetection(CocoCategoriesObjectDetection):
    keypoints: list[str] = field(default_factory=list[str])
    skeleton: list[list[int]] = field(default_factory=list[list[int]])

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        instance = cls(
            id=data.get("id"),
            name=data.get("name"),
            supercategory=data.get("supercategory"),
            keypoints=data.get("keypoints", []),
            skeleton=data.get("skeleton", []),
        )
        return instance


CocoCategoriesDetection: TypeAlias = CocoCategoriesObjectDetection | CocoCategoriesKeypointDetection

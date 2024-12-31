from dataclasses import field
from typing import Self, TypeAlias

from pytorchcocotools.internal.entities import Poly
from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.utils.dataclass import dataclass_dict


@dataclass_dict
class CocoRLE(BaseCocoEntity):
    counts: list[int] = field(default_factory=list[int])
    size: tuple[int, int] = field(default_factory=tuple[int, int])

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        instance = cls(
            counts=data.get("counts", []),
            size=data.get("size", (-1, -1)),
        )
        return instance


Segmentation: TypeAlias = list[Poly] | CocoRLE

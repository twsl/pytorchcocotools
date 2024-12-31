from typing import Self

from pytorchcocotools.internal.structure.base import BaseCocoEntity
from pytorchcocotools.utils.dataclass import dataclass_dict


@dataclass_dict
class CocoLicense(BaseCocoEntity):
    id: int = -1
    name: str = ""
    url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        instance = cls(id=data.get("id"), name=data.get("name"), url=data.get("url"))
        return instance

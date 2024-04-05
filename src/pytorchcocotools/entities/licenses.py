from __future__ import annotations

from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoLicense(BaseCocoEntity):
    id: int = -1
    name: str = ""
    url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> CocoLicense:
        instance = cls(id=data.get("id"), name=data.get("name"), url=data.get("url"))
        return instance

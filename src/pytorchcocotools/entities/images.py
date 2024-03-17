from __future__ import annotations

from datetime import datetime

from dateutil.parser import parse
from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoImage(BaseCocoEntity):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    date_captured: datetime | None
    flickr_url: str
    coco_url: str

    @classmethod
    def from_dict(cls, data: dict) -> CocoImage:
        date = data.get("date_captured")
        return cls(
            id=data.get("id"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            file_name=data.get("file_name"),
            license=data.get("license"),
            date_captured=parse(date) if date else None,
            flickr_url=data.get("flickr_url"),
            coco_url=data.get("coco_url"),
        )

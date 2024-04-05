from __future__ import annotations

from datetime import datetime

from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoImage(BaseCocoEntity):
    id: int = -1
    width: int = 0
    height: int = 0
    file_name: str = ""
    license: int = -1
    date_captured: str = ""
    flickr_url: str = ""
    coco_url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> CocoImage:
        instance = cls(
            id=data.get("id"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            file_name=data.get("file_name"),
            license=data.get("license"),
            date_captured=data.get("date_captured"),
            flickr_url=data.get("flickr_url"),
            coco_url=data.get("coco_url"),
        )
        return instance

    @property
    def datetime_captured(self) -> datetime | None:
        try:
            from dateutil.parser import parse

            return parse(self.date_captured) if self.date_captured else None
        except ImportError:
            return None

from __future__ import annotations

from datetime import datetime

from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoInfo(BaseCocoEntity):
    year: int = datetime.now().year
    version: str = "0.1.0"
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> CocoInfo:
        instance = cls(
            year=data.get("year"),
            version=data.get("version"),
            description=data.get("description"),
            contributor=data.get("contributor"),
            url=data.get("url"),
            date_created=data.get("date_created"),
        )
        return instance

    @property
    def datetime_created(self) -> datetime | None:
        try:
            from dateutil.parser import parse

            return parse(self.date_created) if self.date_created else None
        except ImportError:
            return None

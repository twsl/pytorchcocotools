from __future__ import annotations

from datetime import datetime

from dateutil.parser import parse
from pytorchcocotools.entities.base import BaseCocoEntity
from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class CocoInfo(BaseCocoEntity):
    year: int = datetime.now().year
    version: str = "0.1.0"
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict) -> CocoInfo:
        date = data.get("date_created")
        instance = cls(
            year=data.get("year"),
            version=data.get("version"),
            description=data.get("description"),
            contributor=data.get("contributor"),
            url=data.get("url"),
            date_created=parse(date) if date else None,
        )
        return instance

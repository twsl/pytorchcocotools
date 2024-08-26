from __future__ import annotations

from pytorchcocotools.utils.dataclass import dataclass_dict


@dataclass_dict
class BaseCocoEntity(dict):
    pass

from __future__ import annotations

from pytorchcocotools.utils import dataclass_dict


@dataclass_dict
class BaseCocoEntity(dict):
    pass

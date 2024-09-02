from dataclasses import dataclass, field
from typing import Generic, TypeVar

StorageType = TypeVar("StorageType")


@dataclass
class StageStore(Generic[StorageType]):
    train: StorageType | None = field(default=None)
    val: StorageType | None = field(default=None)
    test: StorageType | None = field(default=None)
    predict: StorageType | None = field(default=None)

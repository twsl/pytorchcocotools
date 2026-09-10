from dataclasses import dataclass, field


@dataclass
class StageStore[StorageType]:
    train: StorageType | None = field(default=None)
    val: StorageType | None = field(default=None)
    test: StorageType | None = field(default=None)
    predict: StorageType | None = field(default=None)

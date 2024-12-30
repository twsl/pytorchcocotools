from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import lightning as L  # noqa: N812
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Transform

from pytorchcocotools.torch.dataset import CocoDetection
from pytorchcocotools.torch.transform import default_transform
from pytorchcocotools.utils.coco.download import CocoDownloader
from pytorchcocotools.utils.logging import get_logger
from pytorchcocotools.utils.stage import StageStore

TransformsType = Callable[[Any], Any] | torch.nn.Module | Transform


class COCODataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        annotation_files: StageStore[str],
        batch_size: int,
        num_workers: int = 2,
        transforms: TransformsType | StageStore[TransformsType] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.annotation_files = annotation_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        transforms = transforms or default_transform()

        self.transforms = (
            StageStore(
                train=transforms,
                val=transforms,
                test=None,
                predict=None,
            )
            if not isinstance(transforms, StageStore)
            else transforms
        )
        self.logger = get_logger(self.__class__.__name__)
        self.download = download

    def prepare_data(self) -> None:
        if self.download:
            self.logger.info("Downloading COCO dataset")
            dl = CocoDownloader(self.root)
            dl.download()
            self.annotation_files = dl.annotation_files

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset: CocoDetection = CocoDetection(
                self.root, self.annotation_files.train or "", transforms=self.transforms.train
            )
        if stage == "validate":
            self.val_dataset: CocoDetection = CocoDetection(
                self.root, self.annotation_files.val or "", transforms=self.transforms.val
            )
        if stage == "test":
            self.test_dataset: CocoDetection = CocoDetection(
                self.root, self.annotation_files.test or "", transforms=self.transforms.test
            )
        if stage == "predict":
            self.predict_dataset: CocoDetection = CocoDetection(
                self.root, self.annotation_files.predict or "", transforms=self.transforms.predict
            )

    def train_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader[CocoDetection]:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx) -> Any:
        # self.trainer.training
        # batch["x"] = transforms(batch["x"])
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx) -> Any:
        # batch["x"] = gpu_transforms(batch["x"])
        return batch
